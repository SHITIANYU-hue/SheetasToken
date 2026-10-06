"""Gradient-cache replay must match ordinary differentiation with dropout."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA replay implementation')
def test_gradient_cache_matches_direct_dropout_gradients():
    path=Path(__file__).resolve().parents[1]/'scripts/experiments/train_representation.py'
    import sys
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('representation_train',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Batch(dict):
        def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})
    class Tokenizer:
        def __call__(self,texts,**kwargs):return Batch(input_ids=torch.tensor([[int(t),2,3] for t in texts]))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(30,16);self.drop=torch.nn.Dropout(.3);self.proj=torch.nn.Linear(16,16)
        def forward(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=self.proj(self.drop(self.emb(input_ids))))
    torch.manual_seed(7);device=torch.device('cuda',0);model=Model().to(device).train();tok=Tokenizer();texts=[str(i) for i in range(1,13)];micro=4
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();direct=[]
    for j in range(0,len(texts),micro):
        with torch.autocast(device_type='cuda',dtype=torch.float16):direct.append(F.normalize(model(**tok(texts[j:j+micro]).to(device)).last_hidden_state[:,0].float(),dim=-1))
    direct=torch.cat(direct);loss=F.cross_entropy(direct[:4]@direct[4:].T/.2,torch.arange(4,device=device));loss.backward();expected=[p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad();torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
    output,states=m.encode(model,tok,texts,device,micro,True);output.requires_grad_();cached_loss=F.cross_entropy(output[:4]@output[4:].T/.2,torch.arange(4,device=device));cached_loss.backward();m.replay(model,tok,texts,output.grad,states,device,micro)
    assert torch.allclose(loss,cached_loss,atol=1e-6)
    for p,g in zip(model.parameters(),expected):assert torch.allclose(p.grad,g,atol=2e-3,rtol=2e-3)
