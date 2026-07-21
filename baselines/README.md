# Zero-shot baselines

These experiments evaluate retrieval without training on the repository's pairwise or query supervision.
Both methods search the complete `data/sheets.json` corpus and use the same sheet serialization and metrics.

## 1. Frozen embedding retrieval

The default model is `BAAI/bge-base-en-v1.5`. Sheet embeddings are computed once, query embeddings use the model's retrieval instruction, and sheets are ranked by cosine similarity.

```bash
pip install -r requirements.txt
bash scripts/baselines/run_embedding.sh
```

Validate the inputs without downloading a model:

```bash
python -m baselines.embedding_retrieval --dry-run
```

Override the model or output path:

```bash
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
OUTPUT=outputs/baselines/embedding_minilm.json \
bash scripts/baselines/run_embedding.sh
```

When changing away from BGE, pass an appropriate `--query-prefix` (an empty string is allowed).

## 2. Full-corpus LLM selector

The LLM receives the query and all sheet descriptions, then returns a ranked list of valid sheet IDs through Structured Outputs. Set an API key before the paid run:

```bash
export OPENAI_API_KEY=...
bash scripts/baselines/run_llm.sh
```

The launcher enables `--resume`, so completed queries in the output file are not billed again. Inspect prompt size without making any API request:

```bash
python -m baselines.llm_selector --dry-run
```

For a cheap smoke test before the full 134-query run:

```bash
python -m baselines.llm_selector \
  --max-queries 1 \
  --output outputs/baselines/llm_smoke.json
```

### Local Ollama

The same full-corpus selector can run without API cost through Ollama:

```bash
ollama pull qwen2.5:1.5b-instruct-q4_K_M
bash scripts/baselines/run_ollama.sh
```

The default Ollama smoke/evaluation launcher uses the explicitly quantized Q4_K_M 1.5B model and the positive-plus-explicit-negative candidate set from each query (about 25 sheets rather than the complete 614-sheet corpus). It uses native JSON Schema output and an 8,192-token context window. This controlled candidate experiment must be reported separately from the full-corpus LLM baseline. Override `MODEL`, `OUTPUT`, or `OLLAMA_HOST` through environment variables.

## Outputs and metrics

Each JSON report contains the experiment configuration, per-query predictions, runtime, and macro-averaged:

- Precision@1/3/5/10
- Recall@1/3/5/10
- HitRate@1/3/5/10
- MRR@1/3/5/10
- nDCG@1/3/5/10

The LLM report also records input, cached-input, output, and total token usage. Failed API queries remain in the report as empty predictions so failures cannot silently improve aggregate metrics.
