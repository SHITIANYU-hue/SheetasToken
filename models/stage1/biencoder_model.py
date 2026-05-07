"""
基于 Hugging Face Transformers 的训练框架
用于表格相似度分类任务 (Bi-Encoder 架构)
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# 默认离线，可通过 --allow-download 覆盖
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


class SheetSimilarityDataset(Dataset):
    """表格相似度数据集 (Bi-Encoder)"""

    def __init__(
        self,
        data_dir,
        split="train",
        tokenizer=None,
        max_length=512,
        sample_ratio=1.0,
        max_samples=0,
        stratified_sample=False,
        random_seed=42,
        features_file=None,
        max_header_texts=12,
        include_shape_feature=True,
        include_source_feature=True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.stratified_sample = stratified_sample
        self.random_seed = random_seed
        self.features_file = features_file
        self.max_header_texts = max_header_texts
        self.include_shape_feature = include_shape_feature
        self.include_source_feature = include_source_feature
        self.sheet_feature_map = self._load_sheet_feature_map()
        self.data = self._sample_data(self._load_data())

    def _load_data(self):
        data_file = os.path.join(self.data_dir, f"{self.split}.json")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_features_file(self):
        if self.features_file:
            return self.features_file

        candidates = [
            os.path.join(self.data_dir, "sheets.json"),
            os.path.join(self.data_dir, "sheet_features.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _load_sheet_feature_map(self):
        feature_path = self._infer_features_file()
        if not feature_path:
            return {}

        with open(feature_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        feature_map = {}
        if isinstance(raw, dict):
            for key, item in raw.items():
                if not isinstance(item, dict):
                    continue
                sheet_id = item.get("sheet_id", key)
                if sheet_id is None:
                    continue
                feature_map[str(sheet_id)] = item
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                sheet_id = item.get("sheet_id")
                if sheet_id is None:
                    continue
                feature_map[str(sheet_id)] = item
        else:
            raise ValueError(f"不支持的 sheet features 格式: {type(raw)}")

        print(f"加载 sheet features: {len(feature_map)} from {feature_path}")
        return feature_map

    def _sheet_feature_to_text(self, sheet_id):
        feat = self.sheet_feature_map.get(str(sheet_id))
        if not feat:
            return str(sheet_id)

        segments = []

        name = str(feat.get("name", "")).strip()
        if name:
            segments.append(f"name: {name}")

        if self.include_shape_feature:
            nr = feat.get("num_rows", "?")
            nc = feat.get("num_cols", "?")
            segments.append(f"shape: {nr}x{nc}")

        columns = feat.get("columns", [])
        column_texts = []
        for col in columns[: self.max_header_texts]:
            if isinstance(col, dict):
                txt = str(col.get("name", "")).strip()
            else:
                txt = str(col).strip()
            if txt:
                column_texts.append(txt)
        if column_texts:
            segments.append("headers: " + " | ".join(column_texts))

        if not segments:
            return str(sheet_id)
        return " ; ".join(segments)

    def _sample_data(self, data):
        if not data:
            return data

        total = len(data)
        ratio = min(max(self.sample_ratio, 0.0), 1.0)
        target = total

        if ratio < 1.0:
            target = max(1, int(total * ratio))
        if self.max_samples and self.max_samples > 0:
            target = min(target, self.max_samples)

        if target >= total:
            return data

        rng = random.Random(self.random_seed)
        if not self.stratified_sample:
            return rng.sample(data, target)

        label_to_items = {}
        for item in data:
            label = item.get("label", 0)
            label_to_items.setdefault(label, []).append(item)

        sampled = []
        labels = sorted(label_to_items.keys())
        remaining = target
        remaining_classes = len(labels)

        for label in labels:
            group = label_to_items[label]
            quota = max(1, remaining // remaining_classes)
            pick = min(len(group), quota)
            sampled.extend(rng.sample(group, pick))
            remaining -= pick
            remaining_classes -= 1

        if len(sampled) < target:
            sampled_ids = {id(x) for x in sampled}
            left = [x for x in data if id(x) not in sampled_ids]
            need = target - len(sampled)
            sampled.extend(rng.sample(left, min(need, len(left))))

        rng.shuffle(sampled)
        return sampled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text1 = item.get("sheet1_text", "")
        text2 = item.get("sheet2_text", "")

        if (not text1 or not text2) and ("table1" in item and "table2" in item):
            text1 = self._sheet_feature_to_text(item.get("table1", {}).get("sheet_id", ""))
            text2 = self._sheet_feature_to_text(item.get("table2", {}).get("sheet_id", ""))

        if (not text1 or not text2) and ("sheet_a" in item and "sheet_b" in item):
            text1 = self._sheet_feature_to_text(item.get("sheet_a", ""))
            text2 = self._sheet_feature_to_text(item.get("sheet_b", ""))

        label = item.get("label", 0)

        # Bi-Encoder: 分别对 text1 和 text2 进行 tokenize
        encoding1 = self.tokenizer(
            text1,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        sample = {
            "input_ids1": encoding1["input_ids"].squeeze(0),
            "attention_mask1": encoding1["attention_mask"].squeeze(0),
            "input_ids2": encoding2["input_ids"].squeeze(0),
            "attention_mask2": encoding2["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        
        if "token_type_ids" in encoding1:
            sample["token_type_ids1"] = encoding1["token_type_ids"].squeeze(0)
        if "token_type_ids" in encoding2:
            sample["token_type_ids2"] = encoding2["token_type_ids"].squeeze(0)
            
        return sample


class SimilarityClassifier(nn.Module):
    """支持多种 embedding 策略的分类模型 (Bi-Encoder 架构)。"""

    def __init__(
        self,
        model_name,
        num_labels,
        local_files_only=True,
        embedding_strategy="cls",
        use_layer_mix=False,
        embedding_dropout=0.1,
        head_hidden_dim=0,
        use_extra_position_embedding=False,
        position_embedding_scale=1.0,
        max_length=512,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.embedding_strategy = embedding_strategy
        self.use_layer_mix = use_layer_mix
        self.use_extra_position_embedding = use_extra_position_embedding
        self.position_embedding_scale = position_embedding_scale

        self.backbone = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        hidden_size = self.backbone.config.hidden_size
        self.embedding_dim = self._get_embedding_dim(hidden_size)

        if self.use_extra_position_embedding:
            self.extra_position_embedding = nn.Embedding(max_length, hidden_size)

        if self.use_layer_mix:
            num_layers = self.backbone.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        # Bi-Encoder 组合特征维度: [u, v, |u-v|]
        combined_dim = self.embedding_dim * 3

        self.dropout = nn.Dropout(embedding_dropout)
        if head_hidden_dim and head_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(embedding_dropout),
                nn.Linear(head_hidden_dim, num_labels),
            )
        else:
            self.classifier = nn.Linear(combined_dim, num_labels)

    def _get_embedding_dim(self, hidden_size):
        if self.embedding_strategy in {"cls", "mean", "max"}:
            return hidden_size
        if self.embedding_strategy in {"cls_mean_concat", "mean_max_concat"}:
            return hidden_size * 2
        if self.embedding_strategy == "cls_mean_max_concat":
            return hidden_size * 3
        raise ValueError(f"不支持的 embedding_strategy: {self.embedding_strategy}")

    def _masked_mean_pool(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (sequence_output * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _masked_max_pool(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).bool()
        masked = sequence_output.masked_fill(~mask, -1e9)
        return masked.max(dim=1).values

    def _build_embedding(self, sequence_output, attention_mask):
        cls_emb = sequence_output[:, 0, :]
        mean_emb = self._masked_mean_pool(sequence_output, attention_mask)
        max_emb = self._masked_max_pool(sequence_output, attention_mask)

        if self.embedding_strategy == "cls":
            return cls_emb
        if self.embedding_strategy == "mean":
            return mean_emb
        if self.embedding_strategy == "max":
            return max_emb
        if self.embedding_strategy == "cls_mean_concat":
            return torch.cat([cls_emb, mean_emb], dim=-1)
        if self.embedding_strategy == "mean_max_concat":
            return torch.cat([mean_emb, max_emb], dim=-1)
        if self.embedding_strategy == "cls_mean_max_concat":
            return torch.cat([cls_emb, mean_emb, max_emb], dim=-1)
        raise ValueError(f"不支持的 embedding_strategy: {self.embedding_strategy}")

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, token_type_ids1=None, token_type_ids2=None):
        # 1. 编码第一个文本
        outputs1 = self.backbone(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            output_hidden_states=self.use_layer_mix,
            return_dict=True,
        )

        if self.use_layer_mix:
            hidden_states1 = torch.stack(outputs1.hidden_states, dim=0)
            layer_probs = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            sequence_output1 = (hidden_states1 * layer_probs).sum(dim=0)
        else:
            sequence_output1 = outputs1.last_hidden_state

        if self.use_extra_position_embedding:
            seq_len1 = sequence_output1.size(1)
            position_ids1 = torch.arange(seq_len1, device=sequence_output1.device).unsqueeze(0)
            pos_emb1 = self.extra_position_embedding(position_ids1)
            sequence_output1 = sequence_output1 + self.position_embedding_scale * pos_emb1

        emb1 = self._build_embedding(sequence_output1, attention_mask1)

        # 2. 编码第二个文本
        outputs2 = self.backbone(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            output_hidden_states=self.use_layer_mix,
            return_dict=True,
        )

        if self.use_layer_mix:
            hidden_states2 = torch.stack(outputs2.hidden_states, dim=0)
            # 复用 layer_probs
            sequence_output2 = (hidden_states2 * layer_probs).sum(dim=0)
        else:
            sequence_output2 = outputs2.last_hidden_state

        if self.use_extra_position_embedding:
            seq_len2 = sequence_output2.size(1)
            position_ids2 = torch.arange(seq_len2, device=sequence_output2.device).unsqueeze(0)
            pos_emb2 = self.extra_position_embedding(position_ids2)
            sequence_output2 = sequence_output2 + self.position_embedding_scale * pos_emb2

        emb2 = self._build_embedding(sequence_output2, attention_mask2)

        # 3. 组合特征 (Sentence-BERT 经典组合方式: [u, v, |u-v|])
        diff = torch.abs(emb1 - emb2)
        combined_emb = torch.cat([emb1, emb2, diff], dim=-1)

        # 4. 分类预测
        logits = self.classifier(self.dropout(combined_emb))
        return logits


class TransformerTrainer:
    """Transformer 分类模型训练器"""

    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=2,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=3,
        warmup_steps=500,
        max_length=512,
        device=None,
        local_files_only=True,
        freeze_backbone=False,
        unfreeze_last_n_layers=0,
        label_smoothing=0.0,
        max_grad_norm=1.0,
        embedding_strategy="cls",
        use_layer_mix=False,
        embedding_dropout=0.1,
        head_hidden_dim=0,
        use_extra_position_embedding=False,
        position_embedding_scale=1.0,
        train_sample_ratio=1.0,
        train_max_samples=0,
        train_stratified_sample=False,
        eval_sample_ratio=1.0,
        eval_max_samples=0,
        eval_stratified_sample=False,
        sample_seed=42,
        train_features_file=None,
        eval_features_file=None,
        max_header_texts=12,
        include_shape_feature=True,
        include_source_feature=True,
        use_tensorboard=False,
        tensorboard_logdir="runs/sheet_similarity",
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.label_smoothing = label_smoothing
        self.max_grad_norm = max_grad_norm
        self.embedding_strategy = embedding_strategy
        self.use_layer_mix = use_layer_mix
        self.embedding_dropout = embedding_dropout
        self.head_hidden_dim = head_hidden_dim
        self.use_extra_position_embedding = use_extra_position_embedding
        self.position_embedding_scale = position_embedding_scale
        self.train_sample_ratio = train_sample_ratio
        self.train_max_samples = train_max_samples
        self.train_stratified_sample = train_stratified_sample
        self.eval_sample_ratio = eval_sample_ratio
        self.eval_max_samples = eval_max_samples
        self.eval_stratified_sample = eval_stratified_sample
        self.sample_seed = sample_seed
        self.train_features_file = train_features_file
        self.eval_features_file = eval_features_file
        self.max_header_texts = max_header_texts
        self.include_shape_feature = include_shape_feature
        self.include_source_feature = include_source_feature
        self.use_tensorboard = use_tensorboard
        self.tensorboard_logdir = tensorboard_logdir
        self.tb_writer = None

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.local_files_only:
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            os.environ["HF_DATASETS_OFFLINE"] = "0"

        print(f"使用设备: {self.device}")
        print(f"模型: {self.model_name}")
        print(f"仅本地加载: {self.local_files_only}")
        print(f"embedding策略: {self.embedding_strategy}")
        print(f"层混合: {self.use_layer_mix}")
        print(f"额外位置编码: {self.use_extra_position_embedding}")
        print(f"TensorBoard: {self.use_tensorboard}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=self.local_files_only,
        )
        self.model = SimilarityClassifier(
            model_name=model_name,
            num_labels=num_labels,
            local_files_only=self.local_files_only,
            embedding_strategy=self.embedding_strategy,
            use_layer_mix=self.use_layer_mix,
            embedding_dropout=self.embedding_dropout,
            head_hidden_dim=self.head_hidden_dim,
            use_extra_position_embedding=self.use_extra_position_embedding,
            position_embedding_scale=self.position_embedding_scale,
            max_length=self.max_length,
        )

        self._configure_trainable_layers()
        self.model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def _calculate_normalized_entropy(self, logits):
        """计算标准化熵: H_norm = -Σ(p_i * log(p_i)) / log(num_labels)
        范围: [0, 1]，0表示完全确定，1表示最大不确定性
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(self.num_labels, dtype=torch.float32, device=logits.device))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        return normalized_entropy.mean().item()

    def _configure_trainable_layers(self):
        if not self.freeze_backbone:
            return

        base_model = self.model.backbone
        for param in base_model.parameters():
            param.requires_grad = False

        if self.unfreeze_last_n_layers > 0:
            if hasattr(base_model, "encoder") and hasattr(base_model.encoder, "layer"):
                layers = base_model.encoder.layer
                n = min(self.unfreeze_last_n_layers, len(layers))
                for layer in layers[-n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"已解冻最后 {n} 层 encoder")
            else:
                print("当前模型不支持按层解冻，保持 backbone 冻结")

        for param in self.model.classifier.parameters():
            param.requires_grad = True
        if hasattr(self.model, "layer_weights"):
            self.model.layer_weights.requires_grad = True

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ratio = 100.0 * trainable_params / max(total_params, 1)
        print(f"可训练参数: {trainable_params}/{total_params} ({ratio:.2f}%)")

    def prepare_data(self, data_dir):
        full_dataset = SheetSimilarityDataset(
            data_dir=data_dir,
            split="train",
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            sample_ratio=1.0,
            max_samples=0,
            stratified_sample=False,
            random_seed=self.sample_seed,
            features_file=self.train_features_file or self.eval_features_file,
            max_header_texts=self.max_header_texts,
            include_shape_feature=self.include_shape_feature,
            include_source_feature=self.include_source_feature,
        )

        total_size = len(full_dataset)
        if total_size < 2:
            raise ValueError("train.json 样本数太少，至少需要 2 条样本")

        rng = random.Random(self.sample_seed)
        indices = list(range(total_size))
        rng.shuffle(indices)

        eval_size = max(1, int(total_size * 0.1))
        if eval_size >= total_size:
            eval_size = 1
        train_size = total_size - eval_size

        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]

        def _apply_subset_sampling(base_indices, sample_ratio, max_samples, stratified_sample):
            if not base_indices:
                return base_indices

            target = len(base_indices)
            ratio = min(max(sample_ratio, 0.0), 1.0)
            if ratio < 1.0:
                target = max(1, int(len(base_indices) * ratio))
            if max_samples and max_samples > 0:
                target = min(target, max_samples)

            if target >= len(base_indices):
                return base_indices

            local_rng = random.Random(self.sample_seed)
            if not stratified_sample:
                return local_rng.sample(base_indices, target)

            label_to_indices = {}
            for idx in base_indices:
                label = int(full_dataset.data[idx].get("label", 0))
                label_to_indices.setdefault(label, []).append(idx)

            sampled = []
            labels = sorted(label_to_indices.keys())
            remaining = target
            remaining_classes = len(labels)

            for label in labels:
                group = label_to_indices[label]
                quota = max(1, remaining // remaining_classes)
                pick = min(len(group), quota)
                sampled.extend(local_rng.sample(group, pick))
                remaining -= pick
                remaining_classes -= 1

            if len(sampled) < target:
                sampled_set = set(sampled)
                left = [x for x in base_indices if x not in sampled_set]
                need = target - len(sampled)
                sampled.extend(local_rng.sample(left, min(need, len(left))))

            local_rng.shuffle(sampled)
            return sampled

        train_indices = _apply_subset_sampling(
            train_indices,
            self.train_sample_ratio,
            self.train_max_samples,
            self.train_stratified_sample,
        )
        eval_indices = _apply_subset_sampling(
            eval_indices,
            self.eval_sample_ratio,
            self.eval_max_samples,
            self.eval_stratified_sample,
        )

        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, eval_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler, global_step=0, tb_writer=None):
        self.model.train()
        total_loss = 0.0
        total_entropy = 0.0
        correct = 0
        total = 0
        batch_count = 0
        total_samples = 0
        total_steps = 0
        epoch_start = time.time()

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            step_start = time.time()
            
            labels = batch["labels"].to(self.device)
            model_inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k != "labels"
            }
            batch_size = labels.size(0)

            outputs = self.model(**model_inputs)
            logits = outputs
            loss = self.loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()
            scheduler.step()

            step_time = time.time() - step_start
            samples_per_sec = batch_size / step_time if step_time > 0 else 0.0

            total_loss += loss.item()
            total_entropy += self._calculate_normalized_entropy(logits)
            batch_count += 1
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_samples += batch_size
            total_steps += 1

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_step", loss.item(), global_step)
                tb_writer.add_scalar("train/step_time_sec", step_time, global_step)
                tb_writer.add_scalar("train/samples_per_sec", samples_per_sec, global_step)

            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": correct / max(total, 1),
                "entropy": total_entropy / batch_count,
                "samples/sec": f"{samples_per_sec:.1f}"
            })
            
            global_step += 1

        avg_loss = total_loss / len(train_loader)
        avg_entropy = total_entropy / batch_count
        accuracy = correct / max(total, 1)
        epoch_time = time.time() - epoch_start
        avg_steps_per_sec = total_steps / epoch_time if epoch_time > 0 else 0.0
        avg_samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0.0
        
        return avg_loss, accuracy, avg_entropy, epoch_time, avg_steps_per_sec, avg_samples_per_sec, global_step

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        total_entropy = 0.0
        correct = 0
        total = 0
        batch_count = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                labels = batch["labels"].to(self.device)
                model_inputs = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "labels"
                }

                outputs = self.model(**model_inputs)
                logits = outputs
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                total_entropy += self._calculate_normalized_entropy(logits)
                batch_count += 1
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(test_loader)
        avg_entropy = total_entropy / batch_count
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, avg_entropy

    def train(self, data_dir, wandb_project="sheet-bert", wandb_run_name=None):
        run_name = wandb_run_name or "run"
        wandb.init(
            mode="offline",
            project=wandb_project,
            name=run_name,
            config={
                "model_name": self.model_name,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "max_length": self.max_length,
                "freeze_backbone": self.freeze_backbone,
                "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
                "label_smoothing": self.label_smoothing,
                "max_grad_norm": self.max_grad_norm,
                "embedding_strategy": self.embedding_strategy,
                "use_layer_mix": self.use_layer_mix,
                "embedding_dropout": self.embedding_dropout,
                "head_hidden_dim": self.head_hidden_dim,
                "use_extra_position_embedding": self.use_extra_position_embedding,
                "position_embedding_scale": self.position_embedding_scale,
                "train_sample_ratio": self.train_sample_ratio,
                "train_max_samples": self.train_max_samples,
                "train_stratified_sample": self.train_stratified_sample,
                "eval_sample_ratio": self.eval_sample_ratio,
                "eval_max_samples": self.eval_max_samples,
                "eval_stratified_sample": self.eval_stratified_sample,
                "sample_seed": self.sample_seed,
                "train_features_file": self.train_features_file,
                "eval_features_file": self.eval_features_file,
                "max_header_texts": self.max_header_texts,
                "include_shape_feature": self.include_shape_feature,
                "include_source_feature": self.include_source_feature,
                "use_tensorboard": self.use_tensorboard,
                "tensorboard_logdir": self.tensorboard_logdir,
            },
        )

        if self.use_tensorboard:
            tb_dir = os.path.join(self.tensorboard_logdir, run_name)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.tb_writer.add_text("run/model_name", self.model_name)
            self.tb_writer.add_text("run/embedding_strategy", self.embedding_strategy)
            self.tb_writer.add_text("run/data_dir", data_dir)
            self.tb_writer.add_text("run/tensorboard_dir", tb_dir)

        print("准备数据...")
        train_loader, test_loader = self.prepare_data(data_dir)
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")

        optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.learning_rate,
        )
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        best_accuracy = 0.0
        global_step = 0
        for epoch in range(self.num_epochs):
            epoch_idx = epoch + 1
            print(f"\nEpoch {epoch_idx}/{self.num_epochs}")

            train_loss, train_acc, train_entropy, epoch_time, avg_steps_per_sec, avg_samples_per_sec, global_step = self.train_epoch(
                train_loader, optimizer, scheduler, global_step=global_step, tb_writer=self.tb_writer
            )
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 训练熵: {train_entropy:.4f}")
            print(f"  耗时: {epoch_time:.2f}s, 步数吞吐: {avg_steps_per_sec:.2f} steps/s, 样本吞吐: {avg_samples_per_sec:.2f} samples/s")

            test_loss, test_acc, test_entropy = self.evaluate(test_loader)
            print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}, 测试熵: {test_entropy:.4f}")

            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_entropy": train_entropy,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "test_entropy": test_entropy,
                    "train_epoch_time_sec": epoch_time,
                    "train_avg_steps_per_sec": avg_steps_per_sec,
                    "train_avg_samples_per_sec": avg_samples_per_sec,
                    "lr": current_lr,
                }
            )

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train", train_loss, epoch_idx)
                self.tb_writer.add_scalar("loss/eval", test_loss, epoch_idx)
                self.tb_writer.add_scalar("accuracy/train", train_acc, epoch_idx)
                self.tb_writer.add_scalar("accuracy/eval", test_acc, epoch_idx)
                self.tb_writer.add_scalar("entropy/train", train_entropy, epoch_idx)
                self.tb_writer.add_scalar("entropy/eval", test_entropy, epoch_idx)
                self.tb_writer.add_scalar("timing/epoch_time_sec", epoch_time, epoch_idx)
                self.tb_writer.add_scalar("timing/avg_steps_per_sec", avg_steps_per_sec, epoch_idx)
                self.tb_writer.add_scalar("timing/avg_samples_per_sec", avg_samples_per_sec, epoch_idx)
                self.tb_writer.add_scalar("optimizer/lr", current_lr, epoch_idx)

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_model("best_model")
                print(f"保存最佳模型，准确率: {best_accuracy:.4f}")
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("best/eval_accuracy", best_accuracy, epoch_idx)
                    self.tb_writer.add_text("checkpoint/best_model", os.path.abspath("best_model"), epoch_idx)

        self.save_model("final_model")
        wandb.save("best_model/*")
        wandb.save("final_model/*")

        if self.tb_writer is not None:
            self.tb_writer.add_text("checkpoint/final_model", os.path.abspath("final_model"), self.num_epochs)
            self.tb_writer.add_hparams(
                {
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                    "max_length": self.max_length,
                    "embedding_strategy": self.embedding_strategy,
                    "label_smoothing": self.label_smoothing,
                    "max_grad_norm": self.max_grad_norm,
                },
                {
                    "hparam/best_eval_accuracy": best_accuracy,
                    "hparam/best_eval_entropy": test_entropy,
                },
            )
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

        wandb.finish()
        print(f"\n训练完成！最佳准确率: {best_accuracy:.4f}")

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.model.backbone.save_pretrained(os.path.join(save_dir, "backbone"))
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "embedding_strategy": self.embedding_strategy,
                "use_layer_mix": self.use_layer_mix,
                "embedding_dropout": self.embedding_dropout,
                "head_hidden_dim": self.head_hidden_dim,
                "use_extra_position_embedding": self.use_extra_position_embedding,
                "position_embedding_scale": self.position_embedding_scale,
            },
            os.path.join(save_dir, "classifier.pt"),
        )
        self.tokenizer.save_pretrained(save_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="训练/测试表格相似度分类模型 (Bi-Encoder)")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="数据目录，应包含 train.json 和 sheets.json",
    )
    parser.add_argument(
        "--model-name",
        default="local_models/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        help="骨干模型名称或本地路径",
    )
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--wandb-project", default="sheet-similarity-bert")
    parser.add_argument("--wandb-run-name", default="backbone-exp")

    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允许在线下载模型（默认仅本地加载）",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="冻结 backbone，仅训练分类头",
    )
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        default=0,
        help="在 freeze-backbone 开启时，解冻最后 N 层 encoder",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="交叉熵 label smoothing，建议 0.0~0.1",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值，<=0 表示关闭",
    )
    parser.add_argument(
        "--embedding-strategy",
        choices=[
            "cls",
            "mean",
            "max",
            "cls_mean_concat",
            "mean_max_concat",
            "cls_mean_max_concat",
        ],
        default="cls",
        help="句向量构造方式",
    )
    parser.add_argument(
        "--use-layer-mix",
        action="store_true",
        help="对所有层 hidden states 做可学习加权融合",
    )
    parser.add_argument(
        "--embedding-dropout",
        type=float,
        default=0.1,
        help="embedding 后 dropout 比例",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=0,
        help="分类头隐藏层维度，0 表示单层线性分类头",
    )
    parser.add_argument(
        "--use-extra-position-embedding",
        action="store_true",
        help="在 backbone 输出上叠加额外可训练位置编码",
    )
    parser.add_argument(
        "--position-embedding-scale",
        type=float,
        default=1.0,
        help="额外位置编码缩放系数",
    )
    parser.add_argument(
        "--train-sample-ratio",
        type=float,
        default=1.0,
        help="训练集采样比例，范围 0~1",
    )
    parser.add_argument(
        "--train-max-samples",
        type=int,
        default=0,
        help="训练集最大样本数，0 表示不限制",
    )
    parser.add_argument(
        "--train-stratified-sample",
        action="store_true",
        help="训练集按 label 分层采样",
    )
    parser.add_argument(
        "--eval-sample-ratio",
        type=float,
        default=1.0,
        help="验证集采样比例，范围 0~1",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help="验证集最大样本数，0 表示不限制",
    )
    parser.add_argument(
        "--eval-stratified-sample",
        action="store_true",
        help="验证集按 label 分层采样",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="采样随机种子",
    )
    parser.add_argument(
        "--train-features-file",
        default=None,
        help="训练集 sheet_features JSON 路径，默认自动推断",
    )
    parser.add_argument(
        "--eval-features-file",
        default=None,
        help="验证集 sheet_features JSON 路径，默认自动推断",
    )
    parser.add_argument(
        "--max-header-texts",
        type=int,
        default=12,
        help="构造 sheet 文本时最多拼接多少个 header",
    )
    parser.add_argument(
        "--disable-shape-feature",
        action="store_true",
        help="不拼接 num_rows/num_cols 形状特征",
    )
    parser.add_argument(
        "--disable-source-feature",
        action="store_true",
        help="不拼接 source 特征",
    )
    parser.add_argument(
        "--use-tensorboard",
        action="store_true",
        help="启用 TensorBoard 日志记录",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default="runs/sheet_similarity",
        help="TensorBoard 日志根目录",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    trainer = TransformerTrainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        local_files_only=not args.allow_download,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        embedding_strategy=args.embedding_strategy,
        use_layer_mix=args.use_layer_mix,
        embedding_dropout=args.embedding_dropout,
        head_hidden_dim=args.head_hidden_dim,
        use_extra_position_embedding=args.use_extra_position_embedding,
        position_embedding_scale=args.position_embedding_scale,
        train_sample_ratio=args.train_sample_ratio,
        train_max_samples=args.train_max_samples,
        train_stratified_sample=args.train_stratified_sample,
        eval_sample_ratio=args.eval_sample_ratio,
        eval_max_samples=args.eval_max_samples,
        eval_stratified_sample=args.eval_stratified_sample,
        sample_seed=args.sample_seed,
        train_features_file=args.train_features_file,
        eval_features_file=args.eval_features_file,
        max_header_texts=args.max_header_texts,
        include_shape_feature=not args.disable_shape_feature,
        include_source_feature=not args.disable_source_feature,
        use_tensorboard=args.use_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
    )

    trainer.train(
        data_dir=args.data_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
