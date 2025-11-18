#!/usr/bin/env python3
"""
Bootstrap SPC 检测器训练（CodeBERT + LoRA + 句子对分类）
输入：bootstrap_filtered_60.json
输出：bootstrap_detector.pt + 评估报告
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from pathlib import Path


class SPCDataset(Dataset):
    """60 对 SPC：before vs after，标签 0=patched，1=vulnerable"""
    def __init__(self, json_path, tokenizer, max_len=256):
        with open(json_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"before: {item['before_source']}\nafter: {item['after_source']}"
        label = int(item['label'])  # 0 = patched, 1 = vulnerable
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch['label']).sum().item()
            total += batch['label'].size(0)
    return correct / total


def train(json_path, model_name, output_dir, epochs=10, batch_size=8, lr=5e-5, lora_r=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = SPCDataset(json_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 基础模型：句子对分类（2 类）
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(loader) * epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        acc = evaluate(model, loader, device)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch+1} loss: {total_loss / len(loader):.4f} | Acc: {acc:.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/bootstrap_detector.pt")
    print(f"✅ Bootstrap 训练完成 | 保存: {output_dir}/bootstrap_detector.pt | Best Acc: {best_acc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bootstrap SPC 检测器训练（CodeBERT + LoRA）")
    parser.add_argument("--spc", default="data/spc_data/processed/bootstrap_filtered_60.json", help="SPC JSON 文件")
    parser.add_argument("--model", default="microsoft/codebert-base", help="预训练模型")
    parser.add_argument("--output", default="data/bootstrap", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    args = parser.parse_args()

    train(args.spc, args.model, args.output, args.epochs, args.batch, args.lr, args.lora_r)