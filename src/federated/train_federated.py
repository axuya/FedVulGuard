#!/usr/bin/env python3
"""
联邦学习训练 - 模拟 5 个客户端
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import copy
import json
import numpy as np
from collections import defaultdict

# 导入 MGVD 模型
from src.models.train_mgvd import MGVDModel, load_graph_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FederatedTrainer:
    """联邦学习训练器"""
    
    def __init__(self, num_clients=5, rounds=20):
        self.num_clients = num_clients
        self.rounds = rounds
        self.global_model = MGVDModel().to(device)
        
    def split_data_non_iid(self, data, num_clients):
        """Non-IID 数据划分"""
        # 按标签分组
        label_groups = defaultdict(list)
        for item in data:
            label_groups[item['label']].append(item)
        
        # 为每个客户端分配数据（不平衡）
        client_data = [[] for _ in range(num_clients)]
        
        for label, items in label_groups.items():
            # 随机打乱
            np.random.shuffle(items)
            
            # Dirichlet 分布模拟 Non-IID
            proportions = np.random.dirichlet([0.5] * num_clients)
            splits = (np.cumsum(proportions) * len(items)).astype(int)
            
            start = 0
            for i, end in enumerate(splits):
                client_data[i].extend(items[start:end])
                start = end
        
        return client_data
    
    def local_train(self, model, data, epochs=3):
        """客户端本地训练"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(epochs):
            for item in data:
                optimizer.zero_grad()
                
                graphs = item['graphs']
                label = torch.tensor([item['label']], dtype=torch.long).to(device)
                
                batch_ast = torch.zeros(graphs['ast'].x.size(0), dtype=torch.long).to(device)
                batch_cfg = torch.zeros(graphs['cfg'].x.size(0), dtype=torch.long).to(device)
                batch_dfg = torch.zeros(graphs['dfg'].x.size(0), dtype=torch.long).to(device)
                batch_pdg = torch.zeros(graphs['pdg'].x.size(0), dtype=torch.long).to(device)
                
                for g_type in graphs:
                    graphs[g_type] = graphs[g_type].to(device)
                
                out = model(graphs['ast'], graphs['cfg'], graphs['dfg'], graphs['pdg'],
                           (batch_ast, batch_cfg, batch_dfg, batch_pdg))
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def fedavg(self, client_weights, client_sizes):
        """FedAvg 聚合"""
        avg_weights = {}
        total_size = sum(client_sizes)
        
        for key in client_weights[0].keys():
            avg_weights[key] = sum(
                w[key] * size / total_size 
                for w, size in zip(client_weights, client_sizes)
            )
        
        return avg_weights
    
    def train(self):
        """联邦训练主流程"""
        print("="*70)
        print("🌐 Federated Learning Training")
        print("="*70)
        
        # 加载数据
        print("\n📦 Loading data...")
        index_file = Path("data/graphs/main_dataset/main_dataset_index.json")
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        vuln_to_idx = {
            'reentrancy': 0, 'overflow': 1, 'access_control': 2,
            'tx_origin': 3, 'timestamp': 4, 'unchecked_call': 5, 'unknown': 6
        }
        
        dataset = []
        for item in index[:50]:  # 使用 50 个样本
            try:
                graphs = load_graph_data(item['graph_path'])
                dataset.append({
                    'graphs': graphs,
                    'label': vuln_to_idx.get(item['vulnerability_type'], 6)
                })
            except:
                pass
        
        # Non-IID 划分
        print(f"📊 Splitting data into {self.num_clients} clients (Non-IID)...")
        client_datasets = self.split_data_non_iid(dataset, self.num_clients)
        
        for i, data in enumerate(client_datasets):
            print(f"   Client {i}: {len(data)} samples")
        
        # 联邦训练
        print(f"\n🔥 Starting {self.rounds} rounds of federated training...")
        
        for round_idx in range(self.rounds):
            print(f"\n--- Round {round_idx + 1}/{self.rounds} ---")
            
            client_weights = []
            client_sizes = []
            
            # 每个客户端本地训练
            for client_id in range(self.num_clients):
                # 复制全局模型
                local_model = copy.deepcopy(self.global_model)
                
                # 本地训练
                weights = self.local_train(
                    local_model, 
                    client_datasets[client_id],
                    epochs=3
                )
                
                client_weights.append(weights)
                client_sizes.append(len(client_datasets[client_id]))
            
            # 聚合
            global_weights = self.fedavg(client_weights, client_sizes)
            self.global_model.load_state_dict(global_weights)
            
            # 评估
            if (round_idx + 1) % 5 == 0:
                acc = self.evaluate(dataset[:10])
                print(f"   Global Model Accuracy: {acc:.4f}")
        
        # 保存模型
        torch.save(self.global_model.state_dict(), 'models/federated_model.pth')
        print("\n✅ Federated training complete!")
        print("📁 Model saved: models/federated_model.pth")
    
    def evaluate(self, data):
        """评估全局模型"""
        self.global_model.eval()
        correct = 0
        
        with torch.no_grad():
            for item in data:
                graphs = item['graphs']
                label = torch.tensor([item['label']], dtype=torch.long).to(device)
                
                batch_ast = torch.zeros(graphs['ast'].x.size(0), dtype=torch.long).to(device)
                batch_cfg = torch.zeros(graphs['cfg'].x.size(0), dtype=torch.long).to(device)
                batch_dfg = torch.zeros(graphs['dfg'].x.size(0), dtype=torch.long).to(device)
                batch_pdg = torch.zeros(graphs['pdg'].x.size(0), dtype=torch.long).to(device)
                
                for g_type in graphs:
                    graphs[g_type] = graphs[g_type].to(device)
                
                out = self.global_model(graphs['ast'], graphs['cfg'], graphs['dfg'], graphs['pdg'],
                                       (batch_ast, batch_cfg, batch_dfg, batch_pdg))
                pred = out.argmax(dim=1)
                correct += (pred == label).sum().item()
        
        return correct / len(data) if len(data) > 0 else 0


def main():
    trainer = FederatedTrainer(num_clients=5, rounds=20)
    trainer.train()


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()