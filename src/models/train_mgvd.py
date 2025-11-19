#!/usr/bin/env python3
"""
MGVD 模型训练 - 多图漏洞检测
使用 GNN 编码 AST/CFG/DFG/PDG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import networkx as nx

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MGVDModel(nn.Module):
    """多图漏洞检测模型"""
    
    def __init__(self, node_dim=64, hidden_dim=128, num_classes=7):
        super().__init__()
        
        # 为每种图类型定义编码器
        # AST 编码器 (GIN)
        self.ast_conv1 = GINConv(nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.ast_conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
        # CFG 编码器 (GAT)
        self.cfg_conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=False)
        self.cfg_conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # DFG 编码器 (GIN)
        self.dfg_conv1 = GINConv(nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.dfg_conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
        # PDG 编码器 (GAT)
        self.pdg_conv1 = GATConv(node_dim, hidden_dim, heads=4, concat=False)
        self.pdg_conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # SENet 注意力机制
        self.se_fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.se_fc2 = nn.Linear(hidden_dim, hidden_dim * 4)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, ast_data, cfg_data, dfg_data, pdg_data, batches):
        # 解包 batch
        batch_ast, batch_cfg, batch_dfg, batch_pdg = batches
        
        # 编码各个图 - 处理空图
        if ast_data.x.size(0) > 0:
            ast_x = F.relu(self.ast_conv1(ast_data.x, ast_data.edge_index))
            ast_x = F.relu(self.ast_conv2(ast_x, ast_data.edge_index))
            ast_emb = global_mean_pool(ast_x, batch_ast)
        else:
            ast_emb = torch.zeros(1, self.ast_conv1.nn[0].out_features).to(ast_data.x.device)
        
        if cfg_data.x.size(0) > 0:
            cfg_x = F.relu(self.cfg_conv1(cfg_data.x, cfg_data.edge_index))
            cfg_x = F.relu(self.cfg_conv2(cfg_x, cfg_data.edge_index))
            cfg_emb = global_mean_pool(cfg_x, batch_cfg)
        else:
            cfg_emb = torch.zeros(1, 128).to(cfg_data.x.device)
        
        if dfg_data.x.size(0) > 0:
            dfg_x = F.relu(self.dfg_conv1(dfg_data.x, dfg_data.edge_index))
            dfg_x = F.relu(self.dfg_conv2(dfg_x, dfg_data.edge_index))
            dfg_emb = global_mean_pool(dfg_x, batch_dfg)
        else:
            dfg_emb = torch.zeros(1, 128).to(dfg_data.x.device)
        
        if pdg_data.x.size(0) > 0:
            pdg_x = F.relu(self.pdg_conv1(pdg_data.x, pdg_data.edge_index))
            pdg_x = F.relu(self.pdg_conv2(pdg_x, pdg_data.edge_index))
            pdg_emb = global_mean_pool(pdg_x, batch_pdg)
        else:
            pdg_emb = torch.zeros(1, 128).to(pdg_data.x.device)
        
        # 拼接
        combined = torch.cat([ast_emb, cfg_emb, dfg_emb, pdg_emb], dim=1)
        
        # SENet 注意力
        se_weight = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(combined))))
        combined = combined * se_weight
        
        # 分类
        out = self.classifier(combined)
        return out


def load_graph_data(graph_file):
    """加载图数据并转换为 PyG 格式"""
    with open(graph_file, 'r') as f:
        data = json.load(f)
    
    pyg_graphs = {}
    
    for graph_type in ['ast', 'cfg', 'dfg', 'pdg']:
        g_data = data[graph_type]
        
        # 节点特征 (简化：使用 one-hot 编码)
        num_nodes = len(g_data['nodes'])
        x = torch.randn(num_nodes, 64)  # 随机特征，实际应该用真实特征
        
        # 边索引
        edge_index = []
        for link in g_data['links']:
            src = int(link['source'].split('_')[-1]) if '_' in str(link['source']) else 0
            tgt = int(link['target'].split('_')[-1]) if '_' in str(link['target']) else 0
            edge_index.append([src, tgt])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        pyg_graphs[graph_type] = Data(x=x, edge_index=edge_index)
    
    return pyg_graphs


def prepare_dataset():
    """准备训练数据集"""
    index_file = Path("data/graphs/main_dataset/main_dataset_index.json")
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    # 漏洞类型映射
    vuln_to_idx = {
        'reentrancy': 0,
        'overflow': 1,
        'access_control': 2,
        'tx_origin': 3,
        'timestamp': 4,
        'unchecked_call': 5,
        'unknown': 6
    }
    
    dataset = []
    
    print("Loading graphs...")
    for item in tqdm(index[:50]):  # 先用 50 个测试
        try:
            graphs = load_graph_data(item['graph_path'])
            label = vuln_to_idx.get(item['vulnerability_type'], 6)
            
            dataset.append({
                'graphs': graphs,
                'label': label,
                'contract_id': item['contract_id']
            })
        except Exception as e:
            print(f"Error loading {item['contract_id']}: {e}")
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    return train_data, val_data, test_data


def train_epoch(model, train_data, optimizer, criterion):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for item in train_data:
        optimizer.zero_grad()
        
        graphs = item['graphs']
        label = torch.tensor([item['label']], dtype=torch.long).to(device)
        
        # 为每个图创建独立的 batch
        batch_ast = torch.zeros(graphs['ast'].x.size(0), dtype=torch.long).to(device)
        batch_cfg = torch.zeros(graphs['cfg'].x.size(0), dtype=torch.long).to(device)
        batch_dfg = torch.zeros(graphs['dfg'].x.size(0), dtype=torch.long).to(device)
        batch_pdg = torch.zeros(graphs['pdg'].x.size(0), dtype=torch.long).to(device)
        
        # 移动数据到设备
        for g_type in graphs:
            graphs[g_type] = graphs[g_type].to(device)
        
        # 前向传播
        out = model(graphs['ast'], graphs['cfg'], graphs['dfg'], graphs['pdg'], 
                   (batch_ast, batch_cfg, batch_dfg, batch_pdg))
        loss = criterion(out, label)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += 1
    
    return total_loss / len(train_data), correct / total


def evaluate(model, val_data):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in val_data:
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
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += 1
    
    return correct / total if total > 0 else 0


def main():
    print("="*70)
    print("🚀 Training MGVD Model")
    print("="*70)
    
    # 准备数据
    print("\n📦 Preparing dataset...")
    train_data, val_data, test_data = prepare_dataset()
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # 初始化模型
    model = MGVDModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print("\n🔥 Training...")
    num_epochs = 10
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_data, optimizer, criterion)
        val_acc = evaluate(model, val_data)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/mgvd_best.pth')
            print(f"  ✅ Saved best model (Val Acc: {val_acc:.4f})")
    
    # 测试
    test_acc = evaluate(model, test_data)
    print(f"\n📊 Test Accuracy: {test_acc:.4f}")
    print(f"✅ Training complete!")


if __name__ == "__main__":
    # 创建模型目录
    Path("models").mkdir(exist_ok=True)
    main()