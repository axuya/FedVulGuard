#!/usr/bin/env python3
"""数据集划分 - 修复版 - 适配实际数据格式"""

import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    print("="*70)
    print("数据集划分工具 - SmartBugs (修复版)")
    print("="*70)
    
    smartbugs_dir = Path("data/graphs/smartbugs")
    
    # 读取所有图文件（排除索引文件）
    graph_files = [f for f in smartbugs_dir.glob("*.json") 
                   if f.name not in ['index.json', 'failed_files.json']]
    
    print(f"\n找到 {len(graph_files)} 个图文件")
    
    # 先检查一个文件的格式
    if graph_files:
        with open(graph_files[0], 'r') as f:
            sample_data = json.load(f)
        print(f"\n样本文件结构:")
        for key in sample_data.keys():
            print(f"  - {key}")
    
    # 读取数据
    contracts = []
    for graph_file in graph_files:
        try:
            with open(graph_file, 'r') as f:
                data = json.load(f)
            
            # 适配不同的键名格式
            contract_id = data.get('contract_id') or data.get('id') or graph_file.stem
            contract_name = data.get('contract_name') or data.get('name') or graph_file.stem
            version = data.get('solidity_version') or data.get('version') or 'unknown'
            
            contracts.append({
                'id': contract_id,
                'name': contract_name,
                'version': version,
                'graph_file': str(graph_file.relative_to(Path.cwd()))
            })
        except Exception as e:
            print(f"警告: 处理 {graph_file} 失败: {e}")
            continue
    
    print(f"\n成功读取 {len(contracts)} 个合约")
    
    # 按版本分组
    version_groups = defaultdict(list)
    for contract in contracts:
        version_groups[contract['version']].append(contract)
    
    print(f"\n版本分布:")
    for version, group in sorted(version_groups.items()):
        print(f"  {version:15s}: {len(group):2d} 个")
    
    # 分层采样
    random.seed(42)
    train_set, val_set, test_set = [], [], []
    
    for version, group in version_groups.items():
        random.shuffle(group)
        n = len(group)
        
        # 70% 训练，15% 验证，15% 测试
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        train_set.extend(group[:train_size])
        val_set.extend(group[train_size:train_size+val_size])
        test_set.extend(group[train_size+val_size:])
    
    # 打乱顺序
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    # 保存划分结果
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
    
    print(f"\n划分结果:")
    total = len(contracts)
    for split_name, dataset in splits.items():
        output_file = output_dir / f"{split_name}_split.json"
        with open(output_file, 'w') as f:
            json.dump({
                'size': len(dataset),
                'contracts': dataset
            }, f, indent=2)
        
        ratio = len(dataset) / total * 100 if total > 0 else 0
        print(f"  {split_name:5s}: {len(dataset):2d} ({ratio:5.1f}%)")
    
    # 保存划分信息
    split_info = {
        'total': total,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
        'random_seed': 42,
        'data_source': 'SmartBugs',
        'note': '50个合约的多版本数据集'
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ 完成! 输出目录: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
