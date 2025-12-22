
import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    print("="*70)
    print("数据集划分工具 - SmartBugs (最终修复版)")
    print("="*70)
    
    # 使用绝对路径
    base_dir = Path.cwd()
    smartbugs_dir = base_dir / "data/graphs/smartbugs"
    
    print(f"工作目录: {base_dir}")
    print(f"图目录: {smartbugs_dir}")
    
    # 读取所有图文件（排除索引文件）
    graph_files = [f for f in smartbugs_dir.glob("*.json") 
                   if f.name not in ['index.json', 'failed_files.json']]
    
    print(f"\n找到 {len(graph_files)} 个图文件")
    
    if len(graph_files) == 0:
        print("❌ 没有找到图文件，请检查目录")
        return
    
    # 检查一个样本文件
    sample_file = graph_files[0]
    print(f"\n样本文件: {sample_file}")
    
    try:
        with open(sample_file, 'r') as f:
            sample_data = json.load(f)
        print(f"文件结构:")
        for key in sample_data.keys():
            print(f"  - {key}")
    except Exception as e:
        print(f"读取样本文件失败: {e}")
        return
    
    # 读取数据 - 使用绝对路径
    contracts = []
    for graph_file in graph_files:
        try:
            with open(graph_file, 'r') as f:
                data = json.load(f)
            
            # 使用相对路径（相对于项目根目录）
            relative_path = graph_file.relative_to(base_dir)
            
            contracts.append({
                'id': data['contract_id'],
                'name': data['contract_name'],
                'version': data['solidity_version'],
                'graph_file': str(relative_path)  # 使用相对路径
            })
        except Exception as e:
            print(f"警告: 处理 {graph_file} 失败: {e}")
            continue
    
    print(f"\n成功读取 {len(contracts)} 个合约")
    
    if len(contracts) == 0:
        print("❌ 没有成功读取任何合约")
        return
    
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
    output_dir = base_dir / "data/splits"
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
        'base_dir': str(base_dir),
        'note': '50个合约的多版本数据集'
    }
    
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ 完成! 输出目录: {output_dir}")
    
    # 验证结果
    print(f"\n验证结果:")
    for split_name in ['train', 'val', 'test']:
        split_file = output_dir / f"{split_name}_split.json"
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        print(f"  {split_name}: {split_data['size']} 个合约")
    
    print("="*70)

if __name__ == "__main__":
    main()