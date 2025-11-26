#!/usr/bin/env python3
"""
为主数据集（SmartBugs + SolidiFI）构建图
用于 Phase 3-5 的主模型训练
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.simple_graph_builder import SimpleGraphBuilder
import json
from tqdm import tqdm

def collect_main_dataset():
    """收集所有主数据集合约"""
    
    contracts = []
    
    # 从 SmartBugs 收集
    smartbugs_dir = Path("/home/xu/FedVulGuard/data/raw/smartbugs/smartbugs")
    if smartbugs_dir.exists():
        print(f"📂 Scanning SmartBugs: {smartbugs_dir}")
        for sol_file in smartbugs_dir.rglob("*.sol"):
            try:
                with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                # 推断漏洞类型（从路径）
                vuln_type = 'unknown'
                path_str = str(sol_file).lower()
                if 'reentrancy' in path_str:
                    vuln_type = 'reentrancy'
                elif 'overflow' in path_str or 'arithmetic' in path_str:
                    vuln_type = 'overflow'
                elif 'access' in path_str:
                    vuln_type = 'access_control'
                elif 'unchecked' in path_str:
                    vuln_type = 'unchecked_call'
                elif 'timestamp' in path_str or 'time' in path_str:
                    vuln_type = 'timestamp'
                elif 'tx_origin' in path_str or 'txorigin' in path_str:
                    vuln_type = 'tx_origin'
                
                contracts.append({
                    'contract_id': f"smartbugs_{sol_file.stem}",
                    'code': code,
                    'vulnerability_type': vuln_type,
                    'source': 'smartbugs',
                    'filename': sol_file.name
                })
                
            except Exception as e:
                print(f"⚠️  Error reading {sol_file}: {e}")
    
    # 从 SolidiFI 收集
    solidifi_dir = Path("/home/xu/FedVulGuard/data/raw/solidifi")
    if solidifi_dir.exists():
        print(f"📂 Scanning SolidiFI: {solidifi_dir}")
        for sol_file in solidifi_dir.rglob("*.sol"):
            try:
                with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                
                vuln_type = 'unknown'
                path_str = str(sol_file).lower()
                if 'reentrancy' in path_str:
                    vuln_type = 'reentrancy'
                elif 'overflow' in path_str:
                    vuln_type = 'overflow'
                elif 'access' in path_str:
                    vuln_type = 'access_control'
                
                contracts.append({
                    'contract_id': f"solidifi_{sol_file.stem}",
                    'code': code,
                    'vulnerability_type': vuln_type,
                    'source': 'solidifi',
                    'filename': sol_file.name
                })
                
            except Exception as e:
                print(f"⚠️  Error reading {sol_file}: {e}")
    
    print(f"\n✅ Collected {len(contracts)} contracts")
    return contracts


def build_main_dataset_graphs():
    """构建主数据集的图"""
    
    print("="*70)
    print("🔧 Building Graphs for Main Dataset")
    print("="*70)
    
    # 收集合约
    contracts = collect_main_dataset()
    
    if not contracts:
        print("\n❌ No contracts found!")
        print("💡 Please check:")
        print("   - data/smartbugs/ exists and contains .sol files")
        print("   - data/solidifi/ exists and contains .sol files")
        return
    
    # 初始化图构建器（输出到不同目录）
    builder = SimpleGraphBuilder(output_dir="data/graphs/main_dataset")
    
    # 构建图
    results = []
    success_count = 0
    
    for contract in tqdm(contracts, desc="Building graphs"):
        contract_id = contract['contract_id']
        
        try:
            graphs = builder.build_all_graphs(contract['code'], contract_id)
            
            # 保存
            output_path = builder.output_dir / f"{contract_id}.json"
            builder.save_graphs(graphs, output_path)
            
            results.append({
                'contract_id': contract_id,
                'vulnerability_type': contract['vulnerability_type'],
                'source': contract['source'],
                'filename': contract['filename'],
                'graph_path': str(output_path),
                'metadata': graphs['metadata']
            })
            
            success_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Error processing {contract_id}: {e}")
    
    # 保存索引
    index_path = builder.output_dir / 'main_dataset_index.json'
    with open(index_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 统计
    print("\n" + "="*70)
    print("📊 Main Dataset Graph Building Results")
    print("="*70)
    print(f"Total contracts: {len(contracts)}")
    print(f"Successfully built: {success_count}")
    print(f"Failed: {len(contracts) - success_count}")
    print(f"Success rate: {success_count/len(contracts)*100:.1f}%")
    
    # 漏洞类型分布
    from collections import Counter
    vuln_dist = Counter(r['vulnerability_type'] for r in results)
    print(f"\n🔖 Vulnerability Distribution:")
    for vtype, count in vuln_dist.most_common():
        print(f"   {vtype:20s}: {count:3d}")
    
    print(f"\n✅ Graphs saved to: {builder.output_dir}")
    print(f"📋 Index saved to: {index_path}")
    
    return results


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║    Main Dataset Graph Builder (SmartBugs + SolidiFI)    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    results = build_main_dataset_graphs()
    
    if results:
        print("\n💡 Next Steps:")
        print("   1. 使用 SPC 检测器清洗数据")
        print("   2. 提取图特征")
        print("   3. 划分训练/验证/测试集")
        print("   4. 开始训练 MGVD 模型 (Phase 3)")


if __name__ == "__main__":
    main()