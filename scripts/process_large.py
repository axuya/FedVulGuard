#!/usr/bin/env python3
"""
处理 smart-contract-sanctuary 大规模数据集
"""

import json
import hashlib
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import multiprocessing as mp
import random

class LargeScaleDataProcessor:
    """大规模数据处理器"""
    
    def __init__(self, 
                 sanctuary_dir="/home/xu/FedVulGuard/data/raw/sanctuary_full",
                 output_dir="data/processed_large_scale"):
        self.sanctuary_dir = Path(sanctuary_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_all_contracts(self):
        """收集所有合约"""
        print("📦 Step 1: 收集所有合约文件")
        
        all_contracts = []
        
        if self.sanctuary_dir.exists():
            print(f"   扫描 sanctuary: {self.sanctuary_dir}")
            for sol_file in tqdm(list(self.sanctuary_dir.rglob("*.sol")), desc="Sanctuary"):
                all_contracts.append({
                    'path': sol_file,
                    'source': 'sanctuary',
                    'chain': self._infer_chain_from_path(sol_file)
                })
        
        print(f"\n✅ 收集到 {len(all_contracts)} 个合约文件")
        return all_contracts
    
    def _infer_chain_from_path(self, path):
        """从路径推断链"""
        path_str = str(path).lower()
        if 'ethereum' in path_str or 'mainnet' in path_str:
            return 'ethereum'
        elif 'bsc' in path_str or 'binance' in path_str:
            return 'bsc'
        elif 'polygon' in path_str or 'matic' in path_str:
            return 'polygon'
        elif 'avalanche' in path_str or 'avax' in path_str:
            return 'avalanche'
        return 'unknown'
    
    def quality_filter(self, contracts, batch_size=10000):
        """质量过滤"""
        print("\n🔍 Step 2: 质量过滤")
        
        filtered = []
        seen_hashes = set()
        
        for i in tqdm(range(0, len(contracts), batch_size), desc="批处理"):
            batch = contracts[i:i+batch_size]
            
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(self._process_contract, batch)
            
            for result in results:
                if result and result['code_hash'] not in seen_hashes:
                    seen_hashes.add(result['code_hash'])
                    filtered.append(result)
        
        print(f"✅ 过滤后: {len(filtered)} 个合约")
        return filtered
    
    def _process_contract(self, contract_info):
        """处理单个合约"""
        try:
            with open(contract_info['path'], 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            if len(code) < 50 or len(code) > 100000:
                return None
            
            if not any(x in code.lower() for x in ['pragma','contract','function']):
                return None
            
            lines = code.count('\n')
            if lines < 5 or lines > 5000:
                return None
            
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            
            return {
                'contract_id': contract_info['path'].stem,
                'code': code,
                'code_hash': code_hash,
                'chain': contract_info['chain'],
                'source': contract_info['source'],
                'sloc': lines
            }
            
        except:
            return None
    
    def _reclassify_unknown(self, unknown_contracts, max_check=50000):
        """重新识别 unknown 合约的链"""
        reclassified = {
            'ethereum': [],
            'bsc': [],
            'polygon': [],
            'avalanche': []
        }
        
        check_list = unknown_contracts[:max_check]
        
        for contract in tqdm(check_list, desc="重新分类"):
            try:
                code = contract.get('code', '')
                
                if any(x in code for x in ['pancake', 'PancakeSwap', '0xbb4']):
                    reclassified['bsc'].append(contract)
                elif any(x in code for x in ['polygon', 'matic', '0x0d500']):
                    reclassified['polygon'].append(contract)
                elif any(x in code for x in ['avalanche', 'avax', '0xB31f']):
                    reclassified['avalanche'].append(contract)
                else:
                    reclassified['ethereum'].append(contract)
            except:
                reclassified['ethereum'].append(contract)
        
        return reclassified
    
    def stratified_sampling(self, contracts, target_size=10000, train_ratio=0.7, val_ratio=0.15):
        """分层抽样 - 平衡链分布"""
        print(f"\n📊 Step 3: 分层抽样 (目标: {target_size})")
        
        by_chain = {}
        for c in contracts:
            chain = c['chain']
            if chain not in by_chain:
                by_chain[chain] = []
            by_chain[chain].append(c)
        
        print("\n原始链分布:")
        for chain, items in by_chain.items():
            print(f"   {chain}: {len(items)}")
        
        print("\n🔍 重新识别 unknown 链...")
        unknown_items = by_chain.get('unknown', [])
        reclassified = self._reclassify_unknown(unknown_items)
        
        for chain, items in reclassified.items():
            if chain not in by_chain:
                by_chain[chain] = []
            by_chain[chain].extend(items)
        by_chain.pop('unknown', None)
        
        print("\n重新分类后:")
        for chain, items in by_chain.items():
            print(f"   {chain}: {len(items)}")
        
        sampled = []
        target_chains = ['ethereum', 'bsc', 'polygon', 'avalanche']
        per_chain = target_size // len(target_chains)
        
        print(f"\n目标：每条链 {per_chain} 个样本")
        
        for chain in target_chains:
            items = by_chain.get(chain, [])
            if len(items) < per_chain:
                print(f"   ⚠️  {chain}: 只有 {len(items)} 个")
                sampled.extend(items)
            else:
                random.shuffle(items)
                sampled.extend(items[:per_chain])
                print(f"   ✅ {chain}: 采样 {per_chain} 个")
        
        random.shuffle(sampled)
        
        train_size = int(len(sampled) * train_ratio)
        val_size = int(len(sampled) * val_ratio)
        
        train_data = sampled[:train_size]
        val_data = sampled[train_size:train_size+val_size]
        test_data = sampled[train_size+val_size:]
        
        print(f"\n✅ 采样完成:")
        print(f"   训练集: {len(train_data)}")
        print(f"   验证集: {len(val_data)}")
        print(f"   测试集: {len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'full': sampled
        }
    
    def save_dataset(self, dataset, prefix='large_scale'):
        """保存数据集"""
        print(f"\n💾 Step 4: 保存数据集")
        
        for split, data in dataset.items():
            if split == 'full':
                continue
            
            output_file = self.output_dir / f"{prefix}_{split}.json"
            
            compact_data = []
            for item in data:
                compact_data.append({
                    'contract_id': item['contract_id'],
                    'code_hash': item['code_hash'],
                    'chain': item['chain'],
                    'sloc': item['sloc'],
                })
            
            with open(output_file, 'w') as f:
                json.dump(compact_data, f, indent=2)
            
            print(f"   ✅ {output_file} ({len(compact_data)} 条)")
            
            code_dir = self.output_dir / f"{prefix}_{split}_code"
            code_dir.mkdir(exist_ok=True)
            
            for item in tqdm(data, desc=f"保存 {split} 代码"):
                code_file = code_dir / f"{item['contract_id']}.sol"
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(item['code'])
        
        print("\n✅ 数据集保存完成")
    
    def generate_statistics(self, dataset):
        """生成数据集统计"""
        print("\n" + "="*70)
        print("📊 数据集统计")
        print("="*70)
        
        for split, data in dataset.items():
            if split == 'full':
                continue
            
            print(f"\n{split.upper()}:")
            print(f"   样本数: {len(data)}")
            
            chain_dist = Counter(item['chain'] for item in data)
            print(f"   链分布:")
            for chain, count in chain_dist.most_common():
                print(f"      {chain}: {count} ({count/len(data)*100:.1f}%)")
            
            slocs = [item['sloc'] for item in data]
            print(f"   代码行数: avg={sum(slocs)/len(slocs):.0f}, min={min(slocs)}, max={max(slocs)}")


def main():
    processor = LargeScaleDataProcessor()
    
    all_contracts = processor.collect_all_contracts()
    
    if len(all_contracts) == 0:
        print("\n❌ 未找到合约！")
        return
    
    filtered = processor.quality_filter(all_contracts)
    dataset = processor.stratified_sampling(filtered, target_size=10000)
    processor.save_dataset(dataset)
    processor.generate_statistics(dataset)
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()