#!/usr/bin/env python3
"""
分析筛选后的数据集
"""

import json
from pathlib import Path
from collections import Counter

def analyze_all_versions():
    """分析所有版本的数据集"""
    
    files = {
        'original': 'data/spc_data/processed/bootstrap_spc_dataset.json',
        'classified': 'data/spc_data/processed/bootstrap_classified.json',
        'filtered': 'data/spc_data/processed/bootstrap_filtered_60.json'
    }
    
    datasets = {}
    
    print("="*70)
    print("📊 Multi-Version Dataset Analysis")
    print("="*70)
    
    # 加载所有版本
    for version, filepath in files.items():
        path = Path(filepath)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                datasets[version] = json.load(f)
            print(f"\n✅ Loaded {version}: {len(datasets[version])} pairs")
        else:
            print(f"\n⚠️  {version} not found: {filepath}")
    
    if not datasets:
        print("\n❌ No datasets found!")
        return
    
    # 详细分析每个版本
    for version, pairs in datasets.items():
        print("\n" + "="*70)
        print(f"📦 {version.upper()} Dataset Analysis")
        print("="*70)
        
        print(f"\nTotal pairs: {len(pairs)}")
        
        # 漏洞类型分布
        vuln_types = Counter(p.get('vulnerability_type', 'unknown') for p in pairs)
        print(f"\n🔖 Vulnerability Types:")
        for vtype, count in vuln_types.most_common():
            percentage = count / len(pairs) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {vtype:20s}: {count:3d} ({percentage:5.1f}%) {bar}")
        
        # 相似度统计
        similarities = [p.get('similarity', 0) for p in pairs if p.get('similarity')]
        if similarities:
            print(f"\n📏 Similarity:")
            print(f"   Average: {sum(similarities)/len(similarities):.3f}")
            print(f"   Range: {min(similarities):.3f} - {max(similarities):.3f}")
            
            # 理想范围内的比例
            ideal = [s for s in similarities if 0.7 <= s <= 0.9]
            print(f"   Ideal range (0.7-0.9): {len(ideal)} ({len(ideal)/len(similarities)*100:.1f}%)")
        
        # 质量评分
        quality_scores = [p.get('quality_score', 0) for p in pairs if p.get('quality_score')]
        if quality_scores:
            high_q = sum(1 for q in quality_scores if q >= 1.0)
            print(f"\n⭐ Quality:")
            print(f"   Average score: {sum(quality_scores)/len(quality_scores):.2f}")
            print(f"   High quality (≥1.0): {high_q} ({high_q/len(pairs)*100:.1f}%)")
    
    # 对比分析
    if len(datasets) > 1:
        print("\n" + "="*70)
        print("📊 Comparison Summary")
        print("="*70)
        
        versions = list(datasets.keys())
        
        print(f"\n{'Metric':<25} {'Original':>15} {'Classified':>15} {'Filtered':>15}")
        print("-" * 70)
        
        # 大小对比
        sizes = {v: len(datasets[v]) for v in versions if v in datasets}
        print(f"{'Total Pairs':<25} {sizes.get('original', 0):>15} {sizes.get('classified', 0):>15} {sizes.get('filtered', 0):>15}")
        
        # Unknown 比例
        for v in versions:
            if v in datasets:
                unknown = sum(1 for p in datasets[v] if p['vulnerability_type'] == 'unknown')
                unknown_pct = {v: f"{unknown} ({unknown/len(datasets[v])*100:.0f}%)"}
        
        if 'original' in datasets and 'classified' in datasets:
            orig_unknown = sum(1 for p in datasets['original'] if p['vulnerability_type'] == 'unknown')
            clas_unknown = sum(1 for p in datasets['classified'] if p['vulnerability_type'] == 'unknown')
            filt_unknown = sum(1 for p in datasets.get('filtered', []) if p['vulnerability_type'] == 'unknown') if 'filtered' in datasets else 0
            
            print(f"{'Unknown Type':<25} {orig_unknown:>15} {clas_unknown:>15} {filt_unknown:>15}")
        
        # 平均相似度
        for v in versions:
            if v in datasets:
                sims = [p['similarity'] for p in datasets[v] if p.get('similarity')]
                avg_sim = sum(sims)/len(sims) if sims else 0
        
        if 'original' in datasets:
            orig_sims = [p['similarity'] for p in datasets['original'] if p.get('similarity')]
            clas_sims = [p['similarity'] for p in datasets.get('classified', []) if p.get('similarity')] if 'classified' in datasets else []
            filt_sims = [p['similarity'] for p in datasets.get('filtered', []) if p.get('similarity')] if 'filtered' in datasets else []
            
            orig_avg = sum(orig_sims)/len(orig_sims) if orig_sims else 0
            clas_avg = sum(clas_sims)/len(clas_sims) if clas_sims else 0
            filt_avg = sum(filt_sims)/len(filt_sims) if filt_sims else 0
            
            print(f"{'Avg Similarity':<25} {orig_avg:>15.3f} {clas_avg:>15.3f} {filt_avg:>15.3f}")
    
    # 推荐使用
    print("\n" + "="*70)
    print("💡 Recommendation")
    print("="*70)
    
    if 'filtered' in datasets:
        print("\n✅ Use FILTERED dataset for Bootstrap training:")
        print(f"   File: data/spc_data/processed/bootstrap_filtered_60.json")
        print(f"   Size: {len(datasets['filtered'])} pairs")
        
        vuln_types = Counter(p['vulnerability_type'] for p in datasets['filtered'])
        unknown_count = vuln_types.get('unknown', 0)
        
        if unknown_count / len(datasets['filtered']) < 0.2:
            print(f"   ✅ Good: Only {unknown_count} unknown types ({unknown_count/len(datasets['filtered'])*100:.0f}%)")
        else:
            print(f"   ⚠️  Still has {unknown_count} unknown types")
        
        print("\n📝 For Paper:")
        print(f"   - Bootstrap dataset: {len(datasets['filtered'])} SPC pairs")
        print(f"   - Vulnerability types: {len(vuln_types)}")
        print(f"   - Coverage: reentrancy, overflow, access_control, etc.")
    
    elif 'classified' in datasets:
        print("\n✅ Use CLASSIFIED dataset:")
        print(f"   File: data/spc_data/processed/bootstrap_classified.json")
        print(f"   Size: {len(datasets['classified'])} pairs")
    
    else:
        print("\n⚠️  Consider running filtering scripts first")


if __name__ == "__main__":
    analyze_all_versions()