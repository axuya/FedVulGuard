#!/usr/bin/env python3
"""
筛选最佳 SPC 对用于 Bootstrap 训练
优先选择相似度适中、漏洞类型明确的高质量对
"""

import json
from pathlib import Path
from collections import Counter

def filter_best_pairs(input_file, output_file, target_size=60):
    """
    筛选策略：
    1. 优先选择漏洞类型明确的（非 unknown）
    2. 相似度在 0.7-0.95 范围（最有价值）
    3. 质量分数高
    4. 保持漏洞类型平衡
    """
    
    print("="*70)
    print("🔍 Filtering Best SPC Pairs for Bootstrap Training")
    print("="*70)
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        all_pairs = json.load(f)
    
    print(f"\n📦 Input: {len(all_pairs)} pairs")
    
    # 分类
    high_value = []    # 高价值：明确类型 + 合适相似度
    medium_value = []  # 中等价值：明确类型但相似度偏高
    low_value = []     # 低价值：unknown 类型
    
    for pair in all_pairs:
        vuln_type = pair.get('vulnerability_type', 'unknown')
        similarity = pair.get('similarity', 0)
        quality = pair.get('quality_score', 0)
        
        # 评分
        score = 0
        
        # 1. 漏洞类型明确 (+30分)
        if vuln_type != 'unknown':
            score += 30
        
        # 2. 相似度理想 (+40分)
        if 0.70 <= similarity <= 0.85:
            score += 40
        elif 0.85 < similarity <= 0.95:
            score += 20
        elif similarity > 0.95:
            score += 5
        
        # 3. 质量分数 (+30分)
        score += quality * 30
        
        pair['_score'] = score
        
        # 分类
        if vuln_type != 'unknown' and 0.70 <= similarity <= 0.90:
            high_value.append(pair)
        elif vuln_type != 'unknown':
            medium_value.append(pair)
        else:
            low_value.append(pair)
    
    print(f"\n📊 Classification:")
    print(f"   High value:   {len(high_value)} pairs (known type + ideal similarity)")
    print(f"   Medium value: {len(medium_value)} pairs (known type + high similarity)")
    print(f"   Low value:    {len(low_value)} pairs (unknown type)")
    
    # 选择策略
    selected = []
    
    # 1. 优先选所有高价值对
    selected.extend(high_value)
    print(f"\n✅ Selected all high-value pairs: {len(high_value)}")
    
    # 2. 从中等价值中补充
    remaining = target_size - len(selected)
    if remaining > 0 and medium_value:
        # 按评分排序
        medium_value.sort(key=lambda x: x['_score'], reverse=True)
        
        # 按漏洞类型平衡选择
        vuln_counts = Counter(p['vulnerability_type'] for p in selected)
        
        for pair in medium_value:
            if len(selected) >= target_size:
                break
            
            vtype = pair['vulnerability_type']
            # 避免某个类型过多
            if vuln_counts[vtype] < target_size // 4:
                selected.append(pair)
                vuln_counts[vtype] += 1
            elif len(selected) < target_size - 10:  # 接近目标时放宽限制
                selected.append(pair)
                vuln_counts[vtype] += 1
        
        print(f"✅ Added from medium-value: {len(selected) - len(high_value)}")
    
    # 3. 如果还不够，从低价值中选择质量最高的
    remaining = target_size - len(selected)
    if remaining > 0 and low_value:
        low_value.sort(key=lambda x: x['_score'], reverse=True)
        selected.extend(low_value[:remaining])
        print(f"✅ Added from low-value: {min(remaining, len(low_value))}")
    
    # 重新分配 pair_id
    for i, pair in enumerate(selected):
        pair['pair_id'] = f"filtered_{i:04d}"
        # 删除临时评分字段
        if '_score' in pair:
            del pair['_score']
    
    # 统计
    print(f"\n📊 Selected Dataset Statistics:")
    print(f"   Total pairs: {len(selected)}")
    
    vuln_dist = Counter(p['vulnerability_type'] for p in selected)
    print(f"\n   Vulnerability distribution:")
    for vtype, count in vuln_dist.most_common():
        print(f"      {vtype:20s}: {count:3d}")
    
    similarities = [p['similarity'] for p in selected if p.get('similarity')]
    if similarities:
        print(f"\n   Similarity statistics:")
        print(f"      Average: {sum(similarities)/len(similarities):.3f}")
        print(f"      Range: {min(similarities):.3f} - {max(similarities):.3f}")
        
        # 区间分布
        ideal_sim = [s for s in similarities if 0.7 <= s <= 0.9]
        print(f"      In ideal range (0.7-0.9): {len(ideal_sim)} ({len(ideal_sim)/len(similarities)*100:.1f}%)")
    
    quality_scores = [p['quality_score'] for p in selected if p.get('quality_score')]
    if quality_scores:
        high_q = sum(1 for q in quality_scores if q >= 1.0)
        print(f"\n   Quality metrics:")
        print(f"      High quality (≥1.0): {high_q} ({high_q/len(selected)*100:.1f}%)")
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # 质量评估
    print("\n" + "="*70)
    print("✅ Quality Assessment")
    print("="*70)
    
    known_type = len([p for p in selected if p['vulnerability_type'] != 'unknown'])
    print(f"✅ Known vulnerability types: {known_type}/{len(selected)} ({known_type/len(selected)*100:.0f}%)")
    
    if ideal_sim:
        print(f"✅ Pairs in ideal similarity range: {len(ideal_sim)}/{len(selected)} ({len(ideal_sim)/len(selected)*100:.0f}%)")
    
    if len(vuln_dist) >= 3:
        print(f"✅ Good type diversity: {len(vuln_dist)} types")
    
    print("\n💡 This filtered dataset is optimized for Bootstrap training!")
    
    return selected


def compare_datasets(original_file, filtered_file):
    """比较原始和筛选后的数据集"""
    
    with open(original_file, 'r') as f:
        original = json.load(f)
    
    with open(filtered_file, 'r') as f:
        filtered = json.load(f)
    
    print("\n" + "="*70)
    print("📊 Before vs After Comparison")
    print("="*70)
    
    print(f"\nDataset size:")
    print(f"   Original:  {len(original)} pairs")
    print(f"   Filtered:  {len(filtered)} pairs")
    print(f"   Reduction: {len(original) - len(filtered)} pairs ({(1-len(filtered)/len(original))*100:.0f}%)")
    
    # 漏洞类型对比
    print(f"\nUnknown type proportion:")
    orig_unknown = sum(1 for p in original if p['vulnerability_type'] == 'unknown')
    filt_unknown = sum(1 for p in filtered if p['vulnerability_type'] == 'unknown')
    print(f"   Original:  {orig_unknown}/{len(original)} ({orig_unknown/len(original)*100:.0f}%)")
    print(f"   Filtered:  {filt_unknown}/{len(filtered)} ({filt_unknown/len(filtered)*100:.0f}%)")
    
    # 相似度对比
    orig_sim = [p['similarity'] for p in original if p.get('similarity')]
    filt_sim = [p['similarity'] for p in filtered if p.get('similarity')]
    
    print(f"\nAverage similarity:")
    print(f"   Original:  {sum(orig_sim)/len(orig_sim):.3f}")
    print(f"   Filtered:  {sum(filt_sim)/len(filt_sim):.3f}")


def main():
    input_file = "data/spc_data/processed/bootstrap_spc_dataset.json"
    output_file = "data/spc_data/processed/bootstrap_filtered_60.json"
    
    # 筛选
    selected = filter_best_pairs(input_file, output_file, target_size=60)
    
    # 对比
    compare_datasets(input_file, output_file)
    
    print("\n" + "="*70)
    print("🎉 Filtering Complete!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("   1. Review the filtered dataset")
    print("   2. [Optional] Manual annotation of 'unknown' types")
    print("   3. Use for Bootstrap SPC detector training")
    print("   4. Proceed to Phase 2 (Multi-graph representation)")


if __name__ == "__main__":
    main()