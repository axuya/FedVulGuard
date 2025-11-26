#!/usr/bin/env python3
"""
合并所有 SPC 数据源并生成最终的 Bootstrap 数据集
"""

import json
from pathlib import Path
from typing import List, Dict

def load_spc_pairs() -> List[Dict]:
    """加载所有 SPC 数据"""
    spc_dir = Path("data/spc_data/raw")
    all_pairs = []
    
    spc_files = [
        'spc_pairs_from_datasets.json',
        'spc_pairs_enhanced.json',
        'spc_pairs_search.json',  # GitHub 搜索
        'spc_pairs_repos.json',   # GitHub 仓库
        'spc_pairs_all.json'      # GitHub 合并
    ]
    
    sources_found = []
    
    for filename in spc_files:
        filepath = spc_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    pairs = json.load(f)
                    if pairs:
                        all_pairs.extend(pairs)
                        sources_found.append(f"{filename}: {len(pairs)} pairs")
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
    
    print("\n📦 Loaded from sources:")
    for source in sources_found:
        print(f"   {source}")
    
    return all_pairs


def deduplicate_pairs(pairs: List[Dict]) -> List[Dict]:
    """去重（基于代码哈希）"""
    seen = set()
    unique = []
    
    for pair in pairs:
        # 创建唯一标识
        key = f"{pair.get('code_before', '')}||{pair.get('code_after', '')}"
        key_hash = hash(key)
        
        if key_hash not in seen:
            seen.add(key_hash)
            unique.append(pair)
    
    print(f"\n🔍 Deduplication: {len(pairs)} → {len(unique)} unique pairs")
    return unique


def filter_high_quality(pairs: List[Dict]) -> List[Dict]:
    """筛选高质量 SPC 对"""
    filtered = []
    
    for pair in pairs:
        # 质量标准
        code_before = pair.get('code_before', '')
        code_after = pair.get('code_after', '')
        similarity = pair.get('similarity', 0)
        
        # 1. 代码长度合理
        if len(code_before) < 50 or len(code_after) < 50:
            continue
        
        # 2. 相似度在合理范围
        if similarity and (similarity < 0.6 or similarity > 0.98):
            continue
        
        # 3. 有明确的漏洞类型（优先）
        if pair.get('vulnerability_type') != 'unknown':
            pair['quality_score'] = 1.0
        else:
            pair['quality_score'] = 0.5
        
        filtered.append(pair)
    
    # 按质量评分排序
    filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    print(f"✅ Quality filtering: {len(pairs)} → {len(filtered)} high-quality pairs")
    return filtered


def create_bootstrap_dataset(pairs: List[Dict], target_size: int = 100):
    """创建 Bootstrap 数据集"""
    print(f"\n🎯 Creating Bootstrap dataset (target: {target_size} pairs)...")
    
    # 按漏洞类型分组
    by_type = {}
    for pair in pairs:
        vtype = pair.get('vulnerability_type', 'unknown')
        if vtype not in by_type:
            by_type[vtype] = []
        by_type[vtype].append(pair)
    
    # 从每个类型选择
    bootstrap = []
    per_type = target_size // len(by_type)
    
    for vtype, type_pairs in by_type.items():
        selected = type_pairs[:per_type]
        bootstrap.extend(selected)
        print(f"   {vtype}: {len(selected)} pairs")
    
    # 如果还不够，从剩余中补充
    if len(bootstrap) < target_size:
        remaining = [p for p in pairs if p not in bootstrap]
        bootstrap.extend(remaining[:target_size - len(bootstrap)])
    
    # 重新分配 pair_id
    for i, pair in enumerate(bootstrap):
        pair['pair_id'] = f"bootstrap_{i:04d}"
    
    return bootstrap[:target_size]


def generate_statistics(pairs: List[Dict]):
    """生成统计报告"""
    print("\n" + "="*60)
    print("📊 Dataset Statistics")
    print("="*60)
    
    print(f"Total pairs: {len(pairs)}")
    
    # 漏洞类型分布
    vuln_dist = {}
    for pair in pairs:
        vtype = pair.get('vulnerability_type', 'unknown')
        vuln_dist[vtype] = vuln_dist.get(vtype, 0) + 1
    
    print("\n🔖 Vulnerability Distribution:")
    for vtype, count in sorted(vuln_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(pairs) * 100
        print(f"   {vtype:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # 相似度分布
    similarities = [p.get('similarity', 0) for p in pairs if p.get('similarity')]
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"\n📏 Average similarity: {avg_sim:.3f}")
        print(f"   Min: {min(similarities):.3f}")
        print(f"   Max: {max(similarities):.3f}")
    
    # 数据来源
    sources = {}
    for pair in pairs:
        method = pair.get('method', pair.get('patch_method', 'unknown'))
        sources[method] = sources.get(method, 0) + 1
    
    print("\n📁 Data Sources:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"   {source}: {count}")


def create_annotation_template(pairs: List[Dict]):
    """创建标注模板"""
    template = []
    
    for pair in pairs:
        item = {
            'pair_id': pair.get('pair_id'),
            'vulnerability_type': pair.get('vulnerability_type'),
            'similarity': pair.get('similarity'),
            'code_before_preview': pair.get('code_before', '')[:200] + '...',
            'code_after_preview': pair.get('code_after', '')[:200] + '...',
            'annotation': {
                'is_valid_spc': None,  # 人工标注: True/False
                'confirmed_vulnerability_type': None,
                'severity': None,  # low/medium/high/critical
                'quality_rating': None,  # 1-5
                'notes': ''
            }
        }
        template.append(item)
    
    output_path = Path("data/spc_data/annotated/bootstrap_annotation_template.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Annotation template: {output_path}")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         SPC Data Merger & Bootstrap Creator             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 1. 加载所有数据
    all_pairs = load_spc_pairs()
    
    if not all_pairs:
        print("\n❌ No SPC pairs found! Run the collection scripts first:")
        print("   python src/data_collection/enhanced_spc_builder.py")
        return
    
    # 2. 去重
    unique_pairs = deduplicate_pairs(all_pairs)
    
    # 3. 质量过滤
    quality_pairs = filter_high_quality(unique_pairs)
    
    # 4. 创建 Bootstrap 数据集
    bootstrap = create_bootstrap_dataset(quality_pairs, target_size=100)
    
    # 5. 保存
    output_dir = Path("data/spc_data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整数据
    with open(output_dir / 'all_spc_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(quality_pairs, f, indent=2, ensure_ascii=False)
    
    # 保存 Bootstrap 数据
    with open(output_dir / 'bootstrap_spc_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(bootstrap, f, indent=2, ensure_ascii=False)
    
    # 6. 统计
    generate_statistics(bootstrap)
    
    # 7. 创建标注模板
    create_annotation_template(bootstrap)
    
    print("\n" + "="*60)
    print("✅ SPC Data Preparation Complete!")
    print("="*60)
    print(f"\n📁 Output files:")
    print(f"   All pairs: data/spc_data/processed/all_spc_pairs.json ({len(quality_pairs)} pairs)")
    print(f"   Bootstrap: data/spc_data/processed/bootstrap_spc_dataset.json ({len(bootstrap)} pairs)")
    print(f"   Annotation template: data/spc_data/annotated/bootstrap_annotation_template.json")
    
    print("\n💡 Next Steps:")
    print("   1. Review bootstrap_annotation_template.json")
    print("   2. Manually annotate the pairs")
    print("   3. Select top 50-100 high-quality pairs for Bootstrap phase")
    print("   4. Proceed to Phase 2 (data preprocessing)")


if __name__ == "__main__":
    main()