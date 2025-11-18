import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def analyze_graph_quality():
    """分析图数据质量"""
    
    print("="*70)
    print("📊 Graph Data Quality Analysis")
    print("="*70)
    
    graph_dir = Path("data/graphs")
    index_file = graph_dir / "graph_index.json"
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return
    
    # 加载索引
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    print(f"\n📦 Total pairs in index: {len(index)}")
    
    # 统计信息
    stats = {
        'ast': {'nodes': [], 'edges': []},
        'cfg': {'nodes': [], 'edges': []},
        'dfg': {'nodes': [], 'edges': []},
        'pdg': {'nodes': [], 'edges': []}
    }
    
    vuln_types = Counter()
    sloc_before = []
    sloc_after = []
    
    # 分析每个图文件
    valid_count = 0
    for item in index:
        pair_id = item['pair_id']
        vuln_types[item['vulnerability_type']] += 1
        
        # 读取 before 图
        before_path = Path(item['before_graphs'])
        if before_path.exists():
            try:
                with open(before_path, 'r') as f:
                    before_data = json.load(f)
                
                # 收集统计信息
                for graph_type in ['ast', 'cfg', 'dfg', 'pdg']:
                    g = before_data[graph_type]
                    stats[graph_type]['nodes'].append(len(g['nodes']))
                    stats[graph_type]['edges'].append(len(g['links']))
                
                sloc_before.append(before_data['metadata']['sloc'])
                valid_count += 1
                
            except Exception as e:
                print(f"⚠️  Error reading {before_path}: {e}")
        
        # 读取 after 图
        after_path = Path(item['after_graphs'])
        if after_path.exists():
            try:
                with open(after_path, 'r') as f:
                    after_data = json.load(f)
                sloc_after.append(after_data['metadata']['sloc'])
            except:
                pass
    
    print(f"✅ Valid graph files: {valid_count}")
    
    # 打印统计
    print("\n" + "="*70)
    print("📊 Graph Statistics")
    print("="*70)
    
    for graph_type in ['ast', 'cfg', 'dfg', 'pdg']:
        nodes = stats[graph_type]['nodes']
        edges = stats[graph_type]['edges']
        
        if nodes:
            print(f"\n{graph_type.upper()}:")
            print(f"   Nodes: avg={np.mean(nodes):.1f}, "
                  f"min={min(nodes)}, max={max(nodes)}, std={np.std(nodes):.1f}")
            print(f"   Edges: avg={np.mean(edges):.1f}, "
                  f"min={min(edges)}, max={max(edges)}, std={np.std(edges):.1f}")
    
    # 漏洞类型分布
    print("\n" + "="*70)
    print("🔖 Vulnerability Type Distribution")
    print("="*70)
    for vtype, count in vuln_types.most_common():
        print(f"   {vtype:20s}: {count:3d} ({count/len(index)*100:5.1f}%)")
    
    # 代码长度
    if sloc_before and sloc_after:
        print("\n" + "="*70)
        print("📝 Code Length (SLOC)")
        print("="*70)
        print(f"   Before: avg={np.mean(sloc_before):.1f}, "
              f"min={min(sloc_before)}, max={max(sloc_before)}")
        print(f"   After:  avg={np.mean(sloc_after):.1f}, "
              f"min={min(sloc_after)}, max={max(sloc_after)}")
    
    # 质量检查
    print("\n" + "="*70)
    print("✅ Quality Checks")
    print("="*70)
    
    checks = []
    
    # 检查1: 节点数合理
    avg_nodes = np.mean(stats['ast']['nodes'])
    if avg_nodes > 10:
        checks.append(("✅", f"AST节点数合理 (avg={avg_nodes:.1f})"))
    else:
        checks.append(("⚠️ ", f"AST节点数偏少 (avg={avg_nodes:.1f})"))
    
    # 检查2: 边数合理
    avg_edges = np.mean(stats['cfg']['edges'])
    if avg_edges > 5:
        checks.append(("✅", f"CFG边数合理 (avg={avg_edges:.1f})"))
    else:
        checks.append(("⚠️ ", f"CFG边数偏少 (avg={avg_edges:.1f})"))
    
    # 检查3: 类型覆盖
    if len(vuln_types) >= 3:
        checks.append(("✅", f"漏洞类型覆盖良好 ({len(vuln_types)}种)"))
    else:
        checks.append(("⚠️ ", f"漏洞类型覆盖不足 ({len(vuln_types)}种)"))
    
    # 检查4: 成功率
    success_rate = valid_count / len(index)
    if success_rate >= 0.9:
        checks.append(("✅", f"图构建成功率高 ({success_rate*100:.1f}%)"))
    else:
        checks.append(("⚠️ ", f"图构建成功率偏低 ({success_rate*100:.1f}%)"))
    
    for status, msg in checks:
        print(f"{status} {msg}")
    
    # 推荐
    print("\n" + "="*70)
    print("💡 Recommendations")
    print("="*70)
    
    if success_rate >= 0.9 and avg_nodes > 20:
        print("✅ 图数据质量优秀，可以进入 Phase 3")
        print("\n下一步:")
        print("   1. 特征提取: python src/preprocessing/extract_features.py")
        print("   2. 训练 MGVD: python src/models/train_mgvd.py")
    elif success_rate >= 0.8:
        print("⚠️  图数据基本可用，但建议检查失败的样本")
        print("   可以继续，但可能需要调整模型参数")
    else:
        print("❌ 图数据质量不足，建议检查并修复")
    
    return stats, vuln_types


def visualize_sample_graph():
    """可视化一个示例图"""
    print("\n" + "="*70)
    print("🎨 Sample Graph Visualization")
    print("="*70)
    
    try:
        import networkx as nx
        
        # 读取第一个图
        graph_file = Path("data/graphs/filtered_0000_before.json")
        if not graph_file.exists():
            print("❌ Sample graph not found")
            return
        
        with open(graph_file, 'r') as f:
            data = json.load(f)
        
        # 只可视化 AST（最简单）
        ast_data = data['ast']
        G = nx.node_link_graph(ast_data)
        
        print(f"\n📊 Sample Graph Info:")
        print(f"   Pair: filtered_0000_before")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        print(f"   Node types: {set(nx.get_node_attributes(G, 'node_type').values())}")
        
        print("\n💡 Tip: 可以使用 networkx 可视化:")
        print("   import networkx as nx")
        print("   import matplotlib.pyplot as plt")
        print("   nx.draw(G, with_labels=True)")
        print("   plt.show()")
        
    except ImportError:
        print("⚠️  networkx not available for visualization")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         Graph Data Quality Verification                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    stats, vuln_types = analyze_graph_quality()
    visualize_sample_graph()
    
    print("\n" + "="*70)
    print("🎉 Verification Complete!")
    print("="*70)


if __name__ == "__main__":
    main()