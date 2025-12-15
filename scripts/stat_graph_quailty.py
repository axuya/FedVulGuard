import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 路径设置
# ================================
DIR_AST = Path("data/graphs_ast_llm")
DIR_CFG = Path("data/graphs_cfg_contract")
DIR_DFG = Path("data/graphs_dfg")

OUTPUT_DIR = Path("graph_quality_reports")
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================
# 核心：图数据质量检查函数
# ================================
def analyze_graph_dir(dir_path, graph_name):
    """
    根据不同图类型解析不同的字段结构：
    - AST: ast_nodes
    - CFG: graph.nodes  或 cfg_nodes（兼容）
    - DFG: dfg_nodes
    """
    print(f"\n====== Checking {graph_name} in {dir_path} ======")

    files = list(dir_path.glob("*.jsonl"))
    if not files:
        print(f"[WARN] No {graph_name} JSONL files found.")
        return None

    sizes = []
    empty_count = 0

    for f in tqdm(files, desc=f"Scanning {graph_name}"):
        with open(f, "r", encoding="utf8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except Exception:
                    continue

                # 根据图类型解析节点
                if graph_name == "AST":
                    nodes = item.get("ast_nodes", [])

                elif graph_name == "CFG":
                    graph_obj = item.get("graph")
                    if isinstance(graph_obj, dict):
                        nodes = graph_obj.get("nodes", [])
                    else:
                        nodes = item.get("cfg_nodes", [])

                elif graph_name == "DFG":
                    nodes = item.get("dfg_nodes", [])

                else:
                    nodes = []

                sizes.append(len(nodes))
                if len(nodes) == 0:
                    empty_count += 1

    sizes = np.array(sizes)

    if len(sizes) == 0:
        print(f"[WARN] No valid {graph_name} entries parsed.")
        return None

    print(f"\n[{graph_name}] Total samples: {len(sizes)}")
    print(f"Min nodes: {sizes.min()}")
    print(f"Max nodes: {sizes.max()}")
    print(f"Mean nodes: {sizes.mean():.2f}")
    print(f"Median nodes: {np.median(sizes)}")
    print(f"p95: {np.percentile(sizes, 95)}")
    print(f"p99: {np.percentile(sizes, 99)}")
    print(f"Empty graphs: {empty_count}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=50)
    plt.title(f"{graph_name} Node Count Distribution")
    plt.xlabel("Node Count")
    plt.ylabel("Frequency")
    plt.grid(True)

    out_img = OUTPUT_DIR / f"{graph_name.lower()}_hist.png"
    plt.savefig(out_img)
    print(f"[OK] Histogram saved → {out_img}")

    return {
        "sizes": sizes,
        "empty": empty_count,
        "files": len(files),
    }


# ================================
# ID 对齐检查（AST / CFG / DFG）
# ================================
def check_id_alignment():
    print("\n====== Checking ID Alignment (AST vs CFG vs DFG) ======")

    def collect_ids(dir_path, key_guess_list):
        ids = set()
        for f in dir_path.glob("*.jsonl"):
            with open(f, "r", encoding="utf8") as fin:
                for line in fin:
                    try:
                        obj = json.loads(line)
                    except:
                        continue

                    gid = obj.get("id")
                    if gid is not None:
                        ids.add(gid)

        return ids

    ast_ids = collect_ids(DIR_AST, ["ast_nodes"])
    cfg_ids = collect_ids(DIR_CFG, ["graph", "cfg_nodes"])
    dfg_ids = collect_ids(DIR_DFG, ["dfg_nodes"])

    print(f"AST samples: {len(ast_ids)}")
    print(f"CFG samples: {len(cfg_ids)}")
    print(f"DFG samples: {len(dfg_ids)}")

    aligned = ast_ids & cfg_ids & dfg_ids
    print(f"Aligned (AST ∩ CFG ∩ DFG): {len(aligned)}")

    print(f"AST but no CFG: {len(ast_ids - cfg_ids)}")
    print(f"AST but no DFG: {len(ast_ids - dfg_ids)}")

    return {
        "aligned": len(aligned),
        "missing_cfg": len(ast_ids - cfg_ids),
        "missing_dfg": len(ast_ids - dfg_ids),
    }


# ================================
# 主程序入口
# ================================
def main():
    print("=== Graph Quality Checker ===")

    ast = analyze_graph_dir(DIR_AST, "AST")
    cfg = analyze_graph_dir(DIR_CFG, "CFG")
    dfg = analyze_graph_dir(DIR_DFG, "DFG")

    align = check_id_alignment()

    print("\n=== Summary ===")
    print("AST:", ast)
    print("CFG:", cfg)
    print("DFG:", dfg)
    print("Alignment:", align)


if __name__ == "__main__":
    main()
