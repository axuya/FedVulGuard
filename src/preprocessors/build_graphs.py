import os
import json
from tqdm import tqdm
from collections import defaultdict

AST_DIR = "data/graphs_ast"
CFG_DIR = "data/graphs_cfg"
OUT_DIR = "data/graphs_dfg"
RAW_DIR = "data/raw"


def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_identifier_tokens(ast_node, source, results, node_counter):
    """
    从 AST 节点中识别 identifier，并根据 start/end 位置从源码中切片。
    """
    if not isinstance(ast_node, dict):
        return node_counter

    # 检测 identifier 节点
    if ast_node.get("type") == "identifier":
        start = ast_node["startPosition"]
        end = ast_node["endPosition"]

        # 切出变量名：使用行列信息
        try:
            # 按行拆分源代码
            lines = source.split("\n")
            line = lines[start["row"]]
            name = line[start["column"]:end["column"]]
        except Exception:
            name = None

        if name and name.isidentifier():
            results[name].append(node_counter)
            node_counter += 1

    # 递归遍历
    for v in ast_node.values():
        if isinstance(v, dict):
            node_counter = extract_identifier_tokens(v, source, results, node_counter)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    node_counter = extract_identifier_tokens(item, source, results, node_counter)

    return node_counter


def build_dfg(identifier_map):
    """
    def-use edges based on order of appearance
    """
    edges = []
    for var, nodes in identifier_map.items():
        if len(nodes) > 1:
            for i in range(len(nodes) - 1):
                edges.append({"src": nodes[i], "dst": nodes[i+1], "var": var})
    return edges


def load_source(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def process_record(ast_item, cfg_item):
    """
    处理单条记录：使用 AST + 源代码 提取变量引用
    """
    src_path = ast_item["id"]
    source = load_source(src_path)

    identifier_map = defaultdict(list)
    node_counter = 0

    # 提取变量引用
    node_counter = extract_identifier_tokens(
        ast_item["ast"], source, identifier_map, node_counter
    )

    # 构建 DFG
    dfg_edges = build_dfg(identifier_map)

    return {
        "id": ast_item["id"],
        "chain": ast_item.get("chain"),
        "variables": list(identifier_map.keys()),
        "dfg_edges": dfg_edges
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    chains = [f.split(".")[0] for f in os.listdir(AST_DIR) if f.endswith(".jsonl")]

    for chain in chains:
        ast_path = f"{AST_DIR}/{chain}.jsonl"
        cfg_path = f"{CFG_DIR}/{chain}.jsonl"
        out_path = f"{OUT_DIR}/{chain}.jsonl"

        print(f"[START DFG] {chain}")

        with open(out_path, "w", encoding="utf-8") as fout:
            for ast_item, cfg_item in tqdm(zip(stream_jsonl(ast_path), stream_jsonl(cfg_path))):
                result = process_record(ast_item, cfg_item)
                fout.write(json.dumps(result) + "\n")

        print(f"[OK] DFG Built → {out_path}")


if __name__ == "__main__":
    main()
