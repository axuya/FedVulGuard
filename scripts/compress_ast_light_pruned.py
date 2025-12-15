import json
from pathlib import Path
from tqdm import tqdm

# 输入输出路径
IN_DIR = Path("data/graphs_ast")
OUT_DIR = Path("data/graphs_ast_llm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DEPTH = 30   # 推荐：LLM + MGVD 友好深度

# =============================
#  Slither AST 专用 KEEP TYPES
# =============================
KEEP_TYPES = {
    # --- structural ---
    "source_file",
    "contract_definition",
    "function_definition",
    "modifier_definition",
    "block",

    # --- control flow ---
    "if_statement",
    "for_statement",
    "while_statement",
    "do_while_statement",
    "return",
    "emit_statement",
    "expression_statement",
    "try_statement",
    "catch_clause",

    # --- expressions ---
    "binary_operation",
    "unary_operation",
    "function_call",
    "member_access",
    "index_access",

    # --- declarations ---
    "variable_declaration",
    "parameter_list",
    "elementary_type_name",
    "user_defined_type_name",
    "array_type_name",
    "mapping",
}

# =============================
#   不需要的节点
# =============================
DROP_TYPES = {
    "comment",
    "pragma",
    "pragma_directive",
    "import_directive",
    "identifier",
    "literal",
    "tuple_expression",
    "assembly",
}


def traverse(node, parent, nodes, edges, next_id, depth=0):
    """深度优先遍历 AST 并执行 LLM-Friendly 压缩"""
    if depth > MAX_DEPTH or not isinstance(node, dict):
        return

    # 当前节点类型（小写处理，保证 Slither AST 匹配成功）
    ntype = node.get("type", "unknown").lower()

    # 丢弃无用节点
    if ntype in DROP_TYPES:
        return

    # 是否保留该节点
    keep = ntype in KEEP_TYPES

    # 如果保留该节点，则加入 AST 图结构中
    if keep:
        this_id = next_id[0]
        next_id[0] += 1

        nodes.append({
            "id": this_id,
            "type": ntype
        })

        if parent is not None:
            edges.append({
                "src": parent,
                "dst": this_id,
                "type": "AST"
            })

        parent = this_id  # 新节点成为新的父节点

    # 继续遍历 children（Slither AST 的主体结构）
    children = node.get("children", [])
    if isinstance(children, list):
        for child in children:
            traverse(child, parent, nodes, edges, next_id, depth + 1)


def process_file(src, dst):
    print(f"\n[START] Processing {src.name}")

    # 统计行数
    total = sum(1 for _ in open(src, "r", encoding="utf8", errors="ignore"))

    with open(src, "r", encoding="utf8", errors="ignore") as fin, \
         open(dst, "w", encoding="utf8") as fout:

        for line in tqdm(fin, total=total, desc=f"LLM AST Light {src.name}"):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            # Slither AST 正确入口
            root = item.get("ast", item)

            nodes, edges = [], []
            traverse(root, None, nodes, edges, [0])

            # 如果 AST 为空，给一个兜底节点避免模型崩溃
            if len(nodes) == 0:
                nodes = [{"id": 0, "type": "contract_definition"}]
                edges = []

            fout.write(json.dumps({
                "id": item.get("id"),
                "chain": item.get("chain"),
                "ast_nodes": nodes,
                "ast_edges": edges
            }) + "\n")

    print(f"[OK] Saved → {dst}")


def main():
    print("[INFO] Building LLM-Friendly AST-Light")
    for f in IN_DIR.glob("*.jsonl"):
        out = OUT_DIR / f.name
        process_file(f, out)


if __name__ == "__main__":
    main()
