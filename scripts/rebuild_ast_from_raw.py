import json
from pathlib import Path
from tree_sitter import Language, Parser
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/graphs_ast_llm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHAINS = ["Arbitrum", "Avalanche", "BSC", "Ethereum", "Fantom", "Polygon"]

# 加载你的 Solidity 语法库
SOL_PARSER = Parser()
SOL_PARSER.set_language(Language("build/libsolidity.so", "solidity"))


def parse_ast_tree(code):
    """解析成轻量 AST（child-only）"""
    try:
        tree = SOL_PARSER.parse(bytes(code, "utf8"))
        root = tree.root_node
        return root
    except Exception:
        return None

def node_to_dict(node):
    """将 tree-sitter 节点转为可序列化 Dict，只保留 type + children"""
    obj = {"type": node.type, "children": []}
    for c in node.children:
        obj["children"].append(node_to_dict(c))
    return obj

def main():
    for chain in CHAINS:
        print(f"\n[AST] Building AST for {chain}")
        raw_chain_dir = RAW_DIR / chain
        out_path = OUT_DIR / f"{chain}.jsonl"

        sol_files = list(raw_chain_dir.glob("*.sol"))
        print(f"[INFO] Found {len(sol_files)} solidity files")

        with open(out_path, "w", encoding="utf8") as fout:
            for sol_file in tqdm(sol_files, desc=f"AST {chain}"):
                try:
                    code = sol_file.read_text(encoding="utf8", errors="ignore")
                except:
                    continue

                root = parse_ast_tree(code)
                if root is None:
                    continue

                ast_obj = node_to_dict(root)

                # id 用文件名（不带后缀），与 CFG/DFG 完全一致
                gid = sol_file.stem  # e.g., 123abc_Token

                rec = {
                    "id": gid,
                    "chain": chain,
                    "ast_nodes": ast_obj["children"],  # 轻量化（可按需修改）
                    "ast_edges": []  # 如需 AST-edge 再补
                }

                fout.write(json.dumps(rec) + "\n")

        print(f"[OK] Saved AST → {out_path}")


if __name__ == "__main__":
    main()
