# src/preprocessing/build_graphs.py

import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 模块搜索路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.statement_graph import StatementGraphExtractor

def build_statement_graphs(graphs_dir: Path, output_dir: Path):
    """
    从 SmartBugs 图文件中提取 statement graphs
    """
    extractor = StatementGraphExtractor(context_length=2)
    graph_files = list(graphs_dir.glob("*.json"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for graph_file in graph_files:
        with open(graph_file, 'r') as f:
            data = json.load(f)
        
        # 检查数据格式
        if "graphs" not in data:
            print(f"Warning: {graph_file.name} does not contain 'graphs' key. Skipping...")
            continue
        
        pdg = data["graphs"].get("pdg", {})
        ast = data["graphs"].get("ast", {})
        
        if not pdg or not ast:
            print(f"Warning: {graph_file.name} does not contain valid PDG or AST. Skipping...")
            continue
        
        statement_graphs = extractor.extract_from_function(pdg, ast)
        
        output_file = output_dir / f"{data['contract_id']}_statement_graphs.json"
        with open(output_file, 'w') as f:
            json.dump(statement_graphs, f, indent=2)
        
        print(f"Processed {graph_file.name} -> {output_file.name}")

if __name__ == "__main__":
    graphs_dir = Path("data/graphs/smartbugs")
    output_dir = Path("data/graphs/statement_graphs")
    build_statement_graphs(graphs_dir, output_dir)