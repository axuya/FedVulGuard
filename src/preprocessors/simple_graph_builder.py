#!/usr/bin/env python3
"""
简化版图构建器 - 不依赖 Slither
基于代码模式匹配和 AST 解析
"""

import json
import re
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGraphBuilder:
    """简化版多图构建器"""
    
    def __init__(self, output_dir: str = "data/graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def parse_contract(self, code: str) -> Dict:
        """解析合约代码，提取关键信息"""
        info = {
            'contract_name': None,
            'functions': [],
            'state_variables': [],
            'modifiers': [],
            'events': []
        }
        
        # 提取合约名
        contract_match = re.search(r'contract\s+(\w+)', code)
        if contract_match:
            info['contract_name'] = contract_match.group(1)
        
        # 提取函数
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*(public|external|internal|private)?'
        for match in re.finditer(function_pattern, code):
            info['functions'].append({
                'name': match.group(1),
                'visibility': match.group(2) or 'public'
            })
        
        # 提取状态变量
        state_var_pattern = r'(uint|int|address|bool|string|bytes)\d*\s+(public|private|internal)?\s*(\w+)\s*;'
        for match in re.finditer(state_var_pattern, code):
            info['state_variables'].append({
                'type': match.group(1),
                'visibility': match.group(2) or 'internal',
                'name': match.group(3)
            })
        
        return info
    
    def build_ast(self, code: str, contract_id: str) -> nx.DiGraph:
        """构建抽象语法树"""
        G = nx.DiGraph()
        
        info = self.parse_contract(code)
        
        # 根节点：合约
        contract_node = f"{contract_id}_contract_0"
        G.add_node(contract_node, 
                  node_type='contract',
                  name=info['contract_name'] or 'UnknownContract')
        
        node_counter = 1
        
        # 状态变量节点
        for var in info['state_variables']:
            var_node = f"{contract_id}_var_{node_counter}"
            G.add_node(var_node,
                      node_type='state_variable',
                      name=var['name'],
                      var_type=var['type'])
            G.add_edge(contract_node, var_node, edge_type='contains')
            node_counter += 1
        
        # 函数节点
        for func in info['functions']:
            func_node = f"{contract_id}_func_{node_counter}"
            G.add_node(func_node,
                      node_type='function',
                      name=func['name'],
                      visibility=func['visibility'])
            G.add_edge(contract_node, func_node, edge_type='contains')
            node_counter += 1
            
            # 函数体内的语句（简化：按行分割）
            func_body = self._extract_function_body(code, func['name'])
            if func_body:
                lines = [line.strip() for line in func_body.split('\n') if line.strip()]
                for i, line in enumerate(lines[:10]):  # 最多10个语句节点
                    stmt_node = f"{contract_id}_stmt_{node_counter}"
                    G.add_node(stmt_node,
                              node_type='statement',
                              content=line[:100])  # 截断长语句
                    G.add_edge(func_node, stmt_node, edge_type='contains')
                    node_counter += 1
        
        return G
    
    def build_cfg(self, code: str, contract_id: str) -> nx.DiGraph:
        """构建控制流图"""
        G = nx.DiGraph()
        
        info = self.parse_contract(code)
        node_counter = 0
        
        for func in info['functions']:
            func_body = self._extract_function_body(code, func['name'])
            if not func_body:
                continue
            
            # 入口节点
            entry_node = f"{contract_id}_cfg_entry_{node_counter}"
            G.add_node(entry_node, 
                      node_type='entry',
                      function=func['name'])
            node_counter += 1
            
            prev_node = entry_node
            
            # 解析控制流关键字
            lines = func_body.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                stmt_node = f"{contract_id}_cfg_{node_counter}"
                node_counter += 1
                
                # 判断语句类型
                if re.search(r'\bif\s*\(', line):
                    G.add_node(stmt_node, node_type='if_statement')
                    G.add_edge(prev_node, stmt_node, edge_type='control_flow')
                    # 创建分支
                    true_branch = f"{contract_id}_cfg_{node_counter}"
                    false_branch = f"{contract_id}_cfg_{node_counter + 1}"
                    node_counter += 2
                    G.add_node(true_branch, node_type='true_branch')
                    G.add_node(false_branch, node_type='false_branch')
                    G.add_edge(stmt_node, true_branch, edge_type='true')
                    G.add_edge(stmt_node, false_branch, edge_type='false')
                    prev_node = true_branch
                    
                elif re.search(r'\bfor\s*\(', line) or re.search(r'\bwhile\s*\(', line):
                    G.add_node(stmt_node, node_type='loop')
                    G.add_edge(prev_node, stmt_node, edge_type='control_flow')
                    # 循环回边
                    G.add_edge(stmt_node, stmt_node, edge_type='loop_back')
                    prev_node = stmt_node
                    
                elif 'return' in line:
                    G.add_node(stmt_node, node_type='return')
                    G.add_edge(prev_node, stmt_node, edge_type='control_flow')
                    # 退出节点
                    exit_node = f"{contract_id}_cfg_exit_{node_counter}"
                    G.add_node(exit_node, node_type='exit')
                    G.add_edge(stmt_node, exit_node, edge_type='control_flow')
                    node_counter += 1
                    break
                    
                else:
                    G.add_node(stmt_node, node_type='statement', content=line[:50])
                    G.add_edge(prev_node, stmt_node, edge_type='control_flow')
                    prev_node = stmt_node
        
        return G
    
    def build_dfg(self, code: str, contract_id: str) -> nx.DiGraph:
        """构建数据流图"""
        G = nx.DiGraph()
        
        info = self.parse_contract(code)
        node_counter = 0
        
        # 状态变量节点
        var_nodes = {}
        for var in info['state_variables']:
            var_node = f"{contract_id}_dfg_var_{node_counter}"
            G.add_node(var_node,
                      node_type='variable',
                      name=var['name'],
                      var_type=var['type'])
            var_nodes[var['name']] = var_node
            node_counter += 1
        
        # 分析每个函数的数据流
        for func in info['functions']:
            func_body = self._extract_function_body(code, func['name'])
            if not func_body:
                continue
            
            lines = func_body.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                
                # 检测读操作
                for var_name, var_node in var_nodes.items():
                    if re.search(rf'\b{var_name}\b', line) and '=' not in line.split(var_name)[0]:
                        read_node = f"{contract_id}_dfg_read_{node_counter}"
                        G.add_node(read_node, node_type='read', var=var_name)
                        G.add_edge(var_node, read_node, edge_type='data_flow')
                        node_counter += 1
                
                # 检测写操作
                write_match = re.search(r'(\w+)\s*=', line)
                if write_match:
                    written_var = write_match.group(1)
                    if written_var in var_nodes:
                        write_node = f"{contract_id}_dfg_write_{node_counter}"
                        G.add_node(write_node, node_type='write', var=written_var)
                        G.add_edge(write_node, var_nodes[written_var], edge_type='data_flow')
                        node_counter += 1
        
        return G
    
    def build_pdg(self, code: str, contract_id: str) -> nx.DiGraph:
        """构建程序依赖图 (CFG + DFG 的组合)"""
        cfg = self.build_cfg(code, contract_id)
        dfg = self.build_dfg(code, contract_id)
        
        # 合并两个图
        G = nx.DiGraph()
        
        # 添加 CFG
        for node, data in cfg.nodes(data=True):
            G.add_node(node, **data)
        for u, v, data in cfg.edges(data=True):
            G.add_edge(u, v, **data)
        
        # 添加 DFG
        for node, data in dfg.nodes(data=True):
            if node not in G:
                G.add_node(node, **data)
        for u, v, data in dfg.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)
        
        return G
    
    def _extract_function_body(self, code: str, func_name: str) -> str:
        """提取函数体"""
        pattern = rf'function\s+{func_name}\s*\([^)]*\)[^{{]*\{{([^}}]*)\}}'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            return match.group(1)
        return ""
    
    def build_all_graphs(self, code: str, contract_id: str) -> Dict:
        """构建所有图"""
        return {
            'contract_id': contract_id,
            'ast': self.build_ast(code, contract_id),
            'cfg': self.build_cfg(code, contract_id),
            'dfg': self.build_dfg(code, contract_id),
            'pdg': self.build_pdg(code, contract_id),
            'metadata': {
                'sloc': code.count('\n') + 1,
                'num_functions': len(re.findall(r'function\s+\w+', code))
            }
        }
    
    def save_graphs(self, graphs: Dict, output_path: Path):
        """保存图到 JSON"""
        serializable = {
            'contract_id': graphs['contract_id'],
            'metadata': graphs['metadata'],
            'ast': nx.node_link_data(graphs['ast']),
            'cfg': nx.node_link_data(graphs['cfg']),
            'dfg': nx.node_link_data(graphs['dfg']),
            'pdg': nx.node_link_data(graphs['pdg']),
        }
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def process_spc_dataset(self, spc_file: str):
        """处理 SPC 数据集"""
        logger.info(f"Processing: {spc_file}")
        
        with open(spc_file, 'r') as f:
            spc_pairs = json.load(f)
        
        logger.info(f"Found {len(spc_pairs)} SPC pairs")
        
        results = []
        success_count = 0
        
        for pair in tqdm(spc_pairs, desc="Building graphs"):
            pair_id = pair['pair_id']
            
            try:
                # Before graphs
                before_graphs = self.build_all_graphs(
                    pair['code_before'],
                    f"{pair_id}_before"
                )
                before_path = self.output_dir / f"{pair_id}_before.json"
                self.save_graphs(before_graphs, before_path)
                
                # After graphs
                after_graphs = self.build_all_graphs(
                    pair['code_after'],
                    f"{pair_id}_after"
                )
                after_path = self.output_dir / f"{pair_id}_after.json"
                self.save_graphs(after_graphs, after_path)
                
                results.append({
                    'pair_id': pair_id,
                    'vulnerability_type': pair.get('vulnerability_type'),
                    'before_graphs': str(before_path),
                    'after_graphs': str(after_path),
                    'metadata': {
                        'before': before_graphs['metadata'],
                        'after': after_graphs['metadata']
                    }
                })
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {pair_id}: {e}")
        
        # 保存索引
        index_path = self.output_dir / 'graph_index.json'
        with open(index_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Successfully processed {success_count}/{len(spc_pairs)} pairs")
        logger.info(f"📁 Graphs saved to: {self.output_dir}")
        logger.info(f"📋 Index: {index_path}")
        
        return results


def main():
    print("="*70)
    print("🔧 Simple Multi-Graph Builder")
    print("="*70)
    
    # 初始化
    builder = SimpleGraphBuilder()
    
    # SPC 数据集路径
    spc_file = "data/spc_data/processed/bootstrap_filtered_60.json"
    
    if not Path(spc_file).exists():
        print(f"\n❌ File not found: {spc_file}")
        print("\n💡 Available options:")
        
        # 尝试其他可能的文件
        alternatives = [
            "data/spc_data/processed/bootstrap_spc_dataset.json",
            "data/spc_data/processed/bootstrap_classified.json",
            "data/spc_data/processed/all_spc_pairs.json"
        ]
        
        for alt in alternatives:
            if Path(alt).exists():
                print(f"   ✓ {alt}")
                response = input(f"\n使用这个文件? (y/n): ")
                if response.lower() == 'y':
                    spc_file = alt
                    break
        else:
            print("\n❌ No SPC dataset found!")
            return
    
    # 处理数据集
    results = builder.process_spc_dataset(spc_file)
    
    # 统计
    print("\n" + "="*70)
    print("📊 Graph Building Results")
    print("="*70)
    print(f"Total pairs: {len(results)}")
    print(f"Total graphs: {len(results) * 2}")
    print(f"Output: {builder.output_dir}")
    
    # 验证
    if results:
        sample = results[0]
        print(f"\n📝 Sample:")
        print(f"   Pair ID: {sample['pair_id']}")
        print(f"   Before: {sample['before_graphs']}")
        print(f"   After: {sample['after_graphs']}")
        
        # 检查图的大小
        with open(sample['before_graphs'], 'r') as f:
            graph_data = json.load(f)
            print(f"\n📊 Graph sizes (before):")
            for graph_type in ['ast', 'cfg', 'dfg', 'pdg']:
                g = graph_data[graph_type]
                print(f"   {graph_type.upper()}: {len(g['nodes'])} nodes, {len(g['links'])} edges")
    
    print("\n✅ Phase 2 Complete!")
    print("\n💡 Next: Feature extraction and GNN training (Phase 3)")


if __name__ == "__main__":
    main()