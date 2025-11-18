#!/usr/bin/env python3
"""
SmartBugs图构建脚本 - 完全独立版本
不依赖其他模块，直接构建图
"""

import json
import re
from pathlib import Path
from typing import Dict, List

class SimpleGraphBuilder:
    """简单的Solidity图构建器"""
    
    def __init__(self):
        self.node_id = 0
    
    def _get_next_id(self):
        """生成唯一节点ID"""
        self.node_id += 1
        return self.node_id
    
    def build_ast(self, code: str) -> Dict:
        """构建简化的AST"""
        nodes = []
        edges = []
        
        # 提取合约名
        contract_match = re.search(r'contract\s+(\w+)', code)
        if contract_match:
            contract_id = self._get_next_id()
            nodes.append({
                'id': contract_id,
                'type': 'contract',
                'name': contract_match.group(1)
            })
        
        # 提取函数
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(function_pattern, code):
            func_id = self._get_next_id()
            nodes.append({
                'id': func_id,
                'type': 'function',
                'name': match.group(1)
            })
            if contract_id:
                edges.append({'from': contract_id, 'to': func_id})
        
        # 提取状态变量
        var_pattern = r'(uint|address|bool|string|bytes)\s+(\w+)\s*;'
        for match in re.finditer(var_pattern, code):
            var_id = self._get_next_id()
            nodes.append({
                'id': var_id,
                'type': 'variable',
                'var_type': match.group(1),
                'name': match.group(2)
            })
            if contract_id:
                edges.append({'from': contract_id, 'to': var_id})
        
        return {'nodes': nodes, 'edges': edges}
    
    def build_cfg(self, code: str) -> Dict:
        """构建简化的CFG"""
        nodes = []
        edges = []
        
        # 为每个语句创建节点
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        prev_id = None
        for i, line in enumerate(lines):
            if line and not line.startswith('//'):
                node_id = self._get_next_id()
                nodes.append({
                    'id': node_id,
                    'type': 'statement',
                    'content': line[:50]  # 截断长语句
                })
                
                if prev_id:
                    edges.append({'from': prev_id, 'to': node_id})
                prev_id = node_id
        
        return {'nodes': nodes, 'edges': edges}
    
    def build_dfg(self, code: str) -> Dict:
        """构建简化的DFG"""
        nodes = []
        edges = []
        
        # 提取变量定义和使用
        var_defs = {}
        
        # 查找赋值语句
        assignment_pattern = r'(\w+)\s*=\s*([^;]+);'
        for match in re.finditer(assignment_pattern, code):
            var_name = match.group(1)
            var_id = self._get_next_id()
            
            nodes.append({
                'id': var_id,
                'type': 'data_node',
                'name': var_name
            })
            
            var_defs[var_name] = var_id
        
        return {'nodes': nodes, 'edges': edges}
    
    def build_pdg(self, code: str) -> Dict:
        """构建简化的PDG（组合CFG和DFG）"""
        cfg = self.build_cfg(code)
        dfg = self.build_dfg(code)
        
        # 简单合并
        all_nodes = cfg['nodes'] + dfg['nodes']
        all_edges = cfg['edges'] + dfg['edges']
        
        return {'nodes': all_nodes, 'edges': all_edges}
    
    def build_all_graphs(self, code: str) -> Dict:
        """构建所有图"""
        self.node_id = 0  # 重置ID计数器
        
        return {
            'ast': self.build_ast(code),
            'cfg': self.build_cfg(code),
            'dfg': self.build_dfg(code),
            'pdg': self.build_pdg(code)
        }
def main():
    print("="*70)
    print("SmartBugs 图构建工具 - 独立版本 (修复版)")
    print("="*70)
    
    # 路径配置
    samples_dir = Path("data/raw/smartbugs/smartbugs/samples")
    output_dir = Path("data/graphs/smartbugs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目录
    if not samples_dir.exists():
        print(f"❌ 错误: 目录不存在 {samples_dir}")
        return
    
    # 收集所有.sol文件
    sol_files = list(samples_dir.glob("**/*.sol"))
    print(f"\n找到 {len(sol_files)} 个Solidity合约")
    
    if len(sol_files) == 0:
        print("❌ 未找到.sol文件")
        return
    
    # 版本统计
    version_stats = {}
    for sol_file in sol_files:
        version = sol_file.parent.name
        version_stats[version] = version_stats.get(version, 0) + 1
    
    print("\n版本分布:")
    for version, count in sorted(version_stats.items()):
        print(f"  {version:10s}: {count:2d} 个")
    
    print(f"\n输出目录: {output_dir}\n")
    
    # 初始化构建器
    builder = SimpleGraphBuilder()
    
    # 统计
    success = 0
    failed = 0
    failed_files = []
    
    # 构建每个合约的图
    for i, sol_file in enumerate(sol_files, 1):
        version = sol_file.parent.name
        contract_name = sol_file.stem
        
        try:
            # 显示进度
            print(f"[{i:2d}/{len(sol_files)}] {version:10s} {contract_name:35s}", end=" ")
            
            # 检查文件是否存在
            if not sol_file.exists():
                raise FileNotFoundError(f"文件不存在: {sol_file}")
            
            # 读取代码
            with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            if not code.strip():
                raise ValueError("文件为空")
            
            # 构建图
            graphs = builder.build_all_graphs(code)
            
            # 准备输出数据
            graph_data = {
                'contract_id': f"{version}_{contract_name}",
                'contract_name': contract_name,
                'solidity_version': version,
                'source_file': str(sol_file),
                'graphs': graphs,
                'statistics': {
                    'ast_nodes': len(graphs['ast']['nodes']),
                    'ast_edges': len(graphs['ast']['edges']),
                    'cfg_nodes': len(graphs['cfg']['nodes']),
                    'cfg_edges': len(graphs['cfg']['edges']),
                    'dfg_nodes': len(graphs['dfg']['nodes']),
                    'dfg_edges': len(graphs['dfg']['edges']),
                    'pdg_nodes': len(graphs['pdg']['nodes']),
                    'pdg_edges': len(graphs['pdg']['edges']),
                }
            }
            
            # 保存
            output_file = output_dir / f"{version}_{contract_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            # 显示统计
            ast_nodes = len(graphs['ast']['nodes'])
            cfg_nodes = len(graphs['cfg']['nodes'])
            print(f"✓ AST:{ast_nodes:3d} CFG:{cfg_nodes:3d}")
            
            success += 1
            
        except Exception as e:
            error_msg = str(e)[:50]
            print(f"✗ {error_msg}")
            failed += 1
            failed_files.append({
                'file': sol_file.name,
                'version': version,
                'path': str(sol_file),
                'error': str(e)
            })
    
    # 输出总结
    print("\n" + "="*70)
    print("构建完成!")
    print("="*70)
    print(f"总计: {len(sol_files)}")
    print(f"成功: {success} ({success/len(sol_files)*100:.1f}%)")
    print(f"失败: {failed} ({failed/len(sol_files)*100:.1f}%)")
    
    # 保存索引文件
    index_data = {
        'total': len(sol_files),
        'success': success,
        'failed': failed,
        'version_stats': version_stats,
        'output_dir': str(output_dir),
        'source_dir': str(samples_dir)
    }
    
    with open(output_dir / 'index.json', 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"\n索引文件: {output_dir / 'index.json'}")
    
    # 保存失败列表
    if failed_files:
        print(f"\n失败文件详情:")
        for item in failed_files[:3]:
            print(f"\n  文件: {item['file']}")
            print(f"  路径: {item['path']}")
            print(f"  错误: {item['error']}")
        
        if len(failed_files) > 3:
            print(f"\n  ... 还有 {len(failed_files)-3} 个失败")
        
        with open(output_dir / 'failed_files.json', 'w') as f:
            json.dump({'count': len(failed_files), 'files': failed_files}, f, indent=2)
    
    print("\n" + "="*70)




if __name__ == "__main__":
    main()