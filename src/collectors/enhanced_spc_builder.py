#!/usr/bin/env python3
"""
增强版 SPC 构建器 - 提高匹配率和数据量
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from difflib import SequenceMatcher
import random

class EnhancedSPCBuilder:
    def __init__(self):
        self.smartbugs_dir = Path("/home/xu/FedVulGuard/data/raw/smartbugs/smartbugs")
        self.solidifi_dir = Path("/home/xu/FedVulGuard/data/raw/solidifi")
        self.output_dir = Path("data/spc_data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_contracts(self):
        """加载合约并增强漏洞类型识别"""
        contracts = []
        
        print("📂 Loading contracts from datasets...")
        
        # SmartBugs
        if self.smartbugs_dir.exists():
            for sol_file in self.smartbugs_dir.rglob("*.sol"):
                try:
                    with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    if len(code) < 100:  # 跳过太小的文件
                        continue
                    
                    # 增强的漏洞类型识别
                    vuln_type = self._enhanced_vuln_detection(code, str(sol_file))
                    
                    contracts.append({
                        'source': 'smartbugs',
                        'path': str(sol_file),
                        'code': code,
                        'vulnerability_type': vuln_type,
                        'name': sol_file.stem,
                        'has_vulnerability': self._detect_vulnerable_patterns(code)
                    })
                except Exception as e:
                    continue
        
        # SolidiFI
        if self.solidifi_dir.exists():
            for sol_file in self.solidifi_dir.rglob("*.sol"):
                try:
                    with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    if len(code) < 100:
                        continue
                    
                    vuln_type = self._enhanced_vuln_detection(code, str(sol_file))
                    
                    contracts.append({
                        'source': 'solidifi',
                        'path': str(sol_file),
                        'code': code,
                        'vulnerability_type': vuln_type,
                        'name': sol_file.stem,
                        'has_vulnerability': self._detect_vulnerable_patterns(code)
                    })
                except Exception as e:
                    continue
        
        print(f"✅ Loaded {len(contracts)} contracts")
        
        # 统计漏洞类型
        vuln_stats = {}
        for c in contracts:
            vtype = c['vulnerability_type']
            vuln_stats[vtype] = vuln_stats.get(vtype, 0) + 1
        
        print("\n📊 Vulnerability types found:")
        for vtype, count in sorted(vuln_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {vtype}: {count}")
        
        return contracts
    
    def _enhanced_vuln_detection(self, code: str, path: str) -> str:
        """增强的漏洞检测 - 同时检查路径和代码内容"""
        path_lower = path.lower()
        code_lower = code.lower()
        
        # 从路径检测
        path_patterns = {
            'reentrancy': ['reentrancy', 'reentrant', 'dao', 're-entrance', 'cross_function'],
            'overflow': ['overflow', 'integer', 'int_bug'],
            'underflow': ['underflow'],
            'access_control': ['access', 'unprotected', 'permission', 'auth', 'owner'],
            'tx_origin': ['tx_origin', 'txorigin'],
            'timestamp': ['timestamp', 'time', 'now'],
            'unchecked_call': ['unchecked', 'call', 'delegatecall'],
            'dos': ['dos', 'denial', 'gas'],
            'bad_randomness': ['random', 'blockhash']
        }
        
        for vuln_type, keywords in path_patterns.items():
            if any(kw in path_lower for kw in keywords):
                return vuln_type
        
        # 从代码内容检测
        code_patterns = {
            'reentrancy': [
                r'\.call\.value\(',
                r'\.call\{value:',
                r'\.transfer\(',
                r'\.send\(',
                r'msg\.sender\.call'
            ],
            'overflow': [
                r'\w+\s*\+\s*\w+',
                r'\w+\s*\*\s*\w+',
                r'uint\d*\s+\w+\s*='
            ],
            'access_control': [
                r'function\s+\w+\s*\([^\)]*\)\s*public',
                r'selfdestruct\(',
                r'suicide\('
            ],
            'tx_origin': [
                r'tx\.origin',
                r'require\(tx\.origin'
            ],
            'timestamp': [
                r'now\s*[><=]',
                r'block\.timestamp',
                r'block\.number'
            ],
            'unchecked_call': [
                r'\.call\(',
                r'\.delegatecall\('
            ]
        }
        
        for vuln_type, patterns in code_patterns.items():
            if any(re.search(pattern, code) for pattern in patterns):
                return vuln_type
        
        return 'unknown'
    
    def _detect_vulnerable_patterns(self, code: str) -> bool:
        """检测代码是否包含已知的漏洞模式"""
        vulnerable_indicators = [
            r'\.call\.value\(',  # 旧式外部调用
            r'tx\.origin',  # tx.origin 使用
            r'selfdestruct\(',  # 自毁函数
            r'msg\.sender\.call',  # 未保护的调用
            # 缺少保护措施
            'SafeMath' not in code and r'\w+\s*\+\s*\w+',  # 无 SafeMath 的运算
            'nonReentrant' not in code and r'\.call\{value:',  # 无重入保护
        ]
        
        return any(
            (isinstance(pattern, str) and pattern in code) or
            (not isinstance(pattern, str) and re.search(pattern, code))
            for pattern in vulnerable_indicators
        )
    
    def create_enhanced_patches(self, contracts: List[Dict]) -> List[Dict]:
        """创建多种修复版本"""
        pairs = []
        
        print("\n🔧 Creating enhanced patches...")
        
        for contract in contracts:
            code = contract['code']
            vuln_type = contract['vulnerability_type']
            
            # 只修复有漏洞的合约
            if not contract.get('has_vulnerability', True):
                continue
            
            patches = []
            
            # 为每种漏洞类型创建多个修复版本
            if vuln_type == 'reentrancy':
                patches.extend(self._create_reentrancy_patches(code))
            elif vuln_type == 'overflow':
                patches.extend(self._create_overflow_patches(code))
            elif vuln_type == 'access_control':
                patches.extend(self._create_access_control_patches(code))
            elif vuln_type == 'tx_origin':
                patches.extend(self._create_tx_origin_patches(code))
            elif vuln_type == 'unchecked_call':
                patches.extend(self._create_unchecked_call_patches(code))
            elif vuln_type == 'timestamp':
                patches.extend(self._create_timestamp_patches(code))
            else:
                # 对 unknown 类型也尝试通用修复
                patches.extend(self._create_generic_patches(code))
            
            # 创建 SPC 对
            for i, patched_code in enumerate(patches):
                if patched_code != code:  # 确保有变化
                    similarity = SequenceMatcher(None, code, patched_code).ratio()
                    
                    if 0.6 < similarity < 0.99:  # 相似但有变化
                        pair = {
                            'pair_id': f"patch_{len(pairs):04d}",
                            'source': contract['source'],
                            'original_file': contract['name'],
                            'code_before': code,
                            'code_after': patched_code,
                            'vulnerability_type': vuln_type,
                            'similarity': round(similarity, 3),
                            'patch_method': f'method_{i+1}',
                            'label_before': 'vulnerable',
                            'label_after': 'patched',
                            'needs_manual_review': True
                        }
                        pairs.append(pair)
        
        print(f"✅ Created {len(pairs)} patched pairs")
        return pairs
    
    def _create_reentrancy_patches(self, code: str) -> List[str]:
        """创建多种重入漏洞修复"""
        patches = []
        
        # 方法1: 添加 ReentrancyGuard
        if 'ReentrancyGuard' not in code:
            p1 = code
            if 'pragma solidity' in p1:
                p1 = p1.replace(
                    'pragma solidity',
                    'import "@openzeppelin/contracts/security/ReentrancyGuard.sol";\n\npragma solidity'
                )
            p1 = re.sub(r'contract\s+(\w+)', r'contract \1 is ReentrancyGuard', p1, count=1)
            p1 = re.sub(
                r'function\s+(withdraw|claim|redeem)\s*\([^\)]*\)\s*(public|external)',
                r'function \1() \2 nonReentrant',
                p1
            )
            patches.append(p1)
        
        # 方法2: CEI 模式 - 状态更新放在调用前
        if '.call{value:' in code or '.transfer(' in code:
            p2 = code
            # 简单的模式：找到余额更新和外部调用，交换顺序
            # 这是简化版，实际需要更复杂的AST分析
            lines = p2.split('\n')
            new_lines = []
            balance_update = None
            external_call = None
            
            for line in lines:
                if 'balance[' in line.lower() and '=' in line and '-=' in line:
                    balance_update = line
                    continue
                if '.call{value:' in line or '.transfer(' in line:
                    external_call = line
                    if balance_update:
                        new_lines.append(balance_update)
                        balance_update = None
                    new_lines.append(external_call)
                    continue
                new_lines.append(line)
            
            p2 = '\n'.join(new_lines)
            if p2 != code:
                patches.append(p2)
        
        # 方法3: 使用 mutex 锁
        p3 = code
        if 'bool private locked' not in p3:
            # 添加锁变量
            p3 = re.sub(
                r'(contract\s+\w+[^{]*\{)',
                r'\1\n    bool private locked = false;',
                p3,
                count=1
            )
            # 在函数开头添加锁检查
            p3 = re.sub(
                r'(function\s+withdraw[^{]*\{)',
                r'\1\n        require(!locked, "Reentrant call");\n        locked = true;',
                p3
            )
            # 在函数结尾解锁
            p3 = re.sub(
                r'(\n\s*\})',
                r'\n        locked = false;\1',
                p3
            )
            patches.append(p3)
        
        return patches
    
    def _create_overflow_patches(self, code: str) -> List[str]:
        """创建溢出修复"""
        patches = []
        
        # 方法1: SafeMath
        if 'SafeMath' not in code and ('uint' in code):
            p1 = code
            p1 = p1.replace(
                'pragma solidity',
                'import "@openzeppelin/contracts/utils/math/SafeMath.sol";\n\npragma solidity'
            )
            p1 = re.sub(
                r'(contract\s+\w+[^{]*\{)',
                r'\1\n    using SafeMath for uint256;',
                p1,
                count=1
            )
            # 替换运算符
            p1 = re.sub(r'(\w+)\s*\+=\s*(\w+)', r'\1 = \1.add(\2)', p1)
            p1 = re.sub(r'(\w+)\s*-=\s*(\w+)', r'\1 = \1.sub(\2)', p1)
            p1 = re.sub(r'(\w+)\s*\*=\s*(\w+)', r'\1 = \1.mul(\2)', p1)
            patches.append(p1)
        
        # 方法2: 升级到 Solidity 0.8+
        if re.search(r'pragma solidity\s*[\^]?0\.[0-7]', code):
            p2 = re.sub(
                r'pragma solidity\s*[\^]?0\.[0-7]\.\d+',
                'pragma solidity ^0.8.0',
                code
            )
            patches.append(p2)
        
        # 方法3: 添加 require 检查
        p3 = code
        p3 = re.sub(
            r'(\w+)\s*\+=\s*(\w+);',
            r'require(\1 + \2 >= \1, "Overflow");\n        \1 += \2;',
            p3
        )
        if p3 != code:
            patches.append(p3)
        
        return patches
    
    def _create_access_control_patches(self, code: str) -> List[str]:
        """创建访问控制修复"""
        patches = []
        
        # 添加 Ownable
        if 'Ownable' not in code:
            p1 = code
            p1 = p1.replace(
                'pragma solidity',
                'import "@openzeppelin/contracts/access/Ownable.sol";\n\npragma solidity'
            )
            p1 = re.sub(r'contract\s+(\w+)', r'contract \1 is Ownable', p1, count=1)
            
            # 给敏感函数添加 onlyOwner
            sensitive_funcs = ['destroy', 'kill', 'selfdestruct', 'withdraw', 'set']
            for func in sensitive_funcs:
                p1 = re.sub(
                    rf'function\s+{func}\w*\s*\([^\)]*\)\s*(public|external)',
                    rf'function {func}() \1 onlyOwner',
                    p1
                )
            
            patches.append(p1)
        
        return patches
    
    def _create_tx_origin_patches(self, code: str) -> List[str]:
        """修复 tx.origin"""
        patches = []
        
        if 'tx.origin' in code:
            p1 = code.replace('tx.origin', 'msg.sender')
            patches.append(p1)
        
        return patches
    
    def _create_unchecked_call_patches(self, code: str) -> List[str]:
        """修复未检查的调用"""
        patches = []
        
        # 添加返回值检查
        p1 = re.sub(
            r'(\w+)\.call\{value:\s*(\w+)\}\(""\);',
            r'(bool success, ) = \1.call{value: \2}("");\n        require(success, "Call failed");',
            code
        )
        if p1 != code:
            patches.append(p1)
        
        return patches
    
    def _create_timestamp_patches(self, code: str) -> List[str]:
        """修复时间戳依赖"""
        patches = []
        
        # 添加时间范围检查
        if 'block.timestamp' in code:
            p1 = re.sub(
                r'require\(block\.timestamp\s*([><=]+)\s*(\w+)\)',
                r'require(block.timestamp \1 \2 && block.timestamp \1 \2 + 900, "Invalid time")',
                code
            )
            if p1 != code:
                patches.append(p1)
        
        return patches
    
    def _create_generic_patches(self, code: str) -> List[str]:
        """通用修复（for unknown类型）"""
        patches = []
        
        # 添加基本的安全措施
        if 'ReentrancyGuard' not in code and '.call' in code:
            patches.extend(self._create_reentrancy_patches(code))
        
        if 'SafeMath' not in code and any(op in code for op in ['+', '-', '*']):
            patches.extend(self._create_overflow_patches(code))
        
        if 'onlyOwner' not in code and 'selfdestruct' in code:
            patches.extend(self._create_access_control_patches(code))
        
        return patches
    
    def create_code_variations(self, contracts: List[Dict]) -> List[Dict]:
        """创建代码变体（轻微修改产生相似对）"""
        pairs = []
        
        print("\n🔄 Creating code variations...")
        
        for contract in contracts[:30]:  # 限制数量
            code = contract['code']
            
            variations = [
                self._rename_variables(code),
                self._reorder_functions(code),
                self._add_comments(code),
                self._change_formatting(code)
            ]
            
            for i, variant in enumerate(variations):
                if variant and variant != code:
                    similarity = SequenceMatcher(None, code, variant).ratio()
                    
                    if 0.85 < similarity < 0.99:
                        pair = {
                            'pair_id': f"var_{len(pairs):04d}",
                            'source': contract['source'],
                            'original_file': contract['name'],
                            'code_before': code,
                            'code_after': variant,
                            'vulnerability_type': contract['vulnerability_type'],
                            'similarity': round(similarity, 3),
                            'method': f'variation_{i+1}',
                            'label_before': 'similar',
                            'label_after': 'similar',
                            'needs_manual_review': True
                        }
                        pairs.append(pair)
        
        print(f"✅ Created {len(pairs)} variation pairs")
        return pairs
    
    def _rename_variables(self, code: str) -> str:
        """重命名变量"""
        # 简单的变量重命名
        replacements = {
            r'\bbalance\b': 'userBalance',
            r'\bamount\b': 'transferAmount',
            r'\bowner\b': 'contractOwner',
            r'\bvalue\b': 'ethValue'
        }
        
        modified = code
        for pattern, replacement in replacements.items():
            modified = re.sub(pattern, replacement, modified)
        
        return modified if modified != code else None
    
    def _reorder_functions(self, code: str) -> str:
        """重新排序函数（简化版）"""
        # 这是简化实现，实际需要AST解析
        return None  # 跳过这个比较复杂的操作
    
    def _add_comments(self, code: str) -> str:
        """添加注释"""
        lines = code.split('\n')
        new_lines = []
        
        for line in lines:
            if 'function' in line and 'public' in line:
                new_lines.append('    /// @notice Public function')
            new_lines.append(line)
        
        modified = '\n'.join(new_lines)
        return modified if modified != code else None
    
    def _change_formatting(self, code: str) -> str:
        """改变代码格式"""
        # 添加/删除空格
        modified = re.sub(r'  +', '    ', code)  # 统一缩进
        return modified if modified != code else None
    
    def build_all(self):
        """构建所有数据"""
        print("="*60)
        print("🚀 Enhanced SPC Builder")
        print("="*60)
        
        # 1. 加载合约
        contracts = self.load_contracts()
        
        if not contracts:
            print("\n❌ No contracts found!")
            return []
        
        # 2. 创建增强的修复对
        patched_pairs = self.create_enhanced_patches(contracts)
        
        # 3. 创建代码变体
        variation_pairs = self.create_code_variations(contracts)
        
        # 4. 合并
        all_pairs = patched_pairs + variation_pairs
        
        # 5. 保存
        output_file = self.output_dir / 'spc_pairs_enhanced.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
        # 统计
        print("\n" + "="*60)
        print("📊 Enhanced Summary")
        print("="*60)
        print(f"Total contracts: {len(contracts)}")
        print(f"Patched pairs: {len(patched_pairs)}")
        print(f"Variation pairs: {len(variation_pairs)}")
        print(f"Total SPC pairs: {len(all_pairs)}")
        print(f"\n✅ Saved to: {output_file}")
        
        # 漏洞分布
        vuln_dist = {}
        for pair in all_pairs:
            vtype = pair.get('vulnerability_type', 'unknown')
            vuln_dist[vtype] = vuln_dist.get(vtype, 0) + 1
        
        print("\n📈 Pair Distribution:")
        for vtype, count in sorted(vuln_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"   {vtype}: {count}")
        
        return all_pairs


def main():
    builder = EnhancedSPCBuilder()
    pairs = builder.build_all()
    
    if pairs:
        print("\n💡 Tips:")
        print("   - Review pairs manually for quality")
        print("   - Focus on pairs with similarity 0.7-0.95")
        print("   - Bootstrap 只需要 50-100 个高质量对")


if __name__ == "__main__":
    main()