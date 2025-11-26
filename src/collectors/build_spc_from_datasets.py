#!/usr/bin/env python3
"""
从 SmartBugs 和 SolidiFI 数据集构建 SPC 样本对
策略：使用漏洞合约和人工修复版本
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
from difflib import SequenceMatcher

class SPCBuilderFromDatasets:
    def __init__(self):
        self.smartbugs_dir = Path("/home/xu/FedVulGuard/data/raw/smartbugs/smartbugs")
        self.solidifi_dir = Path("/home/xu/FedVulGuard/data/raw/solidifi")
        self.output_dir = Path("data/spc_data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.spc_pairs = []
    
    def load_vulnerable_contracts(self):
        """从数据集加载漏洞合约"""
        contracts = []
        
        # 从 SmartBugs 加载
        if self.smartbugs_dir.exists():
            print(f"📂 Scanning SmartBugs: {self.smartbugs_dir}")
            for sol_file in self.smartbugs_dir.rglob("*.sol"):
                try:
                    with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    # 推断漏洞类型（从文件路径或名称）
                    vuln_type = self._infer_vuln_from_path(str(sol_file))
                    
                    contracts.append({
                        'source': 'smartbugs',
                        'path': str(sol_file),
                        'code': code,
                        'vulnerability_type': vuln_type,
                        'name': sol_file.stem
                    })
                except Exception as e:
                    print(f"⚠️  Error reading {sol_file}: {e}")
        
        # 从 SolidiFI 加载
        if self.solidifi_dir.exists():
            print(f"📂 Scanning SolidiFI: {self.solidifi_dir}")
            for sol_file in self.solidifi_dir.rglob("*.sol"):
                try:
                    with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    vuln_type = self._infer_vuln_from_path(str(sol_file))
                    
                    contracts.append({
                        'source': 'solidifi',
                        'path': str(sol_file),
                        'code': code,
                        'vulnerability_type': vuln_type,
                        'name': sol_file.stem
                    })
                except Exception as e:
                    print(f"⚠️  Error reading {sol_file}: {e}")
        
        print(f"✅ Loaded {len(contracts)} vulnerable contracts")
        return contracts
    
    def _infer_vuln_from_path(self, path: str) -> str:
        """从文件路径推断漏洞类型"""
        path_lower = path.lower()
        
        vuln_keywords = {
            'reentrancy': ['reentrancy', 'reentrant', 're-entrance', 'dao'],
            'overflow': ['overflow', 'integer_overflow', 'int_overflow'],
            'underflow': ['underflow', 'integer_underflow'],
            'access_control': ['access', 'permission', 'authorization', 'unprotected'],
            'tx_origin': ['tx_origin', 'txorigin'],
            'timestamp': ['timestamp', 'time_manipulation', 'block_timestamp'],
            'unchecked_call': ['unchecked', 'call_injection', 'delegatecall'],
            'dos': ['dos', 'denial', 'loop'],
            'front_running': ['front', 'racing', 'race'],
            'bad_randomness': ['random', 'entropy']
        }
        
        for vuln_type, keywords in vuln_keywords.items():
            if any(kw in path_lower for kw in keywords):
                return vuln_type
        
        return 'unknown'
    
    def create_synthetic_patches(self, contracts: List[Dict]) -> List[Dict]:
        """
        创建合成修复对
        策略：对漏洞代码应用常见修复模式
        """
        pairs = []
        
        for contract in contracts:
            vuln_type = contract['vulnerability_type']
            code = contract['code']
            
            # 根据漏洞类型生成修复版本
            if vuln_type == 'reentrancy':
                patched_versions = self._patch_reentrancy(code)
            elif vuln_type == 'overflow':
                patched_versions = self._patch_overflow(code)
            elif vuln_type == 'access_control':
                patched_versions = self._patch_access_control(code)
            elif vuln_type == 'tx_origin':
                patched_versions = self._patch_tx_origin(code)
            elif vuln_type == 'unchecked_call':
                patched_versions = self._patch_unchecked_call(code)
            else:
                patched_versions = []
            
            for patched_code in patched_versions:
                # 计算相似度
                similarity = self._calculate_similarity(code, patched_code)
                
                if 0.7 < similarity < 0.99:  # 相似但不完全相同
                    pair = {
                        'pair_id': f"syn_{len(pairs):04d}",
                        'source': contract['source'],
                        'original_file': contract['name'],
                        'code_before': code,
                        'code_after': patched_code,
                        'vulnerability_type': vuln_type,
                        'similarity': similarity,
                        'label_before': 'vulnerable',
                        'label_after': 'patched',
                        'method': 'synthetic_patch',
                        'needs_manual_review': True
                    }
                    pairs.append(pair)
        
        return pairs
    
    def _patch_reentrancy(self, code: str) -> List[str]:
        """生成重入漏洞修复版本"""
        patches = []
        
        # 修复1: 添加 ReentrancyGuard
        if 'ReentrancyGuard' not in code and 'nonReentrant' not in code:
            # 添加导入
            patched = re.sub(
                r'(pragma solidity[^;]+;)',
                r'\1\nimport "@openzeppelin/contracts/security/ReentrancyGuard.sol";',
                code,
                count=1
            )
            # 添加继承
            patched = re.sub(
                r'(contract\s+\w+)',
                r'\1 is ReentrancyGuard',
                patched,
                count=1
            )
            # 添加 modifier 到函数
            patched = re.sub(
                r'(function\s+withdraw\s*\([^\)]*\)\s*public)',
                r'\1 nonReentrant',
                patched
            )
            patches.append(patched)
        
        # 修复2: Checks-Effects-Interactions 模式
        # 查找外部调用并移到最后
        lines = code.split('\n')
        patched_lines = []
        in_function = False
        external_calls = []
        
        for line in lines:
            if 'function' in line and ('public' in line or 'external' in line):
                in_function = True
            elif in_function and ('{' not in line or '}' in line):
                if '.call' in line or '.transfer' in line or '.send' in line:
                    external_calls.append(line)
                    continue
            patched_lines.append(line)
            if in_function and '}' in line:
                # 在函数结束前插入外部调用
                for call in external_calls:
                    patched_lines.insert(-1, call)
                external_calls = []
                in_function = False
        
        if patched_lines != lines:
            patches.append('\n'.join(patched_lines))
        
        return patches
    
    def _patch_overflow(self, code: str) -> List[str]:
        """生成溢出漏洞修复版本"""
        patches = []
        
        # 修复1: 使用 SafeMath
        if 'SafeMath' not in code:
            patched = re.sub(
                r'(pragma solidity[^;]+;)',
                r'\1\nimport "@openzeppelin/contracts/utils/math/SafeMath.sol";',
                code,
                count=1
            )
            patched = re.sub(
                r'(contract\s+\w+\s*{)',
                r'\1\n    using SafeMath for uint256;',
                patched,
                count=1
            )
            # 替换算术运算
            patched = re.sub(r'(\w+)\s*\+\s*(\w+)', r'\1.add(\2)', patched)
            patched = re.sub(r'(\w+)\s*-\s*(\w+)', r'\1.sub(\2)', patched)
            patched = re.sub(r'(\w+)\s*\*\s*(\w+)', r'\1.mul(\2)', patched)
            patches.append(patched)
        
        # 修复2: 使用 Solidity 0.8+ (内置溢出检查)
        if 'pragma solidity' in code:
            patched = re.sub(
                r'pragma solidity\s*[\^]?0\.[0-7]\.\d+',
                'pragma solidity ^0.8.0',
                code
            )
            # 移除 SafeMath (0.8+ 不需要)
            patched = re.sub(r'using SafeMath for uint256;', '', patched)
            patches.append(patched)
        
        return patches
    
    def _patch_access_control(self, code: str) -> List[str]:
        """生成访问控制漏洞修复版本"""
        patches = []
        
        # 添加 onlyOwner modifier
        if 'onlyOwner' not in code:
            patched = code
            
            # 添加 Ownable 导入
            if 'Ownable' not in code:
                patched = re.sub(
                    r'(pragma solidity[^;]+;)',
                    r'\1\nimport "@openzeppelin/contracts/access/Ownable.sol";',
                    patched,
                    count=1
                )
                patched = re.sub(
                    r'(contract\s+\w+)',
                    r'\1 is Ownable',
                    patched,
                    count=1
                )
            
            # 添加 onlyOwner 到敏感函数
            sensitive_patterns = [
                r'(function\s+(?:destroy|kill|selfdestruct|withdraw|transferOwnership)\s*\([^\)]*\)\s*(?:public|external))',
                r'(function\s+set\w+\s*\([^\)]*\)\s*(?:public|external))'
            ]
            
            for pattern in sensitive_patterns:
                patched = re.sub(
                    pattern,
                    r'\1 onlyOwner',
                    patched
                )
            
            patches.append(patched)
        
        return patches
    
    def _patch_tx_origin(self, code: str) -> List[str]:
        """修复 tx.origin 漏洞"""
        patches = []
        
        # 替换 tx.origin 为 msg.sender
        if 'tx.origin' in code:
            patched = code.replace('tx.origin', 'msg.sender')
            patches.append(patched)
        
        return patches
    
    def _patch_unchecked_call(self, code: str) -> List[str]:
        """修复未检查的外部调用"""
        patches = []
        
        # 添加 require 检查返回值
        patched = re.sub(
            r'(\w+)\.call\{value:\s*(\w+)\}\(\);',
            r'(bool success, ) = \1.call{value: \2}("");\nrequire(success, "Call failed");',
            code
        )
        
        if patched != code:
            patches.append(patched)
        
        return patches
    
    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """计算代码相似度"""
        return SequenceMatcher(None, code1, code2).ratio()
    
    def find_similar_pairs(self, contracts: List[Dict]) -> List[Dict]:
        """
        查找相似的合约对（可能是修复版本）
        """
        pairs = []
        
        # 按漏洞类型分组
        by_vuln = {}
        for contract in contracts:
            vuln = contract['vulnerability_type']
            if vuln not in by_vuln:
                by_vuln[vuln] = []
            by_vuln[vuln].append(contract)
        
        # 在同类型中查找相似对
        for vuln_type, group in by_vuln.items():
            print(f"🔍 Checking {len(group)} {vuln_type} contracts for similar pairs...")
            
            for i, c1 in enumerate(group):
                for c2 in group[i+1:]:
                    similarity = self._calculate_similarity(c1['code'], c2['code'])
                    
                    # 高度相似但不完全相同
                    if 0.75 < similarity < 0.98:
                        # 判断哪个是修复版本（启发式）
                        is_c1_patched = self._is_likely_patched(c1['code'])
                        is_c2_patched = self._is_likely_patched(c2['code'])
                        
                        if is_c1_patched != is_c2_patched:
                            before = c2 if is_c1_patched else c1
                            after = c1 if is_c1_patched else c2
                            
                            pair = {
                                'pair_id': f"sim_{len(pairs):04d}",
                                'code_before': before['code'],
                                'code_after': after['code'],
                                'vulnerability_type': vuln_type,
                                'similarity': similarity,
                                'before_source': before['name'],
                                'after_source': after['name'],
                                'method': 'similarity_matching',
                                'needs_manual_review': True
                            }
                            pairs.append(pair)
                            print(f"  ✅ Found pair: {before['name']} -> {after['name']} ({similarity:.2%})")
        
        return pairs
    
    def _is_likely_patched(self, code: str) -> bool:
        """启发式判断代码是否已修复"""
        patch_indicators = [
            'SafeMath',
            'ReentrancyGuard',
            'nonReentrant',
            'onlyOwner',
            'Ownable',
            'require(',
            'assert(',
            'AccessControl',
            '^0.8.'  # Solidity 0.8+ 有内置保护
        ]
        
        score = sum(1 for indicator in patch_indicators if indicator in code)
        return score >= 2
    
    def build_all(self):
        """构建所有 SPC 对"""
        print("\n" + "="*60)
        print("🏗️  Building SPC pairs from datasets")
        print("="*60 + "\n")
        
        # 1. 加载漏洞合约
        contracts = self.load_vulnerable_contracts()
        
        if not contracts:
            print("❌ No contracts found! Check your data directories:")
            print(f"   - SmartBugs: {self.smartbugs_dir}")
            print(f"   - SolidiFI: {self.solidifi_dir}")
            return
        
        # 2. 创建合成修复对
        print("\n🔧 Creating synthetic patches...")
        synthetic_pairs = self.create_synthetic_patches(contracts)
        print(f"✅ Created {len(synthetic_pairs)} synthetic pairs")
        
        # 3. 查找相似对
        print("\n🔍 Finding similar contract pairs...")
        similar_pairs = self.find_similar_pairs(contracts)
        print(f"✅ Found {len(similar_pairs)} similar pairs")
        
        # 4. 合并所有对
        all_pairs = synthetic_pairs + similar_pairs
        
        # 5. 保存
        output_file = self.output_dir / 'spc_pairs_from_datasets.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("📊 Summary")
        print("="*60)
        print(f"Total contracts analyzed: {len(contracts)}")
        print(f"Synthetic patches: {len(synthetic_pairs)}")
        print(f"Similar pairs found: {len(similar_pairs)}")
        print(f"Total SPC pairs: {len(all_pairs)}")
        print(f"\n✅ Saved to: {output_file}")
        
        # 统计漏洞类型分布
        vuln_dist = {}
        for pair in all_pairs:
            vtype = pair.get('vulnerability_type', 'unknown')
            vuln_dist[vtype] = vuln_dist.get(vtype, 0) + 1
        
        print("\n📈 Vulnerability Type Distribution:")
        for vtype, count in sorted(vuln_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"   {vtype}: {count}")
        
        return all_pairs


def main():
    builder = SPCBuilderFromDatasets()
    pairs = builder.build_all()
    
    if pairs:
        print("\n💡 Next steps:")
        print("   1. Review the generated pairs in data/spc_data/raw/")
        print("   2. Manually annotate to verify quality")
        print("   3. Use high-quality pairs for Bootstrap phase")


if __name__ == "__main__":
    main()