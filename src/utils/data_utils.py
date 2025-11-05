"""
数据处理工具函数
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)


def deduplicate_contracts(contracts: List[Dict]) -> List[Dict]:
    """
    去重合约（基于代码哈希）
    """
    seen_hashes = set()
    unique_contracts = []
    
    for contract in contracts:
        code_hash = contract.get('code_hash')
        if not code_hash:
            # 如果没有哈希，计算一个
            code = contract.get('SourceCode', '')
            code_hash = hashlib.md5(code.encode()).hexdigest()
            contract['code_hash'] = code_hash
        
        if code_hash not in seen_hashes:
            seen_hashes.add(code_hash)
            unique_contracts.append(contract)
    
    logger.info(f"Deduplication: {len(contracts)} -> {len(unique_contracts)}")
    return unique_contracts


def extract_contract_addresses_from_datasets():
    """
    从 SmartBugs 和 SolidiFI 数据集中提取合约地址
    """
    addresses = set()
    
    # 从 SmartBugs 提取
    smartbugs_dir = Path("data/smartbugs")
    if smartbugs_dir.exists():
        for sol_file in smartbugs_dir.rglob("*.sol"):
            # SmartBugs 的文件名可能包含地址信息
            # 例如: 0x1234...5678.sol
            filename = sol_file.stem
            if filename.startswith("0x") and len(filename) == 42:
                addresses.add(filename)
        
        # 如果有 JSON 元数据文件
        for json_file in smartbugs_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'address' in data:
                        addresses.add(data['address'])
            except:
                pass
    
    # 从 SolidiFI 提取
    solidifi_dir = Path("data/solidifi")
    if solidifi_dir.exists():
        # SolidiFI 可能有不同的结构
        for sol_file in solidifi_dir.rglob("*.sol"):
            # 尝试从文件内容提取地址
            try:
                with open(sol_file, 'r') as f:
                    content = f.read()
                    # 查找注释中的地址
                    import re
                    found_addresses = re.findall(r'0x[a-fA-F0-9]{40}', content)
                    addresses.update(found_addresses)
            except:
                pass
    
    logger.info(f"Extracted {len(addresses)} addresses from datasets")
    return list(addresses)


def save_addresses_list(addresses: List[str], output_path: str):
    """
    保存地址列表到文件
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for addr in addresses:
            f.write(f"{addr}\n")
    
    logger.info(f"Saved {len(addresses)} addresses to {output_path}")


def load_addresses_list(input_path: str) -> List[str]:
    """
    从文件加载地址列表
    """
    with open(input_path, 'r') as f:
        addresses = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(addresses)} addresses from {input_path}")
    return addresses


def merge_spc_data():
    """
    合并多个来源的 SPC 数据
    """
    spc_dir = Path("data/spc_data/raw")
    all_pairs = []
    
    for json_file in spc_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            pairs = json.load(f)
            all_pairs.extend(pairs)
    
    # 去重（基于 pair_id 或代码哈希）
    unique_pairs = []
    seen_ids = set()
    
    for pair in all_pairs:
        pair_id = pair.get('pair_id')
        if pair_id not in seen_ids:
            seen_ids.add(pair_id)
            unique_pairs.append(pair)
    
    output_path = spc_dir / 'merged_spc_pairs.json'
    with open(output_path, 'w') as f:
        json.dump(unique_pairs, f, indent=2)
    
    logger.info(f"Merged {len(unique_pairs)} unique SPC pairs")
    return unique_pairs


def validate_contract_code(code: str) -> bool:
    """
    验证合约代码是否有效
    """
    if not code or len(code.strip()) < 50:
        return False
    
    # 检查是否包含 Solidity 关键字
    required_keywords = ['pragma', 'solidity', 'contract']
    code_lower = code.lower()
    
    return any(keyword in code_lower for keyword in required_keywords)


def get_vulnerability_distribution(spc_pairs: List[Dict]) -> Dict[str, int]:
    """
    统计漏洞类型分布
    """
    distribution = {}
    
    for pair in spc_pairs:
        vuln_type = pair.get('vulnerability_type', 'unknown')
        distribution[vuln_type] = distribution.get(vuln_type, 0) + 1
    
    return distribution


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 提取地址
    addresses = extract_contract_addresses_from_datasets()
    if addresses:
        save_addresses_list(addresses, "data/contract_addresses.txt")
    
    # 合并 SPC 数据
    spc_pairs = merge_spc_data()
    
    # 统计漏洞分布
    distribution = get_vulnerability_distribution(spc_pairs)
    print("\nVulnerability Distribution:")
    for vuln_type, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vuln_type}: {count}")