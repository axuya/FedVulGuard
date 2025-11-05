#!/usr/bin/env python3
"""
数据收集主脚本
按步骤执行 Etherscan 和 GitHub 数据收集
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.etherscan_crawler import EtherscanCrawler
from src.data_collection.github_spc_crawler import GitHubSPCCrawler
from src.utils.data_utils import (
    extract_contract_addresses_from_datasets,
    save_addresses_list,
    merge_spc_data,
    get_vulnerability_distribution
)
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_extract_addresses():
    """步骤1: 从现有数据集提取地址"""
    print("\n" + "="*60)
    print("STEP 1: Extracting contract addresses from datasets")
    print("="*60)
    
    addresses = extract_contract_addresses_from_datasets()
    
    if addresses:
        save_addresses_list(addresses, "data/contract_addresses.txt")
        print(f"✅ Extracted {len(addresses)} addresses")
    else:
        print("⚠️  No addresses found in datasets")
        print("💡 You can manually add addresses to data/contract_addresses.txt")
    
    return addresses


def step2_crawl_etherscan(use_known_addresses: bool = True):
    """步骤2: 爬取 Etherscan 合约"""
    print("\n" + "="*60)
    print("STEP 2: Crawling Etherscan contracts")
    print("="*60)
    
    crawler = EtherscanCrawler()
    
    # 决定使用哪些地址
    if use_known_addresses:
        # 方案A: 使用已知 DeFi 地址
        print("Using known DeFi contract addresses...")
        addresses = crawler.get_defi_contracts()
    else:
        # 方案B: 使用从数据集提取的地址
        addresses_file = Path("data/contract_addresses.txt")
        if addresses_file.exists():
            print(f"Loading addresses from {addresses_file}...")
            with open(addresses_file, 'r') as f:
                addresses = [line.strip() for line in f if line.strip()]
        else:
            print("❌ No address file found. Run step 1 first or use known addresses.")
            return
    
    print(f"Total addresses to crawl: {len(addresses)}")
    
    if not addresses:
        print("❌ No addresses to crawl!")
        return
    
    # 开始爬取
    crawler.crawl_contracts(addresses, save_batch_size=50)
    
    # 过滤
    print("\nFiltering contracts...")
    filtered = crawler.filter_contracts(
        min_size=100,
        max_size=5000
    )
    
    # 统计
    print("\nGenerating statistics...")
    stats = crawler.generate_statistics()
    
    print("\n" + "-"*60)
    print("📊 Etherscan Crawling Statistics:")
    print(f"  Total contracts: {stats['total_contracts']}")
    print(f"  Average code length: {stats['avg_code_length']:.2f} lines")
    print(f"  Optimization enabled: {stats['optimization_enabled']}")
    print("-"*60)
    
    return filtered


def step3_collect_spc_data(target_pairs: int = 500):
    """步骤3: 收集 SPC 数据"""
    print("\n" + "="*60)
    print("STEP 3: Collecting SPC pairs from GitHub")
    print("="*60)
    
    crawler = GitHubSPCCrawler()
    
    # 方法1: 关键词搜索
    print("\n3.1 Collecting from keyword search...")
    keyword_pairs = crawler.collect_spc_pairs(max_pairs=target_pairs)
    
    # 方法2: 目标仓库
    print("\n3.2 Collecting from target repositories...")
    repo_pairs = crawler.collect_from_target_repos()
    
    # 合并
    all_pairs = keyword_pairs + repo_pairs
    print(f"\n✅ Total SPC pairs collected: {len(all_pairs)}")
    
    # 生成标注模板
    print("\n3.3 Generating annotation template...")
    crawler.generate_annotation_template(all_pairs)
    
    # 统计漏洞分布
    distribution = get_vulnerability_distribution(all_pairs)
    print("\n📊 Vulnerability Distribution:")
    for vuln_type, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vuln_type}: {count}")
    
    return all_pairs


def step4_merge_and_validate():
    """步骤4: 合并和验证数据"""
    print("\n" + "="*60)
    print("STEP 4: Merging and validating data")
    print("="*60)
    
    # 合并 SPC 数据
    print("Merging SPC data...")
    spc_pairs = merge_spc_data()
    
    print(f"\n✅ Data collection pipeline completed!")
    print(f"📁 Check the following directories:")
    print(f"  - Etherscan data: data/etherscan/")
    print(f"  - SPC data: data/spc_data/")
    print(f"  - Logs: logs/")
    
    return spc_pairs


def main():
    parser = argparse.ArgumentParser(description='FedVulGuard Data Collection Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], 
                       help='Run specific step (1-4), or run all if not specified')
    parser.add_argument('--etherscan-mode', choices=['known', 'extracted'], default='known',
                       help='Etherscan address source: known DeFi addresses or extracted from datasets')
    parser.add_argument('--spc-pairs', type=int, default=500,
                       help='Target number of SPC pairs to collect')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         FedVulGuard Data Collection Pipeline            ║
    ║                                                          ║
    ║  This script will collect:                              ║
    ║    1. Contract addresses from existing datasets         ║
    ║    2. Smart contract source code from Etherscan         ║
    ║    3. SPC (Similar Patched Code) pairs from GitHub      ║
    ║    4. Merge and validate collected data                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if args.step:
        # 运行特定步骤
        if args.step == 1:
            step1_extract_addresses()
        elif args.step == 2:
            step2_crawl_etherscan(use_known_addresses=(args.etherscan_mode == 'known'))
        elif args.step == 3:
            step3_collect_spc_data(target_pairs=args.spc_pairs)
        elif args.step == 4:
            step4_merge_and_validate()
    else:
        # 运行完整流程
        try:
            addresses = step1_extract_addresses()
            
            input("\n⏸️  Press Enter to continue to Step 2 (Etherscan crawling)...")
            filtered_contracts = step2_crawl_etherscan(
                use_known_addresses=(args.etherscan_mode == 'known')
            )
            
            input("\n⏸️  Press Enter to continue to Step 3 (SPC collection)...")
            spc_pairs = step3_collect_spc_data(target_pairs=args.spc_pairs)
            
            input("\n⏸️  Press Enter to continue to Step 4 (Merge and validate)...")
            step4_merge_and_validate()
            
            print("\n" + "="*60)
            print("🎉 DATA COLLECTION COMPLETED!")
            print("="*60)
            print("\n📋 Next Steps:")
            print("  1. Review the annotation template in data/spc_data/annotated/")
            print("  2. Manually annotate the SPC pairs")
            print("  3. Run data preprocessing (Phase 2)")
            print("  4. Build multi-graph representations (Phase 3)")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error during data collection: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()