import requests
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yaml
from tqdm import tqdm
import hashlib

class EtherscanCrawler:
    def __init__(self, config_path: str = "configs/data_collection.yaml"):
        """初始化爬虫"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        eth_cfg = self.config['scan_config']['chains']['ethereum']
        self.api_key = eth_cfg['api_key']
        self.base_url = eth_cfg['api_url']
        self.rate_limit = self.config['scan_config']['rate_limit']
        self.retry_attempts = self.config['scan_config']['retry_attempts']

        #self.api_key = self.config['etherscan']['api_key']
        #self.base_url = self.config['etherscan']['base_url']
        #self.base_url = self.config['etherscan']['api_url']
        #self.rate_limit = self.config['etherscan']['rate_limit']
        #self.retry_attempts = self.config['etherscan']['retry_attempts']
        
        # 创建输出目录
        self.output_dir = Path(self.config['output']['etherscan_raw'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 请求计数器（用于限速）
        self.request_count = 0
        self.last_request_time = time.time()
        
        self.logger.info("Etherscan Crawler initialized")
    
    def _setup_logging(self):
        """配置日志"""
        log_dir = Path(self.config['output']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'etherscan_crawler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """限制请求速率"""
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.0 / self.rate_limit:
            time.sleep(1.0 / self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """发送 API 请求（带重试机制）"""
        self.logger.info(f"[debug] 请求 URL: {self.base_url}")
        self.logger.info(f"[debug] 请求参数: {params}")

        params['apikey'] = self.api_key
        params['chainid'] = '1'   #eth主网
        
        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit()
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if data['status'] == '1':
                    return data['result']
                elif data['message'] == 'No transactions found':
                    return []
                else:
                    self.logger.warning(f"API returned error: {data['message']}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.config['scan_config']['retry_delay'])
                else:
                    self.logger.error(f"Max retries reached for params: {params}")
                    return None
        
        return None
    
    def get_contract_source(self, address: str) -> Optional[Dict]:
        params = {
            'module': 'contract',
            'action': 'getsourcecode',
            'address': address
        }

        result = self._make_request(params)
        if result:
            self.logger.info(f"SourceCode 长度: {len(result[0].get('SourceCode', ''))}")
            if result[0].get('SourceCode'):
                return result[0]
        return None
    
    def get_verified_contracts_list(self, page: int = 1, offset: int = 100) -> List[str]:
        """
        获取验证合约列表
        注意：Etherscan 免费 API 不直接提供此接口，需要通过其他方式获取
        这里提供几种替代方案
        """
        # 方案1: 使用合约创建事件
        params = {
            'module': 'logs',
            'action': 'getLogs',
            'fromBlock': 0,
            'toBlock': 'latest',
            'page': page,
            'offset': offset
        }
        
        # 注意：此方法可能需要付费 API
        self.logger.warning("Free API has limitations. Consider using contract addresses from known sources.")
        return []
    
    def get_contracts_from_known_sources(self) -> List[str]:
        """
        从已知来源获取合约地址
        推荐方法：使用 SmartBugs 和 SolidiFI 数据集中的地址
        """
        addresses = []
        
        # 从 SmartBugs 提取地址
        smartbugs_dir = Path("data/smartbugs")
        if smartbugs_dir.exists():
            # 这里假设 SmartBugs 数据中包含地址信息
            # 需要根据实际数据格式调整
            pass
        
        # 从 SolidiFI 提取地址
        solidifi_dir = Path("data/solidifi")
        if solidifi_dir.exists():
            pass
        
        return addresses
    
    def get_defi_contracts(self) -> List[str]:
        """
        获取知名 DeFi 项目的合约地址
        """
        # 知名 DeFi 合约地址（示例）
        known_contracts = {
            'Uniswap V2 Router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
            'Uniswap V3 Router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'Aave V2 LendingPool': '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9',
            'Compound cDAI': '0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643',
            'MakerDAO': '0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2',
            'Curve Finance': '0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7',
            'Balancer Vault': '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
            'SushiSwap Router': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
        }
        
        return list(known_contracts.values())
    
    def crawl_contracts(self, addresses: List[str], save_batch_size: int = 100):
        """
        批量爬取合约源码
        """
        self.logger.info(f"Starting to crawl {len(addresses)} contracts")
        
        contracts_data = []
        failed_addresses = []
        
        for i, address in enumerate(tqdm(addresses, desc="Crawling contracts")):
            contract_data = self.get_contract_source(address)
            
            if contract_data and contract_data.get('SourceCode'):
                # 添加额外元数据
                contract_data['crawled_at'] = datetime.now().isoformat()
                contract_data['address'] = address
                
                # 计算代码哈希（用于去重）
                code_hash = hashlib.md5(
                    contract_data['SourceCode'].encode()
                ).hexdigest()
                contract_data['code_hash'] = code_hash
                
                contracts_data.append(contract_data)
                self.logger.info(f"Successfully crawled contract: {address}")
                
                # 定期保存
                if (i + 1) % save_batch_size == 0:
                    self._save_batch(contracts_data, batch_num=i // save_batch_size)
                    contracts_data = []
            else:
                failed_addresses.append(address)
                self.logger.warning(f"Failed to crawl contract: {address}")
        
        # 保存剩余数据
        if contracts_data:
            self._save_batch(contracts_data, batch_num=len(addresses) // save_batch_size)
        
        # 保存失败列表
        if failed_addresses:
            failed_path = self.output_dir / 'failed_addresses.json'
            with open(failed_path, 'w') as f:
                json.dump(failed_addresses, f, indent=2)
        
        self.logger.info(f"Crawling completed. Success: {len(addresses) - len(failed_addresses)}, Failed: {len(failed_addresses)}")
    
    def _save_batch(self, contracts: List[Dict], batch_num: int):
        """保存批次数据"""
        batch_file = self.output_dir / f'batch_{batch_num:04d}.json'
        with open(batch_file, 'w') as f:
            json.dump(contracts, f, indent=2)
        self.logger.info(f"Saved batch {batch_num} with {len(contracts)} contracts")
    
    def filter_contracts(self, min_size: int = 100, max_size: int = 5000):
        """
        过滤合约（根据代码大小）
        """
        self.logger.info("Filtering contracts...")
        
        all_contracts = []
        for batch_file in self.output_dir.glob('batch_*.json'):
            with open(batch_file, 'r') as f:
                contracts = json.load(f)
                all_contracts.extend(contracts)
        
        filtered = []
        for contract in all_contracts:
            source = contract.get('SourceCode', '')
            lines = source.count('\n')
            
            if min_size <= lines <= max_size:
                filtered.append(contract)
        
        self.logger.info(f"Filtered {len(filtered)} contracts from {len(all_contracts)}")
        
        # 保存过滤后的数据
        output_path = Path(self.config['output']['etherscan_processed']) / 'filtered_contracts.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(filtered, f, indent=2)
        
        return filtered

    def generate_statistics(self):
        """生成数据统计报告"""
        all_contracts = []
        for batch_file in self.output_dir.glob('batch_*.json'):
            with open(batch_file, 'r') as f:
                contracts = json.load(f)
                all_contracts.extend(contracts)
        
        stats = {
            'total_contracts': len(all_contracts),
            'compiler_versions': {},
            'contract_names': {},
            'avg_code_length': 0,
            'optimization_enabled': 0
        }
        
        total_lines = 0
        for contract in all_contracts:
            # 编译器版本统计
            compiler = contract.get('CompilerVersion', 'Unknown')
            stats['compiler_versions'][compiler] = stats['compiler_versions'].get(compiler, 0) + 1
            
            # 合约名称统计
            name = contract.get('ContractName', 'Unknown')
            stats['contract_names'][name] = stats['contract_names'].get(name, 0) + 1
            
            # 代码长度
            lines = contract.get('SourceCode', '').count('\n')
            total_lines += lines
            
            # 优化设置
            if contract.get('OptimizationUsed') == '1':
                stats['optimization_enabled'] += 1
        
        stats['avg_code_length'] = total_lines / len(all_contracts) if all_contracts else 0
        
        # 保存统计
        stats_path = self.output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Statistics saved to {stats_path}")
        return stats


def main():
    """主函数"""
    crawler = EtherscanCrawler()
    
    # 方案1: 使用已知的 DeFi 合约地址
    print("Fetching known DeFi contract addresses...")
    addresses = crawler.get_defi_contracts()
    
    # 方案2: 从现有数据集提取地址
    # addresses = crawler.get_contracts_from_known_sources()
    
    # 方案3: 手动提供地址列表
    # with open('contract_addresses.txt', 'r') as f:
    #     addresses = [line.strip() for line in f if line.strip()]
    
    print(f"Total addresses to crawl: {len(addresses)}")
    
    # 开始爬取
    crawler.crawl_contracts(addresses)
    
    # 过滤合约
    crawler.filter_contracts()
    
    # 生成统计
    stats = crawler.generate_statistics()
    print("\n=== Statistics ===")
    print(f"Total contracts: {stats['total_contracts']}")
    print(f"Average code length: {stats['avg_code_length']:.2f} lines")
    print(f"Optimization enabled: {stats['optimization_enabled']}")


if __name__ == "__main__":
    main()