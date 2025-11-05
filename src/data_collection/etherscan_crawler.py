#!/usr/bin/env python3
"""
Etherscan V2 多链合约爬虫（支持 Key 池轮询 + 区间扫描验证合约）
一键后台：python etherscan_crawler.py --chain ethereum --scan --limit 10000 --batch 1000
"""

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml
from tqdm import tqdm

class EtherscanCrawler:
    def __init__(self, config_path: str = "configs/data_collection.yaml", chain: str = "ethereum"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.chain = chain.lower()
        self._chain_id = {
            "ethereum": "1", "bsc": "56", "polygon": "137", "avalanche": "43114"
        }
        if self.chain not in self._chain_id:
            raise ValueError(f"不支持的链: {chain}")

        # 读取链配置 & 切成 Key 池
        chain_config = self.config["scan_config"]["chains"][self.chain]
        self._key_pool = [k.strip() for k in chain_config["api_key"].split(",")]
        self._key_idx = 0
        self.base_url = chain_config["api_url"]
        self.rate_limit = self.config["scan_config"]["rate_limit"]
        self.retry_attempts = self.config["scan_config"]["retry_attempts"]
        self.retry_delay = self.config["scan_config"]["retry_delay"]

        # 输出按链分文件夹
        self.output_dir = Path(self.config["output"]["etherscan_raw"]) / self.chain
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        self.logger.info(f"Etherscan V2 爬虫初始化 | 链: {self.chain.upper()} | Keys: {len(self._key_pool)}")

    # ---------------- 日志 ----------------
    def _setup_logging(self):
        log_dir = Path(self.config["output"]["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"].upper()),
            format=self.config["logging"]["format"],
            handlers=[
                logging.FileHandler(log_dir / "etherscan_crawler.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    # ---------------- 限速 ----------------
    def _rate_limit(self):
        elapsed = time.time() - getattr(self, "last_request_time", 0)
        if elapsed < 1.0 / self.rate_limit:
            time.sleep(1.0 / self.rate_limit - elapsed)

    # ---------------- 核心请求 ----------------
    def _make_request(self, params: Dict) -> Optional[List[Dict]]:
        params["apikey"] = self._key_pool[self._key_idx]
        self._key_idx = (self._key_idx + 1) % len(self._key_pool)
        params["chainid"] = self._chain_id[self.chain]

        for attempt in range(self.retry_attempts):
            try:
                self._rate_limit()
                resp = requests.get(self.base_url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") == "1":
                    return data.get("result", [])
                elif data.get("message") == "No transactions found":
                    return []
                else:
                    self.logger.warning(f"API 错误: {data.get('message', 'Unknown')}")
                    return None
            except requests.RequestException as e:
                self.logger.warning(f"请求失败（{attempt + 1}/{self.retry_attempts}）: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"重试耗尽 | params: {params}")
                    return None
        return None

    # ---------------- 获取源码 ----------------
    def get_contract_source(self, address: str) -> Optional[Dict]:
        params = {"module": "contract", "action": "getsourcecode", "address": address}
        result = self._make_request(params)
        if result and result[0].get("SourceCode"):
            self.logger.info(f"源码长度: {len(result[0]['SourceCode'])} 字节 | {address}")
            return result[0]
        return None

    # ---------------- 地址来源 1：硬编码 ----------------
    def get_defi_contracts(self) -> List[str]:
        known = {
            "ethereum": [
                "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "0xE592427A0AEce92De3Edee1F18E0157C05861564",
                "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9", "0x5d3a536E4D6DbD6114cc1Ead35777bAB948E3643",
                "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2", "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7",
                "0xBA12222222228d8Ba445958a75a0704d566BF2C8", "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F",
            ],
            "bsc": [
                "0x10ED43C718714eb63d5aA57B78B54704E256024E", "0x00e65A10A1A7d8B98d0CE5085E8cDF04C1eF5261",
            ],
            "polygon": [
                "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff", "0x8dFf5E27EA6b7AC08ebFdf9eB790f79EE98aB2c8",
            ],
            "avalanche": [
                "0x60aE616a2155Ee3d9A68541Ba4544862310933d4", "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
            ],
        }
        addresses = known.get(self.chain, [])
        self.logger.info(f"加载 {len(addresses)} 个 {self.chain.upper()} 知名合约地址")
        return addresses

    # ---------------- 地址来源 2：区间扫描验证合约 ----------------
    def get_verified_contracts_by_range(self, start_block: int, end_block: int, max_addrs: int = 1000) -> List[str]:
        """
        通过 getLogs 扫描 ContractCreation 事件，再调 getsourcecode 验证是否有源码
        适合免费 Key 下大批量扩充地址池
        """
        self.logger.info(f"开始扫描区块 {start_block} -> {end_block}，最多取 {max_addrs} 个验证合约")
        addresses = []
        step = 2000   # 免费版一次最多 1000 条 log，保守 2k 区块
        for from_block in range(start_block, end_block, step):
            to_block = min(from_block + step - 1, end_block)
            # ContractCreation 事件 topic0（Create/Create2 都会触发）
            params = {
                "module": "logs",
                "action": "getLogs",
                "fromBlock": from_block,
                "toBlock": to_block,
                "topic0": "0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0",  # OwnershipTransferred（创建即 owner）
            }
            logs = self._make_request(params)
            if not logs:
                continue
            for log in logs:
                addr = log.get("address")
                if not addr:
                    continue
                # 调 getsourcecode 验证是否已验证
                if self.get_contract_source(addr):
                    addresses.append(addr)
                    if len(addresses) >= max_addrs:
                        self.logger.info(f"已收集够 {max_addrs} 个验证合约，提前结束")
                        return addresses
        self.logger.info(f"扫描完成，共得到 {len(addresses)} 个验证合约")
        return addresses

    # ---------------- 批量爬取 ----------------
    def crawl_contracts(self, addresses: List[str], save_batch_size: int = 100):
        self.logger.info(f"开始爬取 {len(addresses)} 份合约")
        contracts, fails = [], []
        for i, addr in enumerate(tqdm(addresses, desc="Crawling")):
            c = self.get_contract_source(addr)
            if c and c.get("SourceCode"):
                c["crawled_at"] = datetime.now().isoformat()
                c["address"] = addr
                c["code_hash"] = hashlib.md5(c["SourceCode"].encode()).hexdigest()
                contracts.append(c)
                if (i + 1) % save_batch_size == 0:
                    self._save_batch(contracts, i // save_batch_size)
                    contracts = []
            else:
                fails.append(addr)
        if contracts:
            self._save_batch(contracts, len(addresses) // save_batch_size)
        if fails:
            (self.output_dir / "failed_addresses.json").write_text(json.dumps(fails, indent=2))
        self.logger.info(f"爬取完成 | 成功: {len(addresses) - len(fails)} | 失败: {len(fails)}")

    def _save_batch(self, contracts: List[Dict], batch_num: int):
        batch_file = self.output_dir / f"batch_{batch_num:04d}.json"
        batch_file.write_text(json.dumps(contracts, indent=2, ensure_ascii=False))
        self.logger.info(f"已保存批次 {batch_num} | 共 {len(contracts)} 份合约")

    # ---------------- 过滤 & 统计 ----------------
    def filter_contracts(self, min_lines: int = 100, max_lines: int = 5000):
        self.logger.info("开始过滤合约（代码行数）")
        all_contracts = []
        for f in self.output_dir.glob("batch_*.json"):
            all_contracts.extend(json.loads(f.read_text()))
        filtered = [c for c in all_contracts if min_lines <= c["SourceCode"].count("\n") <= max_lines]
        out = Path(self.config["output"]["etherscan_processed"]) / self.chain / "filtered_contracts.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
        self.logger.info(f"过滤完成 | 保留 {len(filtered)} / {len(all_contracts)} 份合约")
        return filtered

    def generate_statistics(self):
        self.logger.info("生成统计报告")
        all_contracts = []
        for f in self.output_dir.glob("batch_*.json"):
            all_contracts.extend(json.loads(f.read_text()))
        stats = {
            "total_contracts": len(all_contracts),
            "compiler_versions": {},
            "contract_names": {},
            "avg_code_length": 0,
            "optimization_enabled": 0,
        }
        total_lines = 0
        for c in all_contracts:
            compiler = c.get("CompilerVersion", "Unknown")
            name = c.get("ContractName", "Unknown")
            lines = c["SourceCode"].count("\n")
            stats["compiler_versions"][compiler] = stats["compiler_versions"].get(compiler, 0) + 1
            stats["contract_names"][name] = stats["contract_names"].get(name, 0) + 1
            total_lines += lines
            if c.get("OptimizationUsed") == "1":
                stats["optimization_enabled"] += 1
        stats["avg_code_length"] = total_lines / len(all_contracts) if all_contracts else 0
        (self.output_dir / "statistics.json").write_text(json.dumps(stats, indent=2))
        self.logger.info(f"统计已保存 | 平均代码行数: {stats['avg_code_length']:.2f}")
        return stats


# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Etherscan V2 多链合约爬虫（Key 池轮询 + 区间扫描）")
    parser.add_argument("--chain", default="ethereum", choices=["ethereum", "bsc", "polygon", "avalanche"], help="目标链")
    parser.add_argument("--limit", type=int, help="限制地址数量（调试用）")
    parser.add_argument("--batch", type=int, default=1000, help="每批保存数量")
    parser.add_argument("--scan", action="store_true", help="用区间扫描代替硬编码")
    parser.add_argument("--start", type=int, default=16000000, help="起始区块")
    parser.add_argument("--end", type=int, default=16100000, help="结束区块")
    args = parser.parse_args()

    crawler = EtherscanCrawler(chain=args.chain)

    if args.scan:
        addresses = crawler.get_verified_contracts_by_range(
            start_block=args.start,
            end_block=args.end,
            max_addrs=args.limit or 1000
        )
    else:
        addresses = crawler.get_defi_contracts()
        if args.limit:
            addresses = addresses[:args.limit]

    print(f"\n🔗 链: {args.chain.upper()} | 待爬地址: {len(addresses)}\n")
    if not addresses:
        print("❌ 地址列表为空")
        exit(0)

    crawler.crawl_contracts(addresses, save_batch_size=args.batch)
    crawler.filter_contracts()
    stats = crawler.generate_statistics()
    print("\n=== 统计摘要 ===")
    print(f"总合约: {stats['total_contracts']}  |  平均行数: {stats['avg_code_length']:.2f}")
    print(f"启用优化: {stats['optimization_enabled']}")


if __name__ == "__main__":
    main()