"""
改进版 GitHub SPC 数据收集器
增加了详细日志和错误处理
"""

import requests
import time
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml
from tqdm import tqdm

class ImprovedGitHubSPCCrawler:
    def __init__(self, config_path: str = "configs/data_collection.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.token = self.config['github']['token']
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        self.output_dir = Path(self.config['output']['spc_raw'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        self.logger.info("Improved GitHub SPC Crawler initialized")
        
        # 统计信息
        self.stats = {
            'commits_searched': 0,
            'commits_processed': 0,
            'security_fixes_found': 0,
            'sol_files_found': 0,
            'patches_extracted': 0,
            'pairs_created': 0,
            'errors': []
        }
    
    def _setup_logging(self):
        log_dir = Path(self.config['output']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'spc_crawler_improved.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def search_commits(self, keyword: str, max_results: int = 50) -> List[Dict]:
        """搜索 commits"""
        self.logger.info(f"Searching for: {keyword}")
        
        query = f'{keyword} language:Solidity'
        url = "https://api.github.com/search/commits"
        
        headers = self.headers.copy()
        headers['Accept'] = 'application/vnd.github.cloak-preview'
        
        params = {
            'q': query,
            'per_page': min(max_results, 30),
            'sort': 'committer-date',
            'order': 'desc'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                self.stats['commits_searched'] += len(items)
                self.logger.info(f"Found {len(items)} commits")
                return items
            else:
                self.logger.warning(f"Search failed: {response.status_code} - {response.text[:100]}")
                return []
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            self.stats['errors'].append(f"Search error: {e}")
            return []
    
    def get_commit_details(self, repo: str, sha: str) -> Optional[Dict]:
        """获取 commit 详情"""
        url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to get commit {sha[:8]}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting commit {sha[:8]}: {e}")
            return None
    
    def is_security_fix(self, commit_message: str) -> bool:
        """判断是否是安全修复（放宽条件）"""
        message_lower = commit_message.lower()
        
        # 安全相关关键词
        security_keywords = [
            'vulnerability', 'vuln', 'security', 'exploit', 'attack',
            'reentrancy', 'overflow', 'underflow', 'access control',
            'fix', 'patch', 'bug', 'issue',
            'CVE', 'critical', 'malicious', 'unsafe'
        ]
        
        # 只要包含任一关键词就认为可能是安全相关
        return any(keyword in message_lower for keyword in security_keywords)
    
    def extract_sol_changes(self, commit_data: Dict) -> List[Dict]:
        """提取 Solidity 文件变更（增强版）"""
        changes = []
        files = commit_data.get('files', [])
        
        self.logger.debug(f"Commit has {len(files)} files")
        
        for file in files:
            filename = file.get('filename', '')
            
            if not filename.endswith('.sol'):
                continue
            
            self.stats['sol_files_found'] += 1
            self.logger.debug(f"Found Solidity file: {filename}")
            
            patch = file.get('patch')
            
            if not patch:
                self.logger.warning(f"No patch for {filename} (file too large or binary)")
                # 对于没有 patch 的文件，尝试获取完整内容
                # （这里暂时跳过，因为需要额外 API 调用）
                continue
            
            # 解析 patch
            before_code, after_code = self._parse_patch(patch)
            
            if before_code and after_code and len(before_code) > 20 and len(after_code) > 20:
                change = {
                    'filename': filename,
                    'before': before_code,
                    'after': after_code,
                    'additions': file.get('additions', 0),
                    'deletions': file.get('deletions', 0)
                }
                changes.append(change)
                self.stats['patches_extracted'] += 1
                self.logger.debug(f"Extracted patch: {len(before_code)} -> {len(after_code)} chars")
            else:
                self.logger.debug(f"Patch too small or invalid for {filename}")
        
        return changes
    
    def _parse_patch(self, patch: str) -> Tuple[str, str]:
        """解析 patch"""
        before_lines = []
        after_lines = []
        
        for line in patch.split('\n'):
            if line.startswith('@@'):
                continue
            elif line.startswith('-') and not line.startswith('---'):
                before_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                after_lines.append(line[1:])
            elif line.startswith(' '):
                # 上下文行，两边都加
                before_lines.append(line[1:])
                after_lines.append(line[1:])
        
        return '\n'.join(before_lines), '\n'.join(after_lines)
    
    def collect_from_search(self, max_pairs: int = 500):
        """从搜索收集数据"""
        self.logger.info(f"Starting collection (target: {max_pairs} pairs)")
        
        keywords = [
            'vulnerability solidity',
            'security fix solidity',
            'reentrancy attack',
            'overflow solidity',
            'smart contract bug'
        ]
        
        spc_pairs = []
        processed_commits = set()
        
        for keyword in keywords:
            if len(spc_pairs) >= max_pairs:
                break
            
            commits = self.search_commits(keyword, max_results=30)
            self.logger.info(f"Processing {len(commits)} commits for '{keyword}'")
            
            for commit in tqdm(commits, desc=f"{keyword}"):
                if len(spc_pairs) >= max_pairs:
                    break
                
                sha = commit.get('sha')
                repo = commit.get('repository', {}).get('full_name')
                
                if not sha or not repo or sha in processed_commits:
                    continue
                
                processed_commits.add(sha)
                self.stats['commits_processed'] += 1
                
                # 获取详情
                commit_details = self.get_commit_details(repo, sha)
                if not commit_details:
                    continue
                
                commit_message = commit_details.get('commit', {}).get('message', '')
                
                # 检查是否是安全修复
                if not self.is_security_fix(commit_message):
                    self.logger.debug(f"Skipping non-security commit: {commit_message[:50]}")
                    continue
                
                self.stats['security_fixes_found'] += 1
                self.logger.info(f"Security fix found: {commit_message[:60]}...")
                
                # 提取代码变更
                changes = self.extract_sol_changes(commit_details)
                
                if not changes:
                    self.logger.debug("No Solidity changes extracted")
                    continue
                
                # 创建 SPC 对
                for change in changes:
                    pair = {
                        'pair_id': f"spc_{len(spc_pairs):04d}",
                        'repo': repo,
                        'commit_sha': sha,
                        'commit_message': commit_message,
                        'commit_date': commit_details.get('commit', {}).get('committer', {}).get('date'),
                        'filename': change['filename'],
                        'code_before': change['before'],
                        'code_after': change['after'],
                        'label_before': 'vulnerable',
                        'label_after': 'patched',
                        'vulnerability_type': self._infer_vuln_type(commit_message),
                        'needs_manual_review': True
                    }
                    spc_pairs.append(pair)
                    self.stats['pairs_created'] += 1
                    self.logger.info(f"Created pair {len(spc_pairs)}: {change['filename']}")
                
                time.sleep(0.5)  # 避免速率限制
        
        return spc_pairs
    
    def collect_from_repos(self):
        """从特定仓库收集"""
        self.logger.info("Collecting from target repositories...")
        
        target_repos = [
            'OpenZeppelin/openzeppelin-contracts',
            'ConsenSys/mythril',
            'crytic/not-so-smart-contracts'
        ]
        
        all_pairs = []
        
        for repo in target_repos:
            self.logger.info(f"Processing repo: {repo}")
            
            url = f"https://api.github.com/repos/{repo}/commits"
            params = {'per_page': 50}
            
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                if response.status_code != 200:
                    continue
                
                commits = response.json()
                
                for commit in commits[:30]:  # 限制每个仓库处理的数量
                    sha = commit['sha']
                    commit_details = self.get_commit_details(repo, sha)
                    
                    if not commit_details:
                        continue
                    
                    message = commit_details.get('commit', {}).get('message', '')
                    
                    if self.is_security_fix(message):
                        changes = self.extract_sol_changes(commit_details)
                        
                        for change in changes:
                            pair = {
                                'pair_id': f"repo_{repo.replace('/', '_')}_{sha[:8]}",
                                'repo': repo,
                                'commit_sha': sha,
                                'commit_message': message,
                                'code_before': change['before'],
                                'code_after': change['after'],
                                'vulnerability_type': self._infer_vuln_type(message)
                            }
                            all_pairs.append(pair)
                
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing repo {repo}: {e}")
        
        return all_pairs
    
    def _infer_vuln_type(self, message: str) -> str:
        """推断漏洞类型"""
        message_lower = message.lower()
        
        vuln_map = {
            'reentrancy': ['reentrancy', 'reentrant', 're-entrance'],
            'overflow': ['overflow'],
            'underflow': ['underflow'],
            'access_control': ['access control', 'permission', 'unauthorized'],
            'tx_origin': ['tx.origin'],
            'timestamp': ['timestamp', 'block.timestamp'],
            'unchecked_call': ['unchecked', 'call return'],
        }
        
        for vuln_type, keywords in vuln_map.items():
            if any(kw in message_lower for kw in keywords):
                return vuln_type
        
        return 'unknown'
    
    def save_pairs(self, pairs: List[Dict], filename: str):
        """保存数据"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(pairs)} pairs to {output_path}")
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 收集统计")
        print("="*60)
        print(f"搜索到的 commits: {self.stats['commits_searched']}")
        print(f"处理的 commits: {self.stats['commits_processed']}")
        print(f"安全修复: {self.stats['security_fixes_found']}")
        print(f"Solidity 文件: {self.stats['sol_files_found']}")
        print(f"提取的 patches: {self.stats['patches_extracted']}")
        print(f"创建的 SPC 对: {self.stats['pairs_created']}")
        if self.stats['errors']:
            print(f"\n错误数量: {len(self.stats['errors'])}")
            for err in self.stats['errors'][:5]:
                print(f"  - {err}")


def main():
    crawler = ImprovedGitHubSPCCrawler()
    
    print("=== Improved GitHub SPC Collection ===\n")
    
    # 方法1: 关键词搜索
    print("1. Collecting from keyword search...")
    search_pairs = crawler.collect_from_search(max_pairs=500)
    crawler.save_pairs(search_pairs, 'spc_pairs_search.json')
    
    # 方法2: 目标仓库
    print("\n2. Collecting from target repositories...")
    repo_pairs = crawler.collect_from_repos()
    crawler.save_pairs(repo_pairs, 'spc_pairs_repos.json')
    
    # 合并
    all_pairs = search_pairs + repo_pairs
    crawler.save_pairs(all_pairs, 'spc_pairs_all.json')
    
    # 统计
    crawler.print_stats()
    
    print(f"\n✅ Total pairs collected: {len(all_pairs)}")
    print(f"📁 Output directory: {crawler.output_dir}")


if __name__ == "__main__":
    main()