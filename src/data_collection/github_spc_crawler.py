"""
GitHub SPC (Similar Patched Code) 数据收集器
用于收集智能合约的漏洞修复历史，构建 SPC 样本对
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
import base64

class GitHubSPCCrawler:
    def __init__(self, config_path: str = "configs/data_collection.yaml"):
        """初始化 GitHub 爬虫"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.token = self.config['github']['token']
        self.rate_limit = self.config['github']['rate_limit']
        self.keywords = self.config['github']['search_keywords']
        self.target_repos = self.config['github']['target_repos']
        self.commit_patterns = self.config['github']['commit_patterns']
        
        # 设置请求头
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # 创建输出目录
        self.output_dir = Path(self.config['output']['spc_raw'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 请求计数
        self.api_calls = 0
        self.last_request_time = time.time()
        
        self.logger.info("GitHub SPC Crawler initialized")
    
    def _setup_logging(self):
        """配置日志"""
        log_dir = Path(self.config['output']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_dir / 'github_spc_crawler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit(self):
        """限制请求速率"""
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送 GitHub API 请求"""
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            self.api_calls += 1
            
            # 检查速率限制
            remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
            if remaining < 10:
                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                wait_time = max(reset_time - time.time(), 0)
                self.logger.warning(f"Approaching rate limit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
    
    def search_commits(self, keyword: str, repo: Optional[str] = None, max_results: int = 100) -> List[Dict]:
        """
        搜索包含特定关键词的 commits
        """
        self.logger.info(f"Searching commits with keyword: {keyword}")
        
        # 构建搜索查询
        query = f'{keyword} language:Solidity'
        if repo:
            query += f' repo:{repo}'
        
        commits = []
        page = 1
        per_page = 30  # GitHub API 限制
        
        while len(commits) < max_results:
            url = "https://api.github.com/search/commits"
            params = {
                'q': query,
                'sort': 'committer-date',
                'order': 'desc',
                'per_page': per_page,
                'page': page
            }
            
            # 需要特殊的 Accept header
            headers = self.headers.copy()
            headers['Accept'] = 'application/vnd.github.cloak-preview'
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                self.logger.warning(f"Search failed: {response.status_code}")
                break
            
            data = response.json()
            items = data.get('items', [])
            
            if not items:
                break
            
            commits.extend(items)
            page += 1
            
            if len(items) < per_page:
                break
            
            time.sleep(1)  # 避免触发速率限制
        
        self.logger.info(f"Found {len(commits)} commits for keyword: {keyword}")
        return commits[:max_results]
    
    def get_commit_details(self, repo_full_name: str, sha: str) -> Optional[Dict]:
        """获取 commit 详细信息"""
        url = f"https://api.github.com/repos/{repo_full_name}/commits/{sha}"
        return self._make_request(url)
    
    def extract_solidity_changes(self, commit_data: Dict) -> List[Dict]:
        """
        从 commit 中提取 Solidity 文件的变更
        """
        changes = []
        
        files = commit_data.get('files', [])
        for file in files:
            filename = file.get('filename', '')
            
            # 只处理 .sol 文件
            if not filename.endswith('.sol'):
                continue
            
            patch = file.get('patch', '')
            if not patch:
                continue
            
            # 解析 patch
            before_code, after_code = self._parse_patch(patch)
            
            if before_code and after_code:
                change = {
                    'filename': filename,
                    'before': before_code,
                    'after': after_code,
                    'status': file.get('status', 'modified'),
                    'additions': file.get('additions', 0),
                    'deletions': file.get('deletions', 0),
                    'changes': file.get('changes', 0)
                }
                changes.append(change)
        
        return changes
    
    def _parse_patch(self, patch: str) -> Tuple[str, str]:
        """
        解析 Git patch，提取修改前后的代码
        """
        before_lines = []
        after_lines = []
        
        for line in patch.split('\n'):
            if line.startswith('@@'):
                continue
            elif line.startswith('-') and not line.startswith('---'):
                # 删除的行（修改前）
                before_lines.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'):
                # 添加的行（修改后）
                after_lines.append(line[1:])
            elif line.startswith(' '):
                # 未修改的行（上下文）
                before_lines.append(line[1:])
                after_lines.append(line[1:])
        
        return '\n'.join(before_lines), '\n'.join(after_lines)
    
    def is_security_fix(self, commit_data: Dict) -> bool:
        """
        判断 commit 是否是安全修复
        """
        commit_message = commit_data.get('commit', {}).get('message', '').lower()
        
        # 检查 commit 消息是否匹配安全修复模式
        for pattern in self.commit_patterns:
            if re.search(pattern, commit_message):
                return True
        
        # 检查是否包含安全相关关键词
        security_keywords = [
            'vulnerability', 'security', 'exploit', 'attack',
            'reentrancy', 'overflow', 'underflow', 'access control',
            'CVE', 'critical', 'malicious'
        ]
        
        return any(keyword in commit_message for keyword in security_keywords)
    
    def collect_spc_pairs(self, max_pairs: int = 500):
        """
        收集 SPC 样本对
        """
        self.logger.info(f"Starting SPC pair collection (target: {max_pairs} pairs)")
        
        spc_pairs = []
        processed_commits = set()
        
        # 遍历关键词
        for keyword in self.keywords:
            if len(spc_pairs) >= max_pairs:
                break
            
            # 搜索 commits
            commits = self.search_commits(keyword, max_results=50)
            
            for commit in tqdm(commits, desc=f"Processing {keyword}"):
                if len(spc_pairs) >= max_pairs:
                    break
                
                sha = commit['sha']
                if sha in processed_commits:
                    continue
                
                # 获取详细信息
                repo = commit['repository']['full_name']
                commit_details = self.get_commit_details(repo, sha)
                
                if not commit_details:
                    continue
                
                # 判断是否是安全修复
                if not self.is_security_fix(commit_details):
                    continue
                
                # 提取代码变更
                changes = self.extract_solidity_changes(commit_details)
                
                for change in changes:
                    if len(spc_pairs) >= max_pairs:
                        break
                    
                    pair = {
                        'pair_id': f"spc_{len(spc_pairs):04d}",
                        'repo': repo,
                        'commit_sha': sha,
                        'commit_message': commit_details['commit']['message'],
                        'commit_date': commit_details['commit']['committer']['date'],
                        'filename': change['filename'],
                        'code_before': change['before'],
                        'code_after': change['after'],
                        'label_before': 'vulnerable',  # 待人工验证
                        'label_after': 'patched',      # 待人工验证
                        'vulnerability_type': self._infer_vulnerability_type(
                            commit_details['commit']['message']
                        ),
                        'needs_manual_review': True
                    }
                    
                    spc_pairs.append(pair)
                
                processed_commits.add(sha)
                time.sleep(0.5)
        
        self.logger.info(f"Collected {len(spc_pairs)} SPC pairs")
        
        # 保存数据
        output_path = self.output_dir / 'spc_pairs_raw.json'
        with open(output_path, 'w') as f:
            json.dump(spc_pairs, f, indent=2)
        
        self.logger.info(f"SPC pairs saved to {output_path}")
        
        return spc_pairs
    
    def _infer_vulnerability_type(self, commit_message: str) -> str:
        """
        根据 commit 消息推断漏洞类型
        """
        message_lower = commit_message.lower()
        
        vulnerability_map = {
            'reentrancy': ['reentrancy', 'reentrant', 're-entrance'],
            'overflow': ['overflow', 'integer overflow'],
            'underflow': ['underflow', 'integer underflow'],
            'access_control': ['access control', 'permission', 'authorization', 'unauthorized'],
            'tx_origin': ['tx.origin', 'tx origin'],
            'timestamp': ['timestamp', 'block.timestamp'],
            'unchecked_call': ['unchecked', 'call return', 'return value'],
            'delegatecall': ['delegatecall', 'delegate call'],
        }
        
        for vuln_type, keywords in vulnerability_map.items():
            if any(keyword in message_lower for keyword in keywords):
                return vuln_type
        
        return 'unknown'
    
    def collect_from_target_repos(self):
        """
        从目标仓库收集数据
        """
        self.logger.info("Collecting from target repositories...")
        
        all_pairs = []
        
        for repo in self.target_repos:
            self.logger.info(f"Processing repository: {repo}")
            
            # 获取仓库的所有 commits
            url = f"https://api.github.com/repos/{repo}/commits"
            params = {'per_page': 100}
            
            commits = self._make_request(url, params)
            if not commits:
                continue
            
            for commit in commits[:50]:  # 限制每个仓库处理的 commit 数
                sha = commit['sha']
                commit_details = self.get_commit_details(repo, sha)
                
                if commit_details and self.is_security_fix(commit_details):
                    changes = self.extract_solidity_changes(commit_details)
                    
                    for change in changes:
                        pair = {
                            'pair_id': f"repo_{repo.replace('/', '_')}_{sha[:8]}",
                            'repo': repo,
                            'commit_sha': sha,
                            'commit_message': commit_details['commit']['message'],
                            'code_before': change['before'],
                            'code_after': change['after'],
                            'vulnerability_type': self._infer_vulnerability_type(
                                commit_details['commit']['message']
                            )
                        }
                        all_pairs.append(pair)
        
        # 保存
        output_path = self.output_dir / 'spc_pairs_from_repos.json'
        with open(output_path, 'w') as f:
            json.dump(all_pairs, f, indent=2)
        
        return all_pairs
    
    def generate_annotation_template(self, spc_pairs: List[Dict]):
        """
        生成标注模板（用于人工审核）
        """
        template_path = Path(self.config['output']['spc_annotated']) / 'annotation_template.json'
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        annotation_data = []
        for pair in spc_pairs:
            annotation_item = {
                'pair_id': pair['pair_id'],
                'code_before': pair['code_before'],
                'code_after': pair['code_after'],
                'inferred_vulnerability': pair.get('vulnerability_type', 'unknown'),
                'commit_message': pair.get('commit_message', ''),
                'annotation': {
                    'is_valid_spc': None,  # True/False
                    'actual_vulnerability_type': None,  # 人工标注的漏洞类型
                    'severity': None,  # low/medium/high/critical
                    'notes': ''  # 额外备注
                }
            }
            annotation_data.append(annotation_item)
        
        with open(template_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        self.logger.info(f"Annotation template saved to {template_path}")
        print(f"\n📝 请人工标注文件: {template_path}")
        print("标注完成后，运行 process_annotations() 方法处理标注数据")


def main():
    """主函数"""
    crawler = GitHubSPCCrawler()
    
    print("=== GitHub SPC Data Collection ===\n")
    
    # 收集 SPC 样本对
    print("1. Collecting SPC pairs from keyword search...")
    spc_pairs = crawler.collect_spc_pairs(max_pairs=500)
    
    print(f"\n2. Collected {len(spc_pairs)} pairs")
    
    # 从目标仓库收集
    print("\n3. Collecting from target repositories...")
    repo_pairs = crawler.collect_from_target_repos()
    
    # 合并数据
    all_pairs = spc_pairs + repo_pairs
    
    # 生成标注模板
    print("\n4. Generating annotation template...")
    crawler.generate_annotation_template(all_pairs)
    
    print("\n✅ SPC data collection completed!")
    print(f"Total pairs collected: {len(all_pairs)}")
    print(f"Output directory: {crawler.output_dir}")


if __name__ == "__main__":
    main()