# FedVulGuard 数据收集指南

## 📋 准备工作

### 1. 安装依赖

```bash
# 激活 conda 环境
conda activate fedvul

# 安装必要的包
pip install requests pyyaml tqdm
```

### 2. 配置 API Keys

编辑 `configs/data_collection.yaml`:

```yaml
etherscan:
  api_key: "YOUR_ETHERSCAN_API_KEY"  # 替换为你的 key

github:
  token: "YOUR_GITHUB_TOKEN"  # 替换为你的 token
```

#### 获取 Etherscan API Key:
1. 访问 https://etherscan.io/
2. 注册账号并登录
3. 进入 "API Keys" 页面
4. 创建新的 API Key
5. 免费版限制：5 次请求/秒

#### 获取 GitHub Token:
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选以下权限：
   - `repo` (访问仓库)
   - `public_repo` (访问公开仓库)
4. 生成并复制 token
5. 认证用户限制：5000 次请求/小时

### 3. 创建必要目录

```bash
mkdir -p scripts
mkdir -p src/{data_collection,preprocessing,utils}
mkdir -p logs
```

## 🚀 运行数据收集

### 方式 1: 完整流程（推荐）

```bash
cd ~/FedVulGuard
python scripts/run_data_collection.py
```

这会依次执行：
1. 从现有数据集提取地址
2. 爬取 Etherscan 合约
3. 收集 GitHub SPC 数据
4. 合并和验证

### 方式 2: 分步执行

```bash
# 步骤 1: 提取地址
python scripts/run_data_collection.py --step 1

# 步骤 2: 爬取 Etherscan（使用已知 DeFi 地址）
python scripts/run_data_collection.py --step 2 --etherscan-mode known

# 或使用从数据集提取的地址
python scripts/run_data_collection.py --step 2 --etherscan-mode extracted

# 步骤 3: 收集 SPC 数据（指定目标数量）
python scripts/run_data_collection.py --step 3 --spc-pairs 500

# 步骤 4: 合并数据
python scripts/run_data_collection.py --step 4
```

### 方式 3: 单独运行爬虫

```bash
# 只运行 Etherscan 爬虫
cd ~/FedVulGuard
python src/data_collection/etherscan_crawler.py

# 只运行 GitHub SPC 爬虫
python src/data_collection/github_spc_crawler.py
```

## 📊 输出结构

```
data/
├── etherscan/
│   ├── raw/
│   │   ├── batch_0000.json          # 原始合约数据
│   │   ├── batch_0001.json
│   │   ├── statistics.json          # 统计信息
│   │   └── failed_addresses.json    # 失败的地址
│   └── processed/
│       └── filtered_contracts.json  # 过滤后的合约
├── spc_data/
│   ├── raw/
│   │   ├── spc_pairs_raw.json       # 关键词搜索的 SPC
│   │   ├── spc_pairs_from_repos.json # 目标仓库的 SPC
│   │   └── merged_spc_pairs.json    # 合并后的 SPC
│   └── annotated/
│       └── annotation_template.json  # 标注模板
└── contract_addresses.txt            # 提取的地址列表
```

## 🔍 数据说明

### Etherscan 合约数据格式

```json
{
  "address": "0x...",
  "SourceCode": "pragma solidity...",
  "ContractName": "MyContract",
  "CompilerVersion": "v0.8.0+commit...",
  "OptimizationUsed": "1",
  "Runs": "200",
  "ConstructorArguments": "",
  "EVMVersion": "Default",
  "Library": "",
  "LicenseType": "MIT",
  "Proxy": "0",
  "Implementation": "",
  "SwarmSource": "",
  "crawled_at": "2024-01-01T00:00:00",
  "code_hash": "abc123..."
}
```

### SPC 数据格式

```json
{
  "pair_id": "spc_0001",
  "repo": "OpenZeppelin/openzeppelin-contracts",
  "commit_sha": "abc123...",
  "commit_message": "Fix reentrancy vulnerability",
  "commit_date": "2024-01-01T00:00:00Z",
  "filename": "contracts/token/ERC20.sol",
  "code_before": "function withdraw() public { ... }",
  "code_after": "function withdraw() public nonReentrant { ... }",
  "label_before": "vulnerable",
  "label_after": "patched",
  "vulnerability_type": "reentrancy",
  "needs_manual_review": true
}
```

## 🏷️ SPC 数据标注

### 标注流程

1. 打开 `data/spc_data/annotated/annotation_template.json`
2. 对于每个样本对，填写以下字段：
   ```json
   "annotation": {
     "is_valid_spc": true,  // 是否是有效的 SPC 对
     "actual_vulnerability_type": "reentrancy",  // 实际漏洞类型
     "severity": "high",  // 严重程度: low/medium/high/critical
     "notes": "Classic reentrancy in withdraw function"  // 备注
   }
   ```

3. 标注完成后保存为 `annotated_spc_pairs.json`

### 漏洞类型参考

- **reentrancy**: 重入攻击
- **overflow**: 整数溢出
- **underflow**: 整数下溢
- **access_control**: 访问控制漏洞
- **tx_origin**: tx.origin 使用不当
- **timestamp**: 时间戳依赖
- **unchecked_call**: 未检查的外部调用
- **delegatecall**: delegatecall 使用不当

### 标注示例

```json
{
  "pair_id": "spc_0001",
  "code_before": "function withdraw(uint amount) public {\n    require(balances[msg.sender] >= amount);\n    msg.sender.call.value(amount)();\n    balances[msg.sender] -= amount;\n}",
  "code_after": "function withdraw(uint amount) public {\n    require(balances[msg.sender] >= amount);\n    balances[msg.sender] -= amount;\n    msg.sender.call.value(amount)();\n}",
  "inferred_vulnerability": "reentrancy",
  "annotation": {
    "is_valid_spc": true,
    "actual_vulnerability_type": "reentrancy",
    "severity": "critical",
    "notes": "Classic reentrancy: external call before state update"
  }
}
```

## ⚠️ 常见问题

### 1. API 速率限制

**问题**: `Rate limit exceeded`

**解决**:
- Etherscan: 等待后重试，或升级到付费 API
- GitHub: 检查 token 是否有效，等待 1 小时后重试

### 2. 没有找到合约地址

**问题**: Step 1 没有提取到地址

**解决**:
- 确保 SmartBugs 和 SolidiFI 数据集已下载
- 手动创建 `data/contract_addresses.txt` 并添加地址
- 使用 `--etherscan-mode known` 使用已知 DeFi 地址

### 3. GitHub 搜索无结果

**问题**: 搜索不到相关 commits

**解决**:
- 检查 GitHub token 是否有效
- 尝试其他关键词
- 直接从目标仓库收集（已在代码中实现）

### 4. 合约编译失败

**问题**: 下载的合约无法编译

**解决**:
- 这是正常的，后续会有过滤步骤
- 在 Phase 2 预处理阶段会使用 Slither 进行验证

## 📈 预期结果

成功运行后，你应该获得：

- ✅ **Etherscan 数据**: 10,000+ 验证合约
- ✅ **SPC 数据**: 500+ 样本对（需人工标注）
- ✅ **统计报告**: 数据分布和质量分析
- ✅ **日志文件**: 详细的爬取记录

## 🔄 下一步

完成数据收集后：

1. **人工标注 SPC 数据** (预计 2-3 天)
2. **运行数据预处理** (Phase 2)
   ```bash
   python scripts/preprocess_data.py
   ```
3. **构建多图表示** (Phase 3)
   ```bash
   python scripts/build_graphs.py
   ```

## 💡 优化建议

### 增加数据量

如需收集更多数据：

```bash
# 增加 SPC 目标数量
python scripts/run_data_collection.py --step 3 --spc-pairs 1000

# 添加更多合约地址
echo "0x..." >> data/contract_addresses.txt
python scripts/run_data_collection.py --step 2 --etherscan-mode extracted
```

### 并行爬取

对于大规模爬取，可以修改代码使用多线程：

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(crawler.get_contract_source, addr) 
               for addr in addresses]
```

### 多链爬取

在 `configs/data_collection.yaml` 中配置其他链的 API，然后修改爬虫逻辑支持多链。

## 📞 需要帮助？

如果遇到问题：
1. 检查 `logs/` 目录下的日志文件
2. 查看 GitHub Issues
3. 联系项目维护者