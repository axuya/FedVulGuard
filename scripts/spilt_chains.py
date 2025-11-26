import os
import re
import shutil
from tqdm import tqdm

RAW_ROOT = "data/raw/sanctuary_full"
OUT_ROOT = "data/raw"

# 正则匹配链来源
CHAIN_PATTERNS = {
    "BSC": re.compile(r"bscscan\.com", re.IGNORECASE),
    "Ethereum": re.compile(r"etherscan\.io", re.IGNORECASE),
    "Polygon": re.compile(r"polygonscan\.com", re.IGNORECASE),
    "Avalanche": re.compile(r"snowtrace\.io", re.IGNORECASE),
    "Arbitrum": re.compile(r"arbiscan\.io", re.IGNORECASE),
    "Fantom": re.compile(r"ftmscan\.com", re.IGNORECASE),
}

def detect_chain(source_code):
    head = source_code[:2000]  # 只检查前 2000 字符
    for chain, pattern in CHAIN_PATTERNS.items():
        if pattern.search(head):
            return chain
    return None

def main():
    print("🔍 scanning sanctuary_full ...")
    for chain in CHAIN_PATTERNS:
        os.makedirs(os.path.join(OUT_ROOT, chain), exist_ok=True)

    total = 0
    assigned = 0

    for root, dirs, files in os.walk(RAW_ROOT):
        for f in files:
            if not f.endswith(".sol"):
                continue

            total += 1
            in_path = os.path.join(root, f)

            # 读取文件头部
            with open(in_path, "r", errors="ignore") as fp:
                head = fp.read(2000)

            chain = detect_chain(head)

            if chain:
                out_path = os.path.join(OUT_ROOT, chain, f)
                shutil.copy(in_path, out_path)
                assigned += 1
            else:
                # 未检测到链的文件留在未分类目录
                unk_dir = os.path.join(OUT_ROOT, "Unknown")
                os.makedirs(unk_dir, exist_ok=True)
                shutil.copy(in_path, os.path.join(unk_dir, f))

            if assigned % 500 == 0:
                print(f"Processed {assigned}/{total}...")

    print(f"✅ 总文件数：{total}")
    print(f"🟢 成功分类：{assigned}")
    print("📂 输出目录：data/raw/<ChainName>/")

if __name__ == "__main__":
    main()