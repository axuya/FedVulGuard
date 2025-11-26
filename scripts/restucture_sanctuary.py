#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from tqdm import tqdm

"""
将 sanctuary_full 中大量平铺的 .sol 文件进行“分桶”重排：
input:  data/raw/sanctuary_full/*.sol
output: data/raw/sanctuary_full/<bucket>/*.sol
bucket = 文件名前 2 个字符（通常是 hash/address 的前两位）

运行方法：
python scripts/restructure_sanctuary.py
"""

ROOT = Path("data/raw/sanctuary_full")

def main():
    sol_files = list(ROOT.glob("*.sol"))
    print(f"发现 {len(sol_files)} 个 .sol 文件，开始分桶...")

    for sol in tqdm(sol_files):
        filename = sol.name
        # 前两位作为 bucket
        bucket = filename[:2].lower()

        bucket_dir = ROOT / bucket
        bucket_dir.mkdir(exist_ok=True, parents=True)

        # 新路径
        target_path = bucket_dir / filename

        # 迁移
        shutil.move(str(sol), str(target_path))

    print("完成！sanctuary_full 已成功重构为按前两位 hash 分桶的结构。")
    print("你现在可以安全、高效地运行 graph 构建脚本。")

if __name__ == "__main__":
    main()
