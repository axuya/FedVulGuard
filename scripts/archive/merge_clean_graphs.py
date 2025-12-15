#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from tqdm import tqdm

CLEAN_DIR = Path("data/graphs_cleaned")
OUT_DIR = Path("data/processed_large_scale")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    chunks = sorted(list(CLEAN_DIR.glob("*.jsonl")))
    print(f"共 {len(chunks)} 个 chunk，将合并成单一大文件（jsonl）")

    out_path = OUT_DIR / "all_graphs.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:

        for ck in tqdm(chunks):
            with open(ck, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

    print(f"合并完成！保存为：{out_path}")

if __name__ == "__main__":
    main()
