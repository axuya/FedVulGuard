#!/bin/bash
# crawl_all.sh
chains=("ethereum" "bsc" "polygon" "avalanche")
for ch in "${chains[@]}"; do
  nohup python src/data_collection/etherscan_crawler.py \
      --chain "$ch" \
      --limit 25000 \
      --batch 1000 > logs/"$ch".log 2>&1 &
  echo "👉 $ch 已后台启动，日志: logs/$ch.log"
done