#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel_parse_worker.py
并行将 sanctuary_raw（或任意源码文件夹）解析为 graphs_raw/*.jsonl（chunked）。
特点：
 - 支持断点续传（processed_ids_file）
 - 支持多进程并行（multiprocessing.Pool）
 - 优先使用 python solidity_parser 解析 AST（可通过 pip 安装）
 - 输出 JSONL，每行为一个样本（id, chain, src_path, src_code, ast, nodes, edges, metadata）
 - chunk_size 控制每个 jsonl 文件包含多少样本
用法示例：
 python src/preprocessors/parallel_parse_worker.py \
    --input-dir data/raw/sanctuary_full/ethereum \
    --out-dir data/graphs_raw \
    --workers 8 --chunk-size 2000
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool, Manager
from functools import partial
from datetime import datetime

# Try import solidity_parser; if not available, we fallback to minimal behavior
try:
    from solidity_parser import parser as sol_parser
    HAS_SOL_PARSER = True
except Exception:
    HAS_SOL_PARSER = False

# ---------- helpers ----------
def setup_logger():
    log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    logger = logging.getLogger("parse_worker")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.handlers = [ch]
    return logger

logger = setup_logger()

def compute_sha256(text: str):
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def safe_read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            logger.warning(f"无法读取文件 {path}: {e}")
            return ""

def build_sample_from_sol(path: Path, chain_tag: str = ""):
    """
    尝试解析 .sol 文件并提取 AST 与简单节点（函数/contract/variable）
    若解析失败，仍返回源码与 minimal metadata，保证不会抛出异常
    """
    src = safe_read_text(path)
    src_hash = compute_sha256(src)
    sample_id = f"{chain_tag}__{src_hash[:16]}"

    sample = {
        "id": sample_id,
        "chain": chain_tag,
        "src_path": str(path),
        "src_hash": src_hash,
        "src_code": src,
        "ast": None,
        "nodes": [],
        "edges": [],
        "metadata": {
            "filename": path.name,
            "filesize": path.stat().st_size if path.exists() else 0,
        }
    }

    if not src.strip():
        sample["metadata"]["note"] = "empty_or_unreadable"
        return sample

    if HAS_SOL_PARSER:
        try:
            ast = sol_parser.parse(src)
            sample["ast"] = ast
            # 简单从 AST 提取节点（contract / function / stateVariable）
            nodes = []
            def extract_nodes(node):
                if not isinstance(node, dict):
                    return
                typ = node.get("type")
                if typ in ("ContractDefinition", "FunctionDefinition", "VariableDeclaration"):
                    name = node.get("name") or node.get("name") or node.get("typeName", {}).get("name") if isinstance(node.get("typeName"), dict) else None
                    nodes.append({"type": typ, "name": name})
                # 递归
                for k, v in node.items():
                    if isinstance(v, dict):
                        extract_nodes(v)
                    elif isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                extract_nodes(it)
            extract_nodes(ast)
            sample["nodes"] = nodes
        except Exception as e:
            # 解析异常：记录异常信息，但仍返回源码
            sample["ast"] = None
            sample["metadata"]["parse_error"] = str(e)
    else:
        sample["metadata"]["note"] = "solidity_parser_not_installed"
        # 作为 fallback：按行做简单节点（函数/contract关键字）
        lines = src.splitlines()
        nodes = []
        for i, ln in enumerate(lines):
            ln_strip = ln.strip()
            if ln_strip.startswith("contract "):
                name = ln_strip.split()[1].split("{")[0].strip()
                nodes.append({"type": "ContractDefinition", "name": name})
            elif ln_strip.startswith("function "):
                # function foo(...)
                name = ln_strip.split()[1].split("(")[0].strip()
                nodes.append({"type": "FunctionDefinition", "name": name})
        sample["nodes"] = nodes

    return sample

# ---------- multiprocessing worker ----------
def worker_process_file(args):
    """
    Worker entry: parse a single file path, return JSON-serializable dict
    args: tuple (file_path_str, chain_tag)
    """
    file_path_str, chain_tag = args
    path = Path(file_path_str)
    try:
        sample = build_sample_from_sol(path, chain_tag)
        return sample
    except Exception as e:
        logger.exception(f"Worker 解析失败: {path} : {e}")
        return None

# ---------- main orchestration ----------
def gather_sol_files(input_dir: Path, allowed_exts=(".sol",)):
    """
    递归收集源码文件路径列表
    返回 list of (path_str, chain_tag)
    chain_tag 通过 input_dir 的上一级目录名或 'unknown' 推断
    """
    files = []
    input_dir = Path(input_dir)
    # chain tag can be parent folder name if input_dir ends with chain folder,
    # else default to folder name
    base_chain = input_dir.name
    for root, _, fnames in os.walk(input_dir):
        for fn in fnames:
            if fn.lower().endswith(allowed_exts):
                full = Path(root) / fn
                files.append((str(full), base_chain))
    return files

def write_chunk(samples, out_dir: Path, chunk_idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{chunk_idx:05d}.jsonl"
    tmp_fname = out_dir / f".{chunk_idx:05d}.jsonl.tmp"
    # 如果文件已存在，跳过（防止并发覆盖）
    if fname.exists():
        logger.info(f"跳过已存在 chunk {fname}")
        return fname
    with tmp_fname.open("w", encoding="utf-8") as w:
        for s in samples:
            if s is None:
                continue
            w.write(json.dumps(s, ensure_ascii=False) + "\n")
    tmp_fname.replace(fname)
    logger.info(f"写入 chunk {fname} ({len(samples)} 样本)")
    return fname

def load_processed_set(processed_file: Path):
    if not processed_file.exists():
        return set()
    try:
        with processed_file.open("r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception:
        return set()

def append_processed_ids(processed_file: Path, ids):
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    with processed_file.open("a", encoding="utf-8") as f:
        for _id in ids:
            f.write(_id + "\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True, help="源码根目录（会递归查找 .sol）")
    p.add_argument("--out-dir", type=str, required=True, help="输出 graphs_raw 目录（jsonl chunk）")
    p.add_argument("--workers", type=int, default=8, help="并行 worker 数量")
    p.add_argument("--chunk-size", type=int, default=2000, help="每个 jsonl chunk 的样本数量")
    p.add_argument("--processed-file", type=str, default="data/processed/parsed_ids.txt", help="已处理 id 列表（断点续传）")
    p.add_argument("--max-samples", type=int, default=0, help="最多解析多少样本（0 表示全部）")
    p.add_argument("--shuffle", action="store_true", help="是否随机打乱样本顺序再处理")
    return p.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    processed_file = Path(args.processed_file)

    logger.info("启动 parallel_parse_worker")
    logger.info(f"input_dir={input_dir}, out_dir={out_dir}, workers={args.workers}, chunk_size={args.chunk_size}")

    files = gather_sol_files(input_dir)
    total_files = len(files)
    logger.info(f"发现候选源码文件数量: {total_files}")

    # load processed ids
    processed_ids = load_processed_set(processed_file)
    logger.info(f"已记录的已处理样本数量: {len(processed_ids)}")

    # 过滤掉已处理的（通过文件 hash id 判断）
    filtered = []
    for (fp, chain) in files:
        # quick compute file content hash to determine id (avoid parsing heavy)
        try:
            text = safe_read_text(Path(fp))
            if not text:
                continue
            sid = f"{chain}__{compute_sha256(text)[:16]}"
            if sid in processed_ids:
                continue
            filtered.append((fp, chain))
        except Exception:
            continue

    logger.info(f"待处理文件数量: {len(filtered)}")
    if args.shuffle:
        import random
        random.shuffle(filtered)

    if args.max_samples > 0:
        filtered = filtered[:args.max_samples]
        logger.info(f"max_samples 限制后数量: {len(filtered)}")

    # multiprocessing pool
    pool = Pool(processes=max(1, args.workers))
    try:
        chunk = []
        chunk_idx = 0
        processed_batch_ids = []
        # Create iterable of args
        iterable = filtered
        for i, res in enumerate(pool.imap_unordered(worker_process_file, iterable, chunksize=16), 1):
            # res is sample dict or None
            if res is None:
                continue
            chunk.append(res)
            processed_batch_ids.append(res["id"])
            # 当 chunk 满了就写入文件并清空
            if len(chunk) >= args.chunk_size:
                write_chunk(chunk, out_dir, chunk_idx)
                append_processed_ids(processed_file, processed_batch_ids)
                chunk_idx += 1
                chunk = []
                processed_batch_ids = []
                logger.info(f"已完成 {i} / {len(filtered)}")
        # 写入剩余的
        if chunk:
            write_chunk(chunk, out_dir, chunk_idx)
            append_processed_ids(processed_file, processed_batch_ids)
            logger.info("写入最后一块 chunk")

    except KeyboardInterrupt:
        logger.warning("接收到 KeyboardInterrupt，准备退出并保存已处理数据")
    finally:
        pool.close()
        pool.join()

    logger.info("解析完成: " + datetime.now().isoformat())

if __name__ == "__main__":
    main()
