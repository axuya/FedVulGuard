import json
import subprocess
from pathlib import Path
from tqdm import tqdm

# =========================
# 路径配置
# =========================
IN_DIR = Path("data/train/crossgraphnet_lite")
OUT_DIR = Path("data/train/crossgraphnet_lite_labeled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_JSON = Path("slither_tmp.json")

# =========================
# 高置信度漏洞 detector 白名单（论文级）
# =========================
STRONG_DETECTORS = {
    "reentrancy-eth",
    "reentrancy-no-eth",
    "uninitialized-state",
    "uninitialized-storage",
    "arbitrary-send",
    "controlled-delegatecall",
    "suicidal",
    "tx-origin",
    "shadowing-state",
    "shadowing-abstract",
}

# =========================
# 单合约 Slither 执行
# =========================
def run_slither(sol_path: Path) -> bool:
    """
    返回：
      True  -> 命中高置信度漏洞 detector
      False -> 未命中 / 失败
    """
    if not sol_path.exists():
        return False

    cmd = [
        "slither",
        str(sol_path),
        "--json",
        str(TMP_JSON),
        "--disable-color",
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
            check=False,
        )
    except Exception:
        return False

    if not TMP_JSON.exists():
        return False

    try:
        with TMP_JSON.open("r", encoding="utf8") as f:
            report = json.load(f)
    except Exception:
        TMP_JSON.unlink(missing_ok=True)
        return False

    TMP_JSON.unlink(missing_ok=True)

    detectors = report.get("results", {}).get("detectors", [])
    if not detectors:
        return False

    # ⭐ 只匹配白名单 detector
    for d in detectors:
        check = (d.get("check") or "").lower()
        if check in STRONG_DETECTORS:
            return True

    return False


# =========================
# 主流程
# =========================
def main():
    for in_path in IN_DIR.glob("*.jsonl"):
        out_path = OUT_DIR / in_path.name
        print(f"\n[LABEL] {in_path.name}")

        total = 0
        positive = 0

        with in_path.open("r", encoding="utf8") as fin, \
             out_path.open("w", encoding="utf8") as fout:

            for line in tqdm(fin):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                total += 1

                # 🔑 正确获取 Solidity 源文件路径
                src_path = (
                    obj.get("src_path")
                    or obj.get("graphs", {})
                           .get("ast", {})
                           .get("id")
                )

                if src_path:
                    is_vul = run_slither(Path(src_path))
                else:
                    is_vul = False

                obj["label"] = 1 if is_vul else 0
                if obj["label"] == 1:
                    positive += 1

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        ratio = positive / max(total, 1)
        print(f"  Total: {total}")
        print(f"  Positive (vulnerable): {positive}")
        print(f"  Positive ratio: {ratio:.4f}")
        print(f"  Output → {out_path}")


if __name__ == "__main__":
    main()
