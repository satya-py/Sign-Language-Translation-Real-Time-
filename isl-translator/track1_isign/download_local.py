"""Fallback: download the iSign part-AD features straight to this laptop.

Only needed if copying the 4.75 GB Kaggle output is not possible. It downloads
about 25 GB from HuggingFace, so the Kaggle route is usually better.

    set HF_TOKEN=hf_xxx           (Windows CMD)       # your own read token
    $env:HF_TOKEN="hf_xxx"        (PowerShell)
    python download_local.py --out "D:/isign_partad_features"

Safe to stop and re-run: finished files are skipped.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

import isign_zip
from features import pose_bytes_to_features


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--part", type=int, default=3, help="3 = part_ad, the paper's subset")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set the HF_TOKEN environment variable to your HuggingFace read token first.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "iSign_v1.1.csv"
    if not csv_path.exists():
        r = requests.get(isign_zip.hf_url("iSign_v1.1.csv"),
                         headers={"Authorization": f"Bearer {token}"}, timeout=120)
        if r.status_code in (401, 403):
            sys.exit("Accept the iSign terms on HuggingFace, then check your token.")
        r.raise_for_status()
        csv_path.write_bytes(r.content)
    print("annotations:", pd.read_csv(csv_path).shape)

    sizes = isign_zip.part_sizes(token)
    src = isign_zip.MultiPartSource(isign_zip.PART_NAMES, sizes, isign_zip.http_fetcher(token))
    members = [m for m in isign_zip.list_members(src) if m.filename.endswith(".pose")]
    selected = isign_zip.members_in_part(members, src, args.part)
    print(f"{len(selected):,} pose files in part {isign_zip.PART_NAMES[args.part]}, "
          f"{sum(m.compress_size for m in selected) / 1e9:.1f} GB to fetch")

    uid_of = lambda m: os.path.splitext(os.path.basename(m.filename))[0]

    def work(m):
        path = out / f"{uid_of(m)}.npy"
        if path.exists():
            return True
        try:
            feats = pose_bytes_to_features(isign_zip.read_member(src, m))
            if feats.size == 0:
                return False
            np.save(path, feats.astype(np.float32))
            return True
        except Exception as e:
            print(f"  {uid_of(m)}: {type(e).__name__}: {e}", flush=True)
            return False

    for attempt in range(1, 6):
        todo = [m for m in selected if not (out / f"{uid_of(m)}.npy").exists()]
        print(f"round {attempt}: {len(todo):,} files to fetch", flush=True)
        if not todo:
            break
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(args.threads) as pool:
            for ok in pool.map(work, todo):
                done += 1
                if done % 250 == 0:
                    print(f"  {done:,}/{len(todo):,}  {(time.time()-t0)/60:.1f} min", flush=True)
        time.sleep(30)

    have = sum(1 for _ in out.glob("*.npy"))
    size = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
    print(f"done: {have:,} feature files, {size:.2f} GB in {out}")


if __name__ == "__main__":
    main()
