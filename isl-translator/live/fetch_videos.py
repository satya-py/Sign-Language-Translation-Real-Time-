"""Bulk-download INCLUDE videos for the chosen demo words (training data).

Groups requests by zip so each archive's index is read once, then pulls only the
wanted members with HTTP range requests. Resumable: existing files are skipped.

    python fetch_videos.py --words hello,i,you --per-word 21
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "track1_isign" / "src"))
import isign_zip  # noqa: E402

from fetch_reference import INDEX_PATH, remote_zip, zenodo_files  # noqa: E402

VIDEO_DIR = HERE / "data" / "videos"

# 15 signs that combine into useful sentences and all exist in INCLUDE.
DEMO_WORDS = ["hello", "thank_you", "how_are_you", "i", "you", "doctor", "hospital",
              "sick", "today", "tomorrow", "good", "bad", "house", "school", "friend"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--words", default=",".join(DEMO_WORDS))
    p.add_argument("--per-word", type=int, default=21)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default=str(VIDEO_DIR))
    args = p.parse_args()

    index = json.loads(INDEX_PATH.read_text())
    words = args.words.split(",")
    out_root = Path(args.out)

    missing = [w for w in words if w not in index]
    if missing:
        sys.exit(f"not in INCLUDE: {missing}")

    # zip -> [(word, member path)]
    by_zip = defaultdict(list)
    for w in words:
        out = out_root / w
        out.mkdir(parents=True, exist_ok=True)
        have = {p.name for p in out.iterdir()}
        for e in index[w][:args.per_word]:
            if Path(e["path"]).name not in have:
                by_zip[e["zip"]].append((w, e["path"]))

    total = sum(len(v) for v in by_zip.values())
    print(f"{total} videos to download from {len(by_zip)} archives", flush=True)
    if not total:
        return

    session = requests.Session()
    files = zenodo_files()
    done = 0
    t0 = time.time()
    for zip_name, items in by_zip.items():
        src, _ = remote_zip(files[zip_name], session)
        members = {m.filename: m for m in isign_zip.list_members(src)}

        def grab(item):
            word, path = item
            dest = out_root / word / Path(path).name
            if dest.exists():
                return True
            m = members.get(path)
            if m is None:
                print(f"  missing in archive: {path}", flush=True)
                return False
            for attempt in range(3):
                try:
                    dest.write_bytes(isign_zip.read_member(src, m))
                    return True
                except Exception as e:
                    if attempt == 2:
                        print(f"  failed {path}: {type(e).__name__}: {e}", flush=True)
                        return False
                    time.sleep(3 * (attempt + 1))

        with ThreadPoolExecutor(args.threads) as pool:
            for ok in pool.map(grab, items):
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"{done}/{total} videos, {(time.time()-t0)/60:.1f} min", flush=True)

    counts = {w: len(list((out_root / w).iterdir())) for w in words}
    size = sum(f.stat().st_size for w in words for f in (out_root / w).iterdir()) / 1e9
    print(f"done: {counts}")
    print(f"total {size:.1f} GB in {out_root}")


if __name__ == "__main__":
    main()
