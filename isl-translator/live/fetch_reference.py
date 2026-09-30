"""Fetch a few INCLUDE reference videos per word, so the signer can learn the sign.

INCLUDE (CC BY 4.0) is 57 GB of ISL videos by deaf signers. Zenodo supports HTTP
range requests, so we read each zip's index remotely and download only the handful
of videos we actually want (about 13 MB each).

    python fetch_reference.py --index                  # build the word -> zip index (once)
    python fetch_reference.py --words hello,i,you --per-word 2
    python fetch_reference.py --list                   # show all 263 available words
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "track1_isign" / "src"))
import isign_zip  # noqa: E402  (reuses the tested range-request zip reader)

ZENODO_RECORD = "https://zenodo.org/api/records/4010759"
INDEX_PATH = HERE / "reference" / "include_index.json"
REF_DIR = HERE / "reference"

# Categories that hold the words useful for a demo sentence. Listing every zip
# takes ~20 s each, so we skip animals, colours, clothes, electronics, transport.
DEFAULT_CATEGORIES = ["Greetings", "Pronouns", "Jobs", "Places", "Adjectives",
                      "Days_and_Time", "People"]


def clean_word(folder):
    """'48. Hello' -> 'hello', '49. How are you' -> 'how_are_you'."""
    name = re.sub(r"^\s*\d+\.\s*", "", folder).strip().lower()
    return re.sub(r"[^a-z]+", "_", name).strip("_")


def zenodo_files():
    r = requests.get(ZENODO_RECORD, timeout=60)
    r.raise_for_status()
    return {f["key"]: f for f in r.json()["files"] if f["key"].endswith(".zip")}


def remote_zip(entry, session):
    url, size = entry["links"]["self"], entry["size"]

    def fetch(part, start, end):
        for attempt in range(4):
            try:
                rr = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
                if rr.status_code == 206:
                    return rr.content
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise IOError(f"range request failed for {entry['key']} {start}-{end}")

    return isign_zip.MultiPartSource([entry["key"]], [size], fetch), fetch


def build_index(categories):
    """word -> [{zip, path, offset info}] for every video, read remotely."""
    session = requests.Session()
    files = zenodo_files()
    wanted = [k for k in sorted(files) if k.rsplit("_", 1)[0] in categories]
    index = {}
    if INDEX_PATH.exists():
        index = json.loads(INDEX_PATH.read_text())
    for i, key in enumerate(wanted, 1):
        if any(v[0]["zip"] == key for v in index.values()):
            print(f"[{i}/{len(wanted)}] {key}: already indexed")
            continue
        t = time.time()
        src, _ = remote_zip(files[key], session)
        try:
            members = isign_zip.list_members(src)
        except Exception as e:
            print(f"[{i}/{len(wanted)}] {key}: FAILED {e}")
            continue
        n = 0
        for m in members:
            parts = m.filename.split("/")
            if len(parts) < 3 or not parts[-1].lower().endswith((".mov", ".mp4")):
                continue
            word = clean_word(parts[-2])
            index.setdefault(word, []).append({"zip": key, "path": m.filename,
                                               "size": m.compress_size})
            n += 1
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps(index, indent=1))
        print(f"[{i}/{len(wanted)}] {key}: {n} videos, {time.time()-t:.0f}s "
              f"({len(index)} words so far)", flush=True)
    return index


def download_words(index, words, per_word):
    session = requests.Session()
    files = zenodo_files()
    for word in words:
        if word not in index:
            close = [w for w in index if word.replace("_", "") in w.replace("_", "")]
            print(f"'{word}' not in INCLUDE. Similar: {close[:5] or 'none'}")
            continue
        out = REF_DIR / word
        out.mkdir(parents=True, exist_ok=True)
        have = len(list(out.glob("*.MOV"))) + len(list(out.glob("*.mp4")))
        if have >= per_word:
            print(f"{word}: already have {have}")
            continue
        entries = sorted(index[word], key=lambda e: e["size"])[:per_word]
        for e in entries:
            src, _ = remote_zip(files[e["zip"]], session)
            members = {m.filename: m for m in isign_zip.list_members(src)}
            m = members.get(e["path"])
            if m is None:
                print(f"  {word}: {e['path']} vanished from {e['zip']}")
                continue
            dest = out / Path(e["path"]).name
            if dest.exists():
                continue
            t = time.time()
            dest.write_bytes(isign_zip.read_member(src, m))
            print(f"{word}: {dest.name} {dest.stat().st_size/1e6:.1f} MB "
                  f"({time.time()-t:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", action="store_true", help="build/extend the word index")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    p.add_argument("--words", default=None, help="comma separated words to download")
    p.add_argument("--per-word", type=int, default=2)
    p.add_argument("--list", action="store_true", help="print the indexed words")
    args = p.parse_args()

    index = json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else {}
    if args.index:
        index = build_index(args.categories.split(","))
    if args.list:
        print(f"{len(index)} words indexed:")
        print(", ".join(sorted(index)))
    if args.words:
        if not index:
            sys.exit("No index yet. Run with --index first.")
        download_words(index, args.words.split(","), args.per_word)


if __name__ == "__main__":
    main()
