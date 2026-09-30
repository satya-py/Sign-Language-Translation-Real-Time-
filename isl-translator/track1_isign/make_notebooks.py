"""Generates the two Kaggle notebooks from the tested sources in src/.

    python make_notebooks.py

Each notebook writes the exact src/*.py files with %%writefile, so Kaggle runs
byte-for-byte the code that tests/ verified locally.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"


def notebook(cells):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}}}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n")}


def writefile(name):
    return code(f"%%writefile {name}\n" + (SRC / name).read_text(encoding="utf-8"))


# ============================================================ notebook 1: download
nb1 = [
    md("""
# iSign Track 1 · Notebook 1 — download the paper's "Part AD" subset

Reproduces the data of **arXiv 2609.12993** ("Investigating Temporal Motion Features for
Pose-to-Text ISL Translation"): *"we use the iSign poses v1.1 Part AD subset, containing pose
representations for 18,867 sign language samples."*

iSign stores all poses as one 170 GB zip split into 4 parts. This notebook reads the zip's
index from the end of the last part and downloads **only the part-AD files** with HTTP range
requests, so nothing large is ever stored. Each `.pose` file is converted with the paper's own
`load_pose_file` into a `(frames, 300)` float32 array (33 body + 21 left-hand + 21 right-hand
landmarks × x, y, z, confidence) and saved as `<uid>.npy`.

### Before running
1. On https://huggingface.co/datasets/Exploration-Lab/iSign click **"Agree and access"**
   (auto-approved; licence CC-BY-NC-SA-4.0, non-commercial).
2. Create a **read** token at https://huggingface.co/settings/tokens
3. In this notebook: **Add-ons → Secrets → Add secret**, label `HF_TOKEN`, paste the token,
   and tick it for this notebook.
4. Settings (right panel): **Accelerator: None** (CPU only, saves GPU quota), **Internet: On**.
5. **Save Version → Save & Run All (Commit)**. About 30–90 min (it downloads about 25 GB).

### After it finishes
Open the finished version → **Output** → **New Dataset** → name it **`isign-partad-features`**
(private). Notebook 2 trains on that dataset.
"""),
    code('!pip install -q "pose-format==0.15.0"'),
    md("### Source files (identical to the locally tested versions)"),
    writefile("paper_pose_utils.py"),
    writefile("isign_zip.py"),
    writefile("features.py"),
    md("### Configuration"),
    code("""
import os, sys, time, json
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
import isign_zip
from features import pose_bytes_to_features

OUT = "/kaggle/working/isign_partad_features"
os.makedirs(OUT, exist_ok=True)
PART_INDEX = 3      # 0=aa, 1=ab, 2=ac, 3=ad. The paper used part AD.
THREADS = 6         # parallel range downloads (16 triggered HTTP 429 rate limits)

from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
"""),
    md("### Step 1 · annotations CSV"),
    code("""
r = requests.get(isign_zip.hf_url("iSign_v1.1.csv"),
                 headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=120)
if r.status_code in (401, 403):
    raise PermissionError("Accept the iSign terms on HuggingFace and check the HF_TOKEN secret.")
r.raise_for_status()
open(f"{OUT}/iSign_v1.1.csv", "wb").write(r.content)
csv_df = pd.read_csv(f"{OUT}/iSign_v1.1.csv")
print(csv_df.shape, list(csv_df.columns))
csv_df.head()
"""),
    md("### Step 2 · read the zip index and select the part-AD members"),
    code("""
sizes = isign_zip.part_sizes(HF_TOKEN)
print("part sizes (GB):", [round(s / 1e9, 2) for s in sizes])
src = isign_zip.MultiPartSource(isign_zip.PART_NAMES, sizes, isign_zip.http_fetcher(HF_TOKEN))

t = time.time()
members = isign_zip.list_members(src)
print(f"{len(members):,} files in the whole archive ({time.time()-t:.0f}s)")
pose_members = [m for m in members if m.filename.endswith(".pose")]
selected = isign_zip.members_in_part(pose_members, src, PART_INDEX)
print(f"{len(selected):,} .pose files in part {isign_zip.PART_NAMES[PART_INDEX]}  "
      "(paper reports 18,867)")
print("example names:", [m.filename for m in selected[:3]])
print("GB to download:", round(sum(m.compress_size for m in selected) / 1e9, 2))

uid_of = lambda m: os.path.splitext(os.path.basename(m.filename))[0]
in_csv = set(csv_df["uid"].astype(str))
print("selected files with a CSV row:", sum(uid_of(m) in in_csv for m in selected))
"""),
    md("### Step 3 · download + convert (resumable: already-saved files are skipped)"),
    code("""
def work(m):
    uid = uid_of(m)
    path = f"{OUT}/{uid}.npy"
    if os.path.exists(path):
        return uid, "exists", None
    try:
        feats = pose_bytes_to_features(isign_zip.read_member(src, m))
        if feats.size == 0:
            return uid, "empty", 0
        np.save(path, feats.astype(np.float32))
        return uid, "ok", feats.shape[0]
    except Exception as e:
        return uid, f"error: {type(e).__name__}: {e}", None

rows, t = [], time.time()
with ThreadPoolExecutor(THREADS) as pool:
    futures = [pool.submit(work, m) for m in selected]
    for i, f in enumerate(as_completed(futures), 1):
        rows.append(f.result())
        if i % 500 == 0 or i == len(futures):
            ok = sum(r[1] in ("ok", "exists") for r in rows)
            print(f"{i:,}/{len(futures):,} done, {ok:,} saved, {(time.time()-t)/60:.1f} min",
                  flush=True)

status = pd.DataFrame(rows, columns=["uid", "status", "frames"])
status.to_csv(f"{OUT}/download_status.csv", index=False)
print(status.status.str.split(":").str[0].value_counts())
"""),
    md("### Step 4 · summary"),
    code("""
saved = status[status.status.isin(["ok", "exists"])]
print(f"saved feature files: {len(saved):,}")
print("frames per sample:", saved.frames.dropna().describe().round(1).to_dict())
print("output size (GB):", round(sum(os.path.getsize(os.path.join(OUT, f))
                                     for f in os.listdir(OUT)) / 1e9, 2))
print(status[~status.status.isin(["ok", "exists"])].head(20))
"""),
]

# ============================================================== notebook 2: train
nb2 = [
    md("""
# iSign Track 1 · Notebook 2 — train + evaluate (exact paper recipe)

Reproduces **arXiv 2609.12993** with the authors' own architecture and settings:

| | Paper setting |
|---|---|
| Input | 33 body + 2×21 hand landmarks × (x, y, z, conf) = **300-d** per frame; **+ motion** `x_t − x_{t−1}` → **600-d** |
| Normalisation | per-example z-score, pad/truncate to **500 frames** |
| Pose encoder | Linear(pose_dim→256) → ReLU → Dropout(0.1) → Linear(256→d_model) |
| Language model | pretrained **T5** (`inputs_embeds` = pose embeddings), fully fine-tuned |
| Optimiser | AdamW, lr **5e-4 constant**, batch **16**, grad-accum **2**, clip **1.0**, **fp16** |
| Epochs | **10** (t5-small, t5-base, t5-small-motion), **3** (t5-large) |
| Model selection | lowest validation loss, checked every 500 steps |
| Decoding | beam **4**, length penalty **2.0**, max **128** tokens |
| Test split | 10% held out, `train_test_split(random_state=42)` → **1,887** examples |

Paper results on that split (BLEU on a 0–100 scale): t5-small **0.189**, t5-base 0.154, t5-large 0.132,
**t5-small + Motion 0.298** (best).

### Settings
* **Add Input** → your dataset **`isign-partad-features`** (from notebook 1).
* **Accelerator: GPU T4 x2** (use T4, not P100: current PyTorch builds may not support the
  P100). Only one GPU is used, like the paper's single-GPU code.
* **Internet: On** (downloads pretrained T5 from HuggingFace).
* **Save Version → Save & Run All (Commit)** so it keeps running after you close the tab.
  t5-small-motion ≈ 45–90 min.
"""),
    code('!pip install -q "sacrebleu==2.6.0" "rouge-score==0.1.2" sentencepiece'),
    md("### Source files (identical to the locally tested versions)"),
    writefile("paper_pose_utils.py"),
    writefile("paper_model.py"),
    writefile("features.py"),
    writefile("pipeline.py"),
    md("### Configuration"),
    code("""
import glob, json, logging, os, sys
import pandas as pd, torch
sys.path.insert(0, ".")
import pipeline
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)

VARIANT = "t5-small-motion"   # "t5-small" | "t5-small-motion" | "t5-base" | "t5-large"
SPLIT = "paper"               # "paper" = exact paper split | "video" = stricter, no shared videos
MIXED_PRECISION = "fp16"      # paper setting; switch to "no" if the log warns about NaN loss
RUN_RELEASED_BASELINE = True  # also score the authors' released checkpoint on the same split

csv_path = glob.glob("/kaggle/input/**/iSign_v1.1.csv", recursive=True)[0]
FEATURES = os.path.dirname(csv_path)
WORK = "/kaggle/working"
print("features:", FEATURES, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
"""),
    md("### Step 1 · data and splits"),
    code("""
df = pipeline.load_annotations(csv_path, FEATURES)
split_fn = pipeline.paper_test_split if SPLIT == "paper" else pipeline.video_test_split
train_df, test_df = split_fn(df)
print(f"usable samples {len(df):,} -> train {len(train_df):,} / test {len(test_df):,}  "
      "(paper: 18,867 -> 16,980 / 1,887)")
test_df.to_csv(f"{WORK}/test_split_{SPLIT}.csv", index=False)
test_df.head()
"""),
    md("""
### Step 2 (optional) · score the authors' released checkpoint on this test split
If our data and split match the paper's, this should land near their **0.298 BLEU** for
t5-small-motion. It is a check that the reproduction is set up correctly.
"""),
    code("""
if RUN_RELEASED_BASELINE:
    from huggingface_hub import snapshot_download
    rel = snapshot_download("manavdhamecha77/iSign-t5-pose-to-text",
                            allow_patterns=[f"{VARIANT}/pose_encoder.pt", f"{VARIANT}/t5/*"])
    released = pipeline.evaluate(VARIANT, os.path.join(rel, VARIANT), test_df, FEATURES,
                                 f"{WORK}/eval_released_{VARIANT}")
    print(json.dumps(released, indent=2))
"""),
    md("### Step 3 · train (exact paper recipe)"),
    code("""
best = pipeline.train(VARIANT, train_df, FEATURES, f"{WORK}/checkpoints/{VARIANT}",
                      cfg={"mixed_precision": MIXED_PRECISION})
print("best model:", best)
"""),
    md("### Step 4 · evaluate on the held-out test split"),
    code("""
metrics = pipeline.evaluate(VARIANT, best, test_df, FEATURES, f"{WORK}/eval_{VARIANT}")
print(json.dumps(metrics, indent=2))
pred = pd.read_csv(f"{WORK}/eval_{VARIANT}/predictions.csv")
pd.set_option("display.max_colwidth", 80)
pred.sample(20, random_state=0)
"""),
    md("### Step 5 · results table (paper vs. ours)"),
    code("""
paper = {"t5-small": 0.189, "t5-base": 0.154, "t5-large": 0.132, "t5-small-motion": 0.298}
rows = [{"model": f"{VARIANT} (paper, Table 1)", "BLEU": paper[VARIANT]},
        {"model": f"{VARIANT} (ours)", **{k: round(metrics[k], 3) for k in
                                          ("BLEU", "chrF", "ROUGE-1", "ROUGE-2", "ROUGE-L")}}]
if RUN_RELEASED_BASELINE:
    rows.insert(1, {"model": f"{VARIANT} (released ckpt, our split)",
                    **{k: round(released[k], 3) for k in ("BLEU", "chrF", "ROUGE-1", "ROUGE-2", "ROUGE-L")}})
table = pd.DataFrame(rows)
table.to_csv(f"{WORK}/results_{VARIANT}.csv", index=False)
table
"""),
    md("""
### Getting the model onto your laptop
Everything is in the version's **Output** tab: `checkpoints/<variant>/best_model/`
(`pose_encoder.pt` + `t5/`), predictions and metrics. Download `best_model`, or save the output as
a dataset. It loads with `PoseT5(...).load_pretrained(best_model)`, the same format as the
paper's HuggingFace checkpoints.
"""),
]

out1, out2 = HERE / "isign_01_download_partAD.ipynb", HERE / "isign_02_train_eval.ipynb"
json.dump(notebook(nb1), open(out1, "w", encoding="utf-8"), indent=1)
json.dump(notebook(nb2), open(out2, "w", encoding="utf-8"), indent=1)
print("wrote", out1.name, "and", out2.name)
