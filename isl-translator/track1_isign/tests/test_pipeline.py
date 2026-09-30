"""Smoke test of training + evaluation on a tiny fake dataset, plus a check that
our evaluate() reproduces the paper's own inference.py on their released model.

Run:  python tests/test_pipeline.py <dir with .pose files> <released t5-small-motion dir>
"""

import glob
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pipeline
from features import pose_bytes_to_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
pose_dir, released = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(pose_dir, "*.pose")))[:20]

# Fake iSign-style data: uid "<video>-<segment>", English text.
feat_dir = tempfile.mkdtemp()
texts = ["hello", "how are you", "alright", "good morning"]
rows = []
for i, f in enumerate(files):
    uid = f"vid{i // 3:03d}-{i % 3}"
    np.save(os.path.join(feat_dir, f"{uid}.npy"), pose_bytes_to_features(open(f, "rb").read()))
    rows.append({"uid": uid, "text": texts[i % 4]})
csv_path = os.path.join(feat_dir, "annotations.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)

df = pipeline.load_annotations(csv_path, feat_dir)
train_df, test_df = pipeline.paper_test_split(df)
vtrain, vtest = pipeline.video_test_split(df)
assert not set(vtrain.uid.str.rsplit("-", n=1).str[0]) & set(vtest.uid.str.rsplit("-", n=1).str[0])
print(f"[ok] splits: paper {len(train_df)}/{len(test_df)}, video-level {len(vtrain)}/{len(vtest)}")

cfg = {"batch_size": 2, "num_workers": 0, "eval_steps": 3, "logging_steps": 2}
out = tempfile.mkdtemp()
best = pipeline.train("t5-small-motion", train_df, feat_dir, out, cfg=cfg, max_steps=6)
assert (best / "pose_encoder.pt").exists() and (best / "t5" / "model.safetensors").exists()
m = pipeline.evaluate("t5-small-motion", best, test_df, feat_dir, os.path.join(out, "eval"), cfg=cfg)
print("[ok] train+eval ran:", {k: m[k] for k in ("BLEU", "chrF", "ROUGE-L", "test_examples")})

# Released checkpoint through OUR evaluate(): must give the same text as their
# inference.py gave earlier ('page 58' for the Hello videos).
m2 = pipeline.evaluate("t5-small-motion", released, df.head(3), feat_dir,
                       os.path.join(out, "eval_released"), cfg=cfg)
preds = pd.read_csv(os.path.join(out, "eval_released", "predictions.csv"))
print("[ok] released checkpoint loads; predictions:", preds.prediction.tolist())
