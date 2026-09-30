"""Training + evaluation, reproducing arXiv 2609.12993 (Sign-Lang-Trans repo).

Faithful to the paper / their src/train.py, src/dataset.py, src/split_data.py,
src/evaluate_all.py:
  * test split  : sklearn train_test_split(test_size=0.1, random_state=42)
                  over the samples that have poses (their split_data.py)
  * val split   : torch random_split 80/10/10 of the remaining rows,
                  Generator seed 42 (their create_data_loaders); the extra 10%
                  "internal test" is unused, exactly as in their code
  * optimizer   : AdamW lr 5e-4, CONSTANT (the paper states warmup was not applied)
  * batch 16, gradient accumulation 2, grad-clip 1.0, fp16 autocast + GradScaler
  * epochs      : 10 (t5-small, t5-base, t5-small-motion), 3 (t5-large)
  * eval on val every 500 micro-batches; best_model = lowest val loss
  * decoding    : beam 4, length_penalty 2.0, max 128 tokens
  * metrics     : sacrebleu BLEU + chrF (corpus), ROUGE-1/2/L (stemmed, averaged)

Deliberate deviations (none change the model or the optimisation):
  * features are read from cached .npy files instead of re-parsing .pose files
    each epoch (numerically identical, see features.py);
  * only best_model and the final model are saved, not every 500 steps
    (their per-step checkpoints would overflow Kaggle's 20 GB output).
"""

import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from features import preprocess_array
from paper_model import PoseT5

logger = logging.getLogger("isign")

# Per-variant settings from the paper (section 4) and their configs/config_*.yaml.
VARIANTS = {
    "t5-small":        dict(base_model="t5-small", pose_dim=300, use_motion=False, epochs=10),
    "t5-small-motion": dict(base_model="t5-small", pose_dim=600, use_motion=True,  epochs=10),
    "t5-base":         dict(base_model="t5-base",  pose_dim=300, use_motion=False, epochs=10),
    "t5-large":        dict(base_model="t5-large", pose_dim=300, use_motion=False, epochs=3),
}

TRAIN_DEFAULTS = dict(
    batch_size=16, learning_rate=5e-4, gradient_accumulation_steps=2, max_grad_norm=1.0,
    mixed_precision="fp16", logging_steps=100, eval_steps=500, num_workers=4, seed=42,
    max_pose_length=500, max_text_length=128, normalization="zscore",
    beam_size=4, max_gen_length=128, pin_memory=True,
)


# ----------------------------------------------------------------------------- data

def load_annotations(csv_path, features_dir):
    """iSign CSV rows (in file order) that have a cached feature file.
    Mirrors their loader: keeps (uid, text), drops rows without a pose."""
    df = pd.read_csv(csv_path)
    if not {"uid", "text"}.issubset(df.columns):
        raise ValueError(f"expected uid,text columns in {csv_path}, got {list(df.columns)}")
    df = df[["uid", "text"]].copy()
    df["uid"] = df["uid"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    features_dir = Path(features_dir)
    have = df["uid"].map(lambda u: (features_dir / f"{u}.npy").exists())
    return df[have].reset_index(drop=True)


def paper_test_split(df, seed=42, test_fraction=0.1):
    """Their split_data.py: sklearn train_test_split(test_size=0.1, random_state=42)."""
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_fraction, random_state=seed,
                                         shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def video_test_split(df, seed=42, test_fraction=0.1):
    """Stricter alternative: no video contributes segments to both train and test
    (what the iSign authors recommend). uid = '<video_id>-<segment>'."""
    vids = df["uid"].str.rsplit("-", n=1).str[0]
    unique = vids.drop_duplicates().sample(frac=1.0, random_state=seed).tolist()
    test_vids, n = set(), 0
    for v in unique:
        if n >= test_fraction * len(df):
            break
        test_vids.add(v)
        n += int((vids == v).sum())
    is_test = vids.isin(test_vids)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


class PoseSLTDataset(Dataset):
    """Same outputs as their PoseSLTDataset, reading cached features."""

    def __init__(self, df, features_dir, tokenizer, pose_dim, use_motion, cfg):
        self.df, self.dir, self.tok = df.reset_index(drop=True), Path(features_dir), tokenizer
        self.pose_dim, self.use_motion, self.cfg = pose_dim, use_motion, cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = np.load(self.dir / f"{row.uid}.npy")
        pose, pose_length = preprocess_array(
            raw, max_length=self.cfg["max_pose_length"],
            normalization=self.cfg["normalization"], pose_dim=self.pose_dim,
            use_motion=self.use_motion)
        enc = self.tok(row.text, max_length=self.cfg["max_text_length"], padding="max_length",
                       truncation=True, return_tensors="pt")
        labels = enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {"pose": torch.from_numpy(pose).float(), "pose_length": torch.tensor(pose_length),
                "labels": labels, "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def make_loaders(train_df, features_dir, tokenizer, variant, cfg):
    """Their create_data_loaders: random_split 80/10/10 with Generator(seed)."""
    v = VARIANTS[variant]
    ds = PoseSLTDataset(train_df, features_dir, tokenizer, v["pose_dim"], v["use_motion"], cfg)
    total = len(ds)
    test_size = max(1, int(total * 0.1))
    val_size = max(1, int(total * 0.1))
    train_size = total - test_size - val_size
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_set, val_set, _unused = random_split(ds, [train_size, val_size, test_size],
                                               generator=gen)
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              collate_fn=collate_fn,
              pin_memory=cfg.get("pin_memory", True) and torch.cuda.is_available(),
              persistent_workers=cfg["num_workers"] > 0)
    return {"train": DataLoader(train_set, shuffle=True, **kw),
            "val": DataLoader(val_set, shuffle=False, **kw)}


# ------------------------------------------------------------------------- training

def train(variant, train_df, features_dir, out_dir, cfg=None, init_from=None,
          max_steps=None):
    """Train one variant. Returns the path of best_model.
    init_from: optional checkpoint dir (e.g. the paper's released weights) to
    start from instead of the plain pretrained T5."""
    cfg = {**TRAIN_DEFAULTS, **(cfg or {})}
    v = VARIANTS[variant]
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(v["base_model"])
    loaders = make_loaders(train_df, features_dir, tokenizer, variant, cfg)
    model = PoseT5(v["base_model"], pose_dim=v["pose_dim"]).to(device)
    if init_from:
        model.load_pretrained(str(init_from))
        model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]))

    amp = cfg["mixed_precision"]
    amp_on = device.type == "cuda" and amp in {"fp16", "bf16"}
    amp_dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and amp == "fp16")
    accum = max(1, cfg["gradient_accumulation_steps"])

    logger.info("variant=%s device=%s train=%d val=%d epochs=%d", variant, device,
                len(loaders["train"].dataset), len(loaders["val"].dataset), v["epochs"])

    def evaluate_loss():
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in loaders["val"]:
                out = model(pose=b["pose"].to(device), pose_length=b["pose_length"].to(device),
                            input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["labels"].to(device))
                total += out.loss.item()
        model.train()
        return total / len(loaders["val"])

    global_step, best, bad_loss_streak = 0, float("inf"), 0
    history = []
    model.train()
    t0 = time.time()
    for epoch in range(v["epochs"]):
        running = 0.0
        n_batches = len(loaders["train"])
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, b in enumerate(loaders["train"]):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                out = model(pose=b["pose"].to(device), pose_length=b["pose_length"].to(device),
                            input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["labels"].to(device))
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accum == 0 or batch_idx + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * accum
            bad_loss_streak = bad_loss_streak + 1 if not math.isfinite(step_loss) else 0
            if bad_loss_streak == 50:
                logger.warning("loss has been NaN/inf for 50 steps: fp16 overflow. "
                               "Re-run with cfg={'mixed_precision': 'no'}.")
            running += step_loss if math.isfinite(step_loss) else 0.0
            global_step += 1

            if global_step % cfg["logging_steps"] == 0:
                logger.info("epoch %d step %d  train_loss %.4f  (%.1f min)", epoch + 1,
                            global_step, running / (batch_idx + 1), (time.time() - t0) / 60)
            if global_step % cfg["eval_steps"] == 0:
                val = evaluate_loss()
                history.append({"step": global_step, "epoch": epoch + 1, "val_loss": val})
                if val < best:
                    best = val
                    model.save_pretrained(str(out_dir / "best_model"))
                    logger.info("step %d  val_loss %.4f  -> new best_model", global_step, val)
                else:
                    logger.info("step %d  val_loss %.4f", global_step, val)
            if max_steps and global_step >= max_steps:
                break
        logger.info("epoch %d done, train_loss %.4f", epoch + 1, running / max(1, batch_idx + 1))
        if max_steps and global_step >= max_steps:
            break

    # Also evaluate at the very end so short runs still produce a best_model.
    val = evaluate_loss()
    history.append({"step": global_step, "epoch": "end", "val_loss": val})
    if val < best or not (out_dir / "best_model").exists():
        model.save_pretrained(str(out_dir / "best_model"))
    model.save_pretrained(str(out_dir / "final_model"))
    json.dump(history, open(out_dir / "val_history.json", "w"), indent=1)
    return out_dir / "best_model"


# ----------------------------------------------------------------------- evaluation

def evaluate(variant, checkpoint, test_df, features_dir, out_dir, cfg=None, limit=None):
    """Their evaluate_all.evaluate_model: BLEU, chrF, ROUGE-1/2/L + predictions.csv."""
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU, CHRF

    cfg = {**TRAIN_DEFAULTS, **(cfg or {})}
    v = VARIANTS[variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(v["base_model"])
    df = test_df if limit is None else test_df.head(limit)
    ds = PoseSLTDataset(df, features_dir, tokenizer, v["pose_dim"], v["use_motion"], cfg)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], collate_fn=collate_fn,
                        pin_memory=cfg.get("pin_memory", True) and torch.cuda.is_available())

    model = PoseT5(v["base_model"], pose_dim=v["pose_dim"],
                   local_model_path=str(Path(checkpoint) / "t5")).to(device)
    model.load_pretrained(str(checkpoint))
    model.to(device).eval()

    preds = []
    with torch.no_grad():
        for b in loader:
            ids = model.generate(pose=b["pose"].to(device), pose_length=b["pose_length"].to(device),
                                 max_length=cfg["max_gen_length"], num_beams=cfg["beam_size"])
            preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    preds = [p.strip() for p in preds]
    refs = [r.strip() for r in df["text"].tolist()]

    bleu = BLEU().corpus_score(preds, [refs]).score
    chrf = CHRF().corpus_score(preds, [refs]).score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
    rouge = {k: sum(s[k].fmeasure for s in scores) / len(scores)
             for k in ("rouge1", "rouge2", "rougeL")}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "reference", "prediction"])
        w.writeheader()
        w.writerows({"uid": u, "reference": r, "prediction": p}
                    for u, r, p in zip(df["uid"], refs, preds))
    metrics = {"variant": variant, "checkpoint": str(checkpoint), "test_examples": len(refs),
               "BLEU": bleu, "chrF": chrf, "ROUGE-1": rouge["rouge1"] * 100,
               "ROUGE-2": rouge["rouge2"] * 100, "ROUGE-L": rouge["rougeL"] * 100,
               "unique_predictions": len(set(preds))}
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    return metrics
