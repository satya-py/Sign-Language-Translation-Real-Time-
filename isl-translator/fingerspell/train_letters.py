"""Train the ISL fingerspelling classifier (A-Z, 0-10) on single frames.

    python train_letters.py                 # trains and writes models/letters.pt
    python train_letters.py --epochs 300

Frames where MediaPipe found no hand are dropped: they carry no letter, and
training on them teaches "empty frame = some letter", which then fires constantly
in the live app.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "live"))
import letterfeat  # noqa: E402

DATA = HERE / "data" / "letters"
MODEL_DIR = HERE / "models"


class LetterNet(nn.Module):
    """Small MLP: a letter is a pose, not a sequence, so no time dimension."""

    def __init__(self, in_dim, n_classes, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(width // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset():
    samples, skipped = [], 0
    for folder in sorted(DATA.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.npz")):
            data = np.load(path, allow_pickle=True)
            row = np.asarray(data["keypoints"], dtype=np.float32)
            if not letterfeat.has_hand(row):
                skipped += 1
                continue
            samples.append({"features": letterfeat.row_to_features(row),
                            "label": str(data["label"]),
                            "signer": str(data.get("signer", "unknown")),
                            "name": path.name})
    return samples, skipped


def split_by_time(samples, val_fraction=0.2):
    """Hold out the LAST frames recorded for each letter.

    The frames of one letter come from a single hold in front of the camera, a few
    seconds apart, so two random frames of the same hold are nearly the same
    picture. Splitting them at random therefore scores the model on what it already
    memorised - it read 96.7% that way, which said nothing about live use. Holding
    out the end of each recording is not perfect either (same session, same light),
    but it is the honest number available without recording again.
    """
    by_label = {}
    for sample in samples:
        by_label.setdefault(sample["label"], []).append(sample)
    train, val = [], []
    for items in by_label.values():
        items = sorted(items, key=lambda s: s["name"])      # names carry the timestamp
        n_val = max(1, int(len(items) * val_fraction))
        val += items[-n_val:]
        train += items[:-n_val]
    return train, val


def split(samples, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s["label"], []).append(s)
    train, val = [], []
    for items in by_label.values():
        order = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_fraction))
        val += [items[i] for i in order[:n_val]]
        train += [items[i] for i in order[n_val:]]
    return train, val


class LetterDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels, train, seed=0):
        self.samples, self.train = samples, train
        self.index = {l: i for i, l in enumerate(labels)}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        f = self.samples[i]["features"]
        if self.train:
            f = letterfeat.augment(f, self.rng)
        return torch.from_numpy(np.asarray(f, dtype=np.float32)), \
            self.index[self.samples[i]["label"]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--out", default=str(MODEL_DIR / "letters.pt"))
    p.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    p.add_argument("--split", default="time", choices=["time", "random"],
                   help="'time' holds out the end of each recording (honest); "
                        "'random' mixes frames of the same hold and flatters the score")
    args = p.parse_args(argv)

    samples, skipped = load_dataset()
    if not samples:
        sys.exit(f"no usable samples under {DATA}")
    counts = Counter(s["label"] for s in samples)
    labels = sorted(counts, key=lambda s: (s.isdigit(), s))
    print(f"{len(samples)} usable frames, {skipped} skipped (no hand), {len(labels)} letters")
    print("per letter:", dict(counts))

    train_s, val_s = (split_by_time(samples) if args.split == "time"
                      else split(samples))
    train_dl = torch.utils.data.DataLoader(LetterDataset(train_s, labels, True),
                                           batch_size=args.batch_size, shuffle=True,
                                           drop_last=len(train_s) > args.batch_size)
    val_dl = torch.utils.data.DataLoader(LetterDataset(val_s, labels, False),
                                         batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = LetterNet(letterfeat.FEATURE_DIM, len(labels)).to(device)
    print(f"device {device}, {sum(p.numel() for p in model.parameters())/1e3:.0f}K parameters, "
          f"train {len(train_s)} / val {len(val_s)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * max(1, len(train_dl)))
    best_acc, best_state, t0 = 0.0, None, time.time()
    confusion = np.zeros((len(labels), len(labels)), dtype=int)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        model.eval()
        correct = seen = 0
        epoch_confusion = np.zeros_like(confusion)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                seen += len(y)
                for t, pr in zip(y.tolist(), pred.tolist()):
                    epoch_confusion[t, pr] += 1
        acc = correct / max(1, seen)
        if acc >= best_acc:
            best_acc, confusion = acc, epoch_confusion
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  loss {total/max(1,len(train_dl)):.3f}  "
                  f"val {acc*100:.1f}%  (best {best_acc*100:.1f}%)  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "labels": labels,
                "feature_dim": letterfeat.FEATURE_DIM, "val_accuracy": best_acc,
                "samples": len(samples)}, out)
    print(f"\nbest validation accuracy {best_acc*100:.1f}%  ->  {out}")

    print("\nper letter (val):")
    for i, label in enumerate(labels):
        n = confusion[i].sum()
        if not n:
            continue
        others = confusion[i].copy()
        others[i] = -1
        worst = int(np.argmax(others))
        note = f"   often read as {labels[worst]}" if confusion[i, worst] > 0 else ""
        print(f"  {label:<4} {confusion[i, i]}/{n}{note}")
    json.dump({"labels": labels, "val_accuracy": best_acc,
               "confusion": confusion.tolist()},
              open(out.with_suffix(".report.json"), "w"), indent=1)
    return {"labels": labels, "val_accuracy": best_acc, "samples": len(samples)}


if __name__ == "__main__":
    main()
