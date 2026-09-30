"""Train the live-demo sign classifier on keypoints (INCLUDE + your own takes).

Why this works on a webcam when the INCLUDE pretrained model does not:
  * features are body-relative (signfeat.py), not raw frame pixels;
  * heavy augmentation (scale, shift, rotate, speed, mirror, dropped landmarks);
  * an "idle" class, mined from the still moments of every clip, so the model can
    answer "no sign right now" instead of always guessing a word.

    python train.py                       # trains on data/keypoints (+ recordings/)
    python train.py --epochs 60 --out models/signs.pt
"""

import argparse
import json
import shutil
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
import signfeat  # noqa: E402

KEYPOINT_DIR = HERE / "data" / "keypoints"
RECORDING_DIR = HERE / "recordings"
MODEL_DIR = HERE / "models"
IDLE = "idle"


# --------------------------------------------------------------------------- data

def load_clips(dirs):
    """Every .npz under the given folders -> (raw rows, label, group)."""
    clips = []
    for root in dirs:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.npz")):
            if path.parent == root:          # summary files etc.
                continue
            data = np.load(path, allow_pickle=True)
            rows = data["keypoints"].astype(np.float32)
            if len(rows) < 8:
                continue
            label = str(data["label"]) if "label" in data else path.parent.name
            signer = str(data["signer"]) if "signer" in data else "unknown"
            clips.append({"rows": rows, "label": label, "signer": signer,
                          "name": path.name, "full": rows})
    return clips


def crop_to_sign(clips):
    """Keep only the moving part of each clip, the same cut the live app makes.

    Dataset clips start and end with the signer standing still; the live app never
    sends those frames, so training on them teaches the wrong thing.
    """
    out = []
    for c in clips:
        if c["label"] == IDLE:
            out.append(c)
            continue
        lo, hi = signfeat.active_span(c["rows"])
        if hi - lo >= 10:
            c = {**c, "rows": c["rows"][lo:hi]}
        out.append(c)
    return out


def mine_idle(clips, per_clip=3, max_total=None):
    """Cut 'hands at rest' windows out of real clips to build the idle class.

    Signers stand still before and after the sign, which is exactly what the camera
    sees between signs in the live demo.
    """
    idle = []
    for c in clips:
        if c["label"] == IDLE:
            continue
        speed = signfeat.hand_speed(c["full"])
        n = len(speed)
        window = max(10, n // 4)
        best = []
        for start in range(0, max(1, n - window + 1), max(3, window // 3)):
            best.append((float(speed[start:start + window].mean()), start))
        best.sort()
        for mean_speed, start in best[:per_clip]:
            if mean_speed < 0.02:                      # genuinely still
                idle.append({"rows": c["full"][start:start + window], "label": IDLE,
                             "signer": c["signer"], "name": f"{c['name']}:idle@{start}"})
    if max_total and len(idle) > max_total:
        rng = np.random.default_rng(0)
        idle = [idle[i] for i in rng.choice(len(idle), max_total, replace=False)]
    return idle


def mine_transitions(clips, max_total=None):
    """Windows where the hands MOVE but no sign is being made.

    Between two signs the hand drops and rises again. Without examples of that, the
    classifier is forced to call the movement a word, which showed up in testing as
    confident nonsense ("friend 93%") between real signs.
    """
    out = []
    margin = 8            # never touch the sign itself
    for c in clips:
        if c["label"] == IDLE:
            continue
        full = c["full"]
        lo, hi = signfeat.active_span(full)
        if hi - lo >= len(full) - 4:      # span detection failed: skip this clip
            continue
        speed = signfeat.hand_speed(full)
        for region_start, region_end in ((0, max(0, lo - margin)),
                                         (min(len(full), hi + margin), len(full))):
            if region_end - region_start < 12:
                continue
            # Slide instead of taking the region whole: the live spotter cuts at a
            # speed dip, so it sees short pieces of the hand rising and dropping,
            # not the tidy whole gap. One window per gap left most of that motion
            # unseen, and 22% of it came back as a confident wrong sign.
            span = region_end - region_start
            window = max(12, min(span, 24))
            for start in range(region_start, region_end - window + 1, max(4, window // 2)):
                mean_speed = float(speed[start:start + window].mean())
                # Anything from a still hand to a fast drop, but short of a sign.
                if 0.015 < mean_speed < 0.25:
                    piece = full[start:start + window]
                    out.append({"rows": piece, "label": IDLE, "signer": c["signer"],
                                "name": f"{c['name']}:transition@{start}", "full": piece})
    if max_total and len(out) > max_total:
        rng = np.random.default_rng(1)
        out = [out[i] for i in rng.choice(len(out), max_total, replace=False)]
    return out


def augment(points, flags, rng):
    """Geometric + temporal + dropout augmentation on body-relative points."""
    p = points.copy()
    f = flags.copy()

    angle = rng.uniform(-0.26, 0.26)                   # +-15 degrees
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    p = p @ rot.T
    p *= rng.uniform(0.85, 1.15)                       # distance from the camera
    p += rng.uniform(-0.12, 0.12, size=(1, 1, 2)).astype(np.float32)   # position in frame

    if rng.random() < 0.5:                             # left-handed signer
        p[:, :, 0] *= -1
        n_pose = len(signfeat.POSE_KEEP)
        left = p[:, n_pose:n_pose + signfeat.N_HAND].copy()
        p[:, n_pose:n_pose + signfeat.N_HAND] = p[:, n_pose + signfeat.N_HAND:]
        p[:, n_pose + signfeat.N_HAND:] = left
        f[:, [1, 2]] = f[:, [2, 1]]

    if rng.random() < 0.3:                             # MediaPipe loses a hand
        which = rng.integers(1, 3)
        n_pose = len(signfeat.POSE_KEEP)
        sl = slice(n_pose, n_pose + signfeat.N_HAND) if which == 1 \
            else slice(n_pose + signfeat.N_HAND, None)
        drop = rng.random(len(p)) < rng.uniform(0.2, 0.9)
        p[drop, sl] = 0.0
        f[drop, which] = 0.0

    p += rng.normal(0, 0.012, size=p.shape).astype(np.float32)  # landmark jitter
    return p.astype(np.float32), f.astype(np.float32)


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips, labels, train, seed=0):
        self.clips, self.labels, self.train = clips, labels, train
        self.index = {l: i for i, l in enumerate(labels)}
        self.rng = np.random.default_rng(seed)
        # Normalising is the slow part, so do it once per clip up front.
        self.cache = [signfeat.normalize(c["rows"]) for c in clips]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        points, flags = self.cache[i]
        if self.train:
            speed = self.rng.uniform(0.75, 1.35)       # signing faster or slower
            length = max(10, int(signfeat.SEQ_LEN * speed))
            points, flags = signfeat.resample(points, flags, length)
            points, flags = augment(points, flags, self.rng)
        points, flags = signfeat.resample(points, flags, signfeat.SEQ_LEN)
        x = signfeat.to_features(points, flags)
        return torch.from_numpy(x), self.index[self.clips[i]["label"]]


# -------------------------------------------------------------------------- model

class Block(nn.Module):
    """Depthwise 1D conv over time + pointwise mix, residual."""

    def __init__(self, dim, kernel=5):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, dim * 2, 1)
        self.pw2 = nn.Conv1d(dim * 2, dim, 1)

    def forward(self, x):
        h = self.norm(self.dw(x))
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + h


class SignNet(nn.Module):
    """Small 1D-CNN + one transformer layer over time (about 0.6M parameters)."""

    def __init__(self, in_dim, n_classes, dim=192):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList([Block(dim) for _ in range(3)])
        self.attn = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2,
                                               dropout=0.2, batch_first=True,
                                               norm_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Dropout(0.4),
                                  nn.Linear(dim * 2, n_classes))

    def forward(self, x):                              # x: (B, T, F)
        h = self.stem(x).transpose(1, 2)               # (B, dim, T)
        for block in self.blocks:
            h = block(h)
        h = self.attn(h.transpose(1, 2))               # (B, T, dim)
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(pooled)


# ----------------------------------------------------------------------- training

def split_clips(clips, val_fraction=0.2, seed=42):
    """Hold out whole clips per label (never two takes of the same video split)."""
    rng = np.random.default_rng(seed)
    by_label = {}
    for c in clips:
        by_label.setdefault(c["label"], []).append(c)
    train, val = [], []
    for label, items in by_label.items():
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_fraction))
        val += [items[i] for i in idx[:n_val]]
        train += [items[i] for i in idx[n_val:]]
    return train, val


def main(argv=None):
    """Train. Pass argv to call this from another program instead of the shell:
    a second Python process would load torch twice, which a 8 GB machine cannot do."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keypoints", default=str(KEYPOINT_DIR))
    p.add_argument("--recordings", default=str(RECORDING_DIR))
    p.add_argument("--out", default=str(MODEL_DIR / "signs.pt"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--idle-still", type=float, default=8,
                   help="still windows to mine, as a multiple of the median class size")
    p.add_argument("--idle-move", type=float, default=30,
                   help="hand-transition windows to mine, same multiple. These are "
                        "what stops movement between signs becoming a wrong word.")
    p.add_argument("--webcam-repeat", type=int, default=3,
                   help="how many times each webcam take counts against studio clips")
    p.add_argument("--max-weight", type=float, default=2.5,
                   help="cap on the class weight: a sign with 5 takes must not "
                        "outshout one with 20")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-idle", action="store_true", help="skip the mined idle class")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="cpu is safer while the web server holds the GPU (4 GB cards)")
    args = p.parse_args(argv)

    clips = load_clips([args.keypoints, args.recordings])
    if not clips:
        sys.exit("No keypoint files found. Run extract_keypoints.py first.")
    before = np.mean([len(c["rows"]) for c in clips])
    clips = crop_to_sign(clips)
    print(f"cropped clips to the moving part: {before:.0f} -> "
          f"{np.mean([len(c['rows']) for c in clips]):.0f} frames on average")
    if not args.no_idle and not any(c["label"] == IDLE for c in clips):
        # Scale the idle set with the whole dataset, not with the median class.
        # Adding two dozen classes of ten clips each halved that median, which
        # quietly halved the idle data too - and the idle class immediately fell
        # from 91% to 67%, letting movement between signs through as words again.
        per_class = max(8, len(clips) // 100)
        # Transitions are the main source of false signs between words, so use all
        # of them; the class weights below stop "idle" from dominating.
        still = mine_idle(clips, max_total=int(per_class * args.idle_still))
        moving = mine_transitions(clips, max_total=int(per_class * args.idle_move))
        print(f"mined {len(still)} still windows and {len(moving)} hand-transition "
              "windows for the idle class")
        clips += still + moving

    counts = Counter(c["label"] for c in clips)
    labels = sorted(counts)
    print(f"{len(clips)} clips, {len(labels)} classes")
    print(dict(counts))

    train_clips, val_clips = split_clips(clips)
    # Clips recorded from a webcam are the only ones in the domain the demo runs in,
    # and there are five of them against twenty studio clips per sign. Repeating
    # them keeps a newly taught sign from pulling the older ones off balance.
    webcam = [c for c in train_clips if c["name"].startswith("web_")]
    train_clips = train_clips + webcam * (args.webcam_repeat - 1)
    print(f"{len(webcam)} webcam takes counted {args.webcam_repeat}x")
    train_ds = ClipDataset(train_clips, labels, train=True)
    val_ds = ClipDataset(val_clips, labels, train=False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    use_cuda = torch.cuda.is_available() and args.device in ("auto", "cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SignNet(signfeat.FEATURE_DIM, len(labels)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device}, {n_params/1e6:.2f}M parameters, "
          f"train {len(train_ds)} / val {len(val_ds)}")

    # The idle class has more samples than any single word; weight it down so the
    # model does not learn to answer "idle" whenever it is unsure.
    train_counts = Counter(c["label"] for c in train_clips)
    raw = [len(train_clips) / (len(labels) * max(1, train_counts[l])) for l in labels]
    # Uncapped, a class with 5 takes gets six times the pull of one with 20, so each
    # newly taught sign quietly outvotes everything taught before it - which is what
    # "the old signs stopped working after I taught a new one" actually was.
    weights = torch.tensor([min(args.max_weight, max(0.4, w)) for w in raw],
                           dtype=torch.float32, device=device)
    print("class weights:", {l: round(float(w), 2) for l, w in zip(labels, weights)})

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * max(1, len(train_dl)))
    best_acc, best_state = 0.0, None
    t0 = time.time()

    best_confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, weight=weights, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        confusion = np.zeros((len(labels), len(labels)), dtype=int)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
                for t, pr in zip(y.tolist(), pred.tolist()):
                    confusion[t, pr] += 1
        acc = correct / max(1, total)
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_confusion = confusion.copy()   # report the model we keep, not epoch 60
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  loss {total_loss/max(1,len(train_dl)):.3f}  "
                  f"val acc {acc*100:.1f}%  (best {best_acc*100:.1f}%)  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    confusion = best_confusion
    per_class = {l: (int(confusion[i, i]), int(confusion[i].sum()))
                 for i, l in enumerate(labels) if confusion[i].sum()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Compare with the model this one replaces before overwriting it, and keep a copy.
    # A retrain that quietly breaks signs that used to work is the failure to catch
    # here, not a lower average.
    previous, regressed = None, []
    if out.exists():
        try:
            previous = torch.load(out, map_location="cpu", weights_only=False)
        except Exception:
            previous = None
        shutil.copy2(out, out.with_name(out.stem + "_previous.pt"))
    if previous and previous.get("per_class"):
        for label, (was_ok, was_n) in previous["per_class"].items():
            now = per_class.get(label)
            if now is None:
                regressed.append(f"{label}: gone from the model")
            elif was_n and now[1] and now[0] / now[1] < was_ok / was_n - 0.15:
                regressed.append(f"{label}: {was_ok}/{was_n} -> {now[0]}/{now[1]}")

    torch.save({"state_dict": best_state, "labels": labels, "per_class": per_class,
                "feature_dim": signfeat.FEATURE_DIM, "seq_len": signfeat.SEQ_LEN,
                "val_accuracy": best_acc, "n_clips": len(clips)}, out)
    print(f"\nbest validation accuracy {best_acc*100:.1f}%  ->  {out}")

    if previous is not None:
        was = previous.get("val_accuracy")
        if was:
            print(f"previous model: {was*100:.1f}% on {len(previous['labels'])-1} signs"
                  f"  ->  now {best_acc*100:.1f}% on {len(labels)-1}")
    if regressed:
        print("SIGNS THAT GOT WORSE (models/signs_previous.pt is the one to fall back on):")
        for line in regressed:
            print("  " + line)
    elif previous is not None:
        print("no sign got worse than in the previous model")

    # Per-class report from the best epoch's confusion matrix.
    print("\nper class (val):")
    for i, label in enumerate(labels):
        n = confusion[i].sum()
        if n:
            worst = int(np.argmax(np.where(np.arange(len(labels)) == i, -1, confusion[i])))
            print(f"  {label:<14} {confusion[i, i]}/{n}"
                  + (f"   most confused with {labels[worst]}" if confusion[i, worst] else ""))
    json.dump({"labels": labels, "val_accuracy": best_acc, "regressed": regressed,
               "per_class": per_class, "confusion": confusion.tolist()},
              open(out.with_suffix(".report.json"), "w"), indent=1)
    return {"labels": labels, "val_accuracy": best_acc, "clips": len(clips),
            "regressed": regressed,
            "model": str(out)}


if __name__ == "__main__":
    main()
