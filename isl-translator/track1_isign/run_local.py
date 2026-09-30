"""Track 1 training + evaluation on the local laptop (RTX 3050, 4 GB VRAM).

Same paper recipe as the Kaggle notebook (arXiv 2609.12993). The only changes are
memory settings for a 4 GB GPU: the paper used batch 16 x accumulation 2
(effective 32); batch 16 does not fit in 4 GB, so we use batch 8 x accumulation 4,
which gives the same effective batch of 32.

    python run_local.py --features "D:/path/to/isign_partad_features" --variant t5-small-motion
    python run_local.py --features ... --smoke     # 30-step check that everything runs

Resuming: pass --init-from checkpoints/<variant>/final_model to continue from a
previous run (the paper's constant learning rate makes this safe).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

import pandas as pd
import torch

import pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True,
                   help="folder with the .npy files and iSign_v1.1.csv")
    p.add_argument("--variant", default="t5-small-motion", choices=list(pipeline.VARIANTS))
    p.add_argument("--split", default="paper", choices=["paper", "video"])
    p.add_argument("--out", default=str(HERE / "runs"))
    p.add_argument("--batch-size", type=int, default=8,
                   help="8 fits 4 GB VRAM (16 gives CUDA OOM); with --grad-accum 4 the effective batch is the paper's 32")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = load data in the main process; safest on 8 GB RAM")
    p.add_argument("--precision", default="fp16", choices=["fp16", "no"],
                   help="use 'no' if the log warns about NaN loss")
    p.add_argument("--init-from", default=None, help="checkpoint dir to continue from")
    p.add_argument("--eval-only", default=None, help="only evaluate this checkpoint dir")
    p.add_argument("--eval-limit", type=int, default=None,
                   help="evaluate on only the first N test samples (quick check)")
    p.add_argument("--smoke", action="store_true", help="30 training steps, tiny eval")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("run_local")

    features = Path(args.features)
    csv_path = features / "iSign_v1.1.csv"
    if not csv_path.exists():
        found = list(features.glob("*.csv"))
        if not found:
            sys.exit(f"No CSV in {features}. Expected iSign_v1.1.csv next to the .npy files.")
        csv_path = found[0]
    n_npy = sum(1 for _ in features.glob("*.npy"))
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU found: this will run on the CPU and be very slow.")
    else:
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), gb)
    log.info("features: %s (%d .npy files)", features, n_npy)

    df = pipeline.load_annotations(csv_path, features)
    split_fn = pipeline.paper_test_split if args.split == "paper" else pipeline.video_test_split
    train_df, test_df = split_fn(df)
    log.info("usable %d -> train %d / test %d  (paper: 18,867 -> 16,980 / 1,887)",
             len(df), len(train_df), len(test_df))

    out_dir = Path(args.out) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(out_dir / f"test_split_{args.split}.csv", index=False)

    cfg = {"batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
           "num_workers": args.workers, "mixed_precision": args.precision,
           "pin_memory": args.workers > 0}
    if args.smoke:
        cfg.update({"eval_steps": 10, "logging_steps": 5})

    if args.eval_only:
        checkpoint = Path(args.eval_only)
    else:
        t0 = time.time()
        checkpoint = pipeline.train(args.variant, train_df, features, out_dir, cfg=cfg,
                                    init_from=args.init_from,
                                    max_steps=30 if args.smoke else None)
        log.info("training finished in %.1f min -> %s", (time.time() - t0) / 60, checkpoint)

    limit = 20 if args.smoke else args.eval_limit
    metrics = pipeline.evaluate(args.variant, checkpoint, test_df, features,
                                out_dir / "eval", cfg=cfg, limit=limit)
    log.info("metrics: %s", json.dumps(metrics, indent=2))

    paper = {"t5-small": 0.189, "t5-base": 0.154, "t5-large": 0.132, "t5-small-motion": 0.298}
    print("\n| model | BLEU | chrF | ROUGE-L |")
    print("|---|---|---|---|")
    print(f"| {args.variant} (paper) | {paper[args.variant]} | - | - |")
    print(f"| {args.variant} (ours) | {metrics['BLEU']:.3f} | {metrics['chrF']:.2f} | "
          f"{metrics['ROUGE-L']:.2f} |")
    pred = pd.read_csv(out_dir / "eval" / "predictions.csv")
    print(f"\n{metrics['unique_predictions']} unique predictions out of {len(pred)}")
    print(pred.head(10).to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
