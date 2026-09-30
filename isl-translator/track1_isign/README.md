# Track 1 — iSign pose-to-text (exact reproduction of arXiv 2609.12993)

Paper: https://arxiv.org/abs/2609.12993 · Authors' code: https://github.com/manavdhamecha77/Sign-Lang-Trans
Released weights: https://huggingface.co/manavdhamecha77/iSign-t5-pose-to-text

## Datasets
| What | Link | Notes |
|---|---|---|
| iSign poses v1.1 (part AD used, as in the paper) | https://huggingface.co/datasets/Exploration-Lab/iSign | Gated (auto-approve), CC-BY-NC-SA-4.0 |
| iSign annotations | same repo, `iSign_v1.1.csv` | columns `uid,text` |
| Pretrained T5 | https://huggingface.co/google-t5/t5-small (also t5-base / t5-large) | downloaded automatically |

## Steps
1. Accept the terms at https://huggingface.co/datasets/Exploration-Lab/iSign, then create a read token at https://huggingface.co/settings/tokens
2. Kaggle → import `isign_01_download_partAD.ipynb`; add secret `HF_TOKEN`; Accelerator **None**, Internet **On**; Save & Run All.
3. Save its output as dataset `isign-partad-features`.
4. Kaggle → import `isign_02_train_eval.ipynb`; add that dataset; Accelerator **GPU T4 x2**, Internet **On**; Save & Run All.
5. Download `checkpoints/<variant>/best_model` from the Output tab.

## Requirements (Kaggle images already have torch, transformers, pandas, scikit-learn)
- notebook 1: `pose-format==0.15.0`
- notebook 2: `sacrebleu==2.6.0`, `rouge-score==0.1.2`, `sentencepiece`

## Files
- `src/paper_model.py`, `src/paper_pose_utils.py`: verbatim copies of the authors' code
- `src/isign_zip.py`: range-request reader for the split iSign zip (downloads only part AD)
- `src/features.py`: .pose to (frames, 300) using the paper's own loader
- `src/pipeline.py`: splits, training loop and evaluation, following the paper settings
- `tests/`: local checks: byte-identical zip reads, bit-identical preprocessing, train/eval smoke test
- `make_notebooks.py`: regenerates both notebooks from `src/`
