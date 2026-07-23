# PACL Reproduction

A runnable re-implementation of **PACL — Patch Aligned Contrastive Learning**, from:

> **Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning**
> Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang, Ashish Shah, Philip H.S. Torr, Ser-Nam Lim.
> arXiv:2212.04994 (Dec 2022). [[paper]](https://arxiv.org/abs/2212.04994)

This repo trains a patch-aligned CLIP on **MS-COCO** and produces **patch-level activation
maps** for a text prompt (which region of an image a caption refers to). It follows the
open-source [NMS05/Patch-Aligned-Contrastive-Learning](https://github.com/NMS05/Patch-Aligned-Contrastive-Learning)
re-implementation, with practical fixes for headless servers and CLI-driven configuration.

---

## ⚠️ Scope & honest status

This is a faithful reproduction of the PACL **method/mechanism** and the community COCO demo —
**not** of the paper's headline benchmark results. Please read this before citing it as "reproducing PACL."

- The method has been **trained and evaluated end-to-end** here (see [Results](#results)):
  **17.72% mIoU on Pascal VOC2012 `val`**, versus the paper's **72.3**. This is a working,
  repeatable pipeline — **not** a reproduction of the paper's numbers.
- The paper trains on **~30M image-text pairs (GCC-3M + GCC-12M + YFCC-15M)** for 10 days on
  4× A100; this repo trains on **MS-COCO 2017 (~118k images)**. It will **not** reproduce the
  paper's mIoU numbers.
- Segmentation-benchmark evaluation covers **Pascal VOC only** (`evaluate_miou.py`).
  Pascal Context, COCO-Stuff and ADE20K are still not implemented.
- **No trained checkpoint is committed** (`*.pth` is git-ignored), so the number below cannot be
  re-derived from a fresh clone without retraining.

### Differences from the paper

| Aspect | Paper (Mukhoti et al. 2022) | This repo |
|---|---|---|
| Training data | GCC-3M + GCC-12M + YFCC-15M (~30M pairs) | MS-COCO 2017 (~118k images) |
| Trainable params | frozen text encoder + ~1.1M-param residual **vision embedder** | `visual_projection` **and** `text_projection` |
| Backbone | OpenAI CLIP ViT-B/16 & ViT-L/14 (frozen) | open_clip ViT-B-16, `laion2b-s34b-b88K` (frozen) |
| Optimizer | AdamW, lr 5e-4 (cosine), weight decay 0.2 | Adam, lr 1e-4 |
| Batch / schedule | 4096 global, 10 epochs (4× A100, ~10 days) | 1024 global, 10 epochs |
| Patch weighting | `softmax(s)` over tokens | `sigmoid(10·s)` |
| Inference | stride-4 trick + bilinear upscale → full segmentation | 25×25 patch activation map |
| Evaluation | mIoU on 4 seg benchmarks + 12 classification datasets | mIoU on Pascal VOC only |
| **Result** | **72.3 mIoU** (VOC, ViT-B/16) | **17.72 mIoU** (VOC2012 val) |

Since 2026-07-22 the repo also ships **opt-in flags** to run the paper's exact choices
(`--weighting softmax`, `--activation relu`, `--freeze_text_projection`, `--optimizer adamw`,
`--lr_schedule cosine`, `--paper_faithful_prompts`). They all **default to off**, so the "This repo"
column above describes the default behavior, and both configurations stay comparable under the
same eval harness.

---

## Results

Run of **2026-07-22** (commit `68068b4`). Two training runs were done:

| Run | Training data | Checkpoint | VOC2012 val mIoU |
|---|---|---|---|
| Sanity run (Phase 3) | COCO `val2017` (~5k images) | `pacl_ft.pth` | not evaluated |
| Full run (Phase 5a) | COCO `train2017` (~118k images) | `pacl_ft_full.pth` | **17.72%** |
| *Paper, for reference* | *~30M pairs (GCC + YFCC)* | — | *72.3* |

Evaluation used the `evaluate_miou.py` defaults: 400px input, prompt template
`"a picture of a {}."`, `sigmoid(10·s)` patch weighting, background by threshold at `0.7`,
no prompt ensemble.

**Qualitative (Phase 4).** Activation maps localize, but imperfectly. Moving from the sanity run
to the full run visibly sharpened the *keyboard* and *cat* maps — compare `results/*.png` (sanity)
with `results/full_*.png` (full), side by side in `results/old_vs_new.png`. **Dog and cat are still
confused** on the anecdotal test image, so text conditioning is only partly working.

### What was not recorded

The run predates any result-logging, so beyond the single mIoU figure above **nothing was
persisted**: no training log or loss curve, no per-class IoU breakdown, no epoch count or
wall-clock time, and no record of the exact CLI arguments used. The training defaults
(batch 1024, 10 epochs, Adam lr 1e-4, `sigmoid`, `gelu`) are the *likely* configuration because
the paper-fidelity flags default to off — but this is inferred, not logged. Treat 17.72% as a
baseline to beat, not a precisely characterized data point.

---

## Method in one paragraph

CLIP's contrastive loss only aligns the **CLS** tokens of the image and text encoders, so CLIP
has poor **patch-level** alignment. PACL modifies CLIP's compatibility function: it computes the
cosine similarity between the text CLS token and **every vision patch token**, turns those
similarities into per-patch weights, takes a weighted sum over patch tokens, and contrasts the
pooled vector against the text CLS with InfoNCE. After training, the per-patch weights localize
the region an input caption refers to — i.e. zero-shot open-vocabulary segmentation with no
segmentation masks used during training.

## Repository layout

```
PACL-reproduction/
├── model/pacl.py              # PACL model (open_clip ViT-B-16 backbone) + ClipLoss
├── data/
│   ├── image_caption_data.py  # MS-COCO dataset + noun-phrase prompt sampling
│   └── utils.py               # inference preprocessing (resize/normalize/tokenize)
├── train_pacl.py              # training loop (DataParallel, bf16 AMP), argparse-driven
├── pacl_inference.py          # single-image activation map for a text prompt (headless)
├── evaluate_miou.py           # Pascal VOC mIoU harness (per-class IoU + mIoU)
├── scripts/
│   ├── download_coco.sh       # fetch MS-COCO 2017 into the expected layout
│   └── smoke_test.py          # shape/loss sanity check, no data needed
├── results/                   # sample activation maps (sanity: *.png, full run: full_*.png)
├── docs/                      # the paper + a short summary
└── requirements.txt
```

## Setup

CUDA GPU(s) required for training. Install a CUDA build of PyTorch matching your driver first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if not pulled in by requirements
```

## Data

```bash
bash scripts/download_coco.sh /path/to/MSCOCO
# creates train2017/, val2017/, annotations/captions_{train,val}2017.json
```

## Train

```bash
python train_pacl.py \
  --train_dir  /path/to/MSCOCO/train2017/ \
  --train_anno /path/to/MSCOCO/annotations/captions_train2017.json \
  --val_dir    /path/to/MSCOCO/val2017/ \
  --val_anno   /path/to/MSCOCO/annotations/captions_val2017.json \
  --batch_size 1024 --epochs 10 --lr 1e-4 --amp \
  --save_path pacl_ft.pth
```

`--batch_size` is the **global** batch — it sets the number of in-batch contrastive negatives,
so lower it only if you OOM. `--amp` enables bf16 (needed to fit batch 1024 on 40GB GPUs).

## Inference

```bash
python pacl_inference.py \
  --weights pacl_ft.pth \
  --image  https://example.com/cat.jpg \
  --caption "a picture of a cat." \
  --out results/activation_map.png
```

Produces a side-by-side of the input image and the patch activation map for the caption.

## Evaluate (Pascal VOC mIoU)

Needs the VOC2012 devkit (`JPEGImages/`, `SegmentationClass/`, `ImageSets/Segmentation/`):

```bash
python evaluate_miou.py \
  --weights  pacl_ft_full.pth \
  --voc_root /path/to/VOC \
  --split    val
```

Prints per-class IoU and the mean. Useful knobs: `--bg_threshold` (background cut-off, default
`0.7`), `--bg_method entropy`, `--prompt_ensemble`, `--weighting softmax`, and `--limit N` to
smoke-test on the first N images. Evaluation flags must **match how the checkpoint was trained** —
e.g. a model trained with `--weighting softmax` should be evaluated with it too.

## Citation

```bibtex
@article{mukhoti2022pacl,
  title   = {Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning},
  author  = {Mukhoti, Jishnu and Lin, Tsung-Yu and Poursaeed, Omid and Wang, Rui
             and Shah, Ashish and Torr, Philip H.S. and Lim, Ser-Nam},
  journal = {arXiv preprint arXiv:2212.04994},
  year    = {2022}
}
```

## Attribution

PACL is the work of the paper's authors. This repository is an independent reproduction for
study, based on the [NMS05](https://github.com/NMS05/Patch-Aligned-Contrastive-Learning)
open-source re-implementation. It is not affiliated with or endorsed by the authors or Meta AI.
