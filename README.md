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

- **No results have been verified in this environment.** There is no trained checkpoint
  (`pacl_ft.pth`) committed, and training has not been run here.
- The paper trains on **~30M image-text pairs (GCC-3M + GCC-12M + YFCC-15M)** for 10 days on
  4× A100; this repo trains on **MS-COCO 2017 (~118k images)**. It will **not** reproduce the
  paper's mIoU numbers.
- There is **no segmentation-benchmark evaluation** (mIoU on Pascal VOC / Pascal Context /
  COCO-Stuff / ADE20K) here — only single-image activation-map inference.

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
| Evaluation | mIoU on 4 seg benchmarks + 12 classification datasets | single-image activation map only |

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
├── scripts/download_coco.sh   # fetch MS-COCO 2017 into the expected layout
├── results/                   # sample activation maps (cat.png, dog.png)
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
