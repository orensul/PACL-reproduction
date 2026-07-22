"""
Pascal VOC 2012 mIoU evaluation for a trained PACL checkpoint. See PLAN_220726.md Phase 5a.

For each validation image: encode patches once, then for every VOC class build a text
prompt, run patch_alignment to get a per-patch score map, upsample all 21 (20 classes +
background) maps to the image's original resolution, argmax per pixel (with a
background threshold), and accumulate per-class intersection/union across the whole
split to report per-class IoU and mIoU -- the paper's headline metric (Table in the
paper reports 72.3 mIoU on this same benchmark, at a much larger training scale).
"""
import argparse
import os

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.pacl import open_clip_pacl

# Standard Pascal VOC class-index order (0 = background), matching SegmentationClass palettes.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Paper's exact 7-prompt CLIP ensemble (Appendix A.2.2) -- used for zero-shot inference:
# average the text embedding across all 7 per class, instead of a single template.
PAPER_PROMPT_TEMPLATES = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "{} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]


def load_weights(model, weights_path):
    saved = torch.load(weights_path, map_location="cpu")
    if isinstance(saved, dict) and "state_dict" in saved:
        saved = saved["state_dict"]
    target = model.state_dict()
    new_state = {}
    for name in target:
        if name in saved:
            new_state[name] = saved[name]
        elif "module." + name in saved:
            new_state[name] = saved["module." + name]
        else:
            raise KeyError(f"Missing key in checkpoint: {name}")
    model.load_state_dict(new_state)
    return model


class VOCSegDataset(Dataset):
    def __init__(self, voc_root, split="val", img_size=400, limit=None):
        seg_root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
        list_file = os.path.join(seg_root, "ImageSets", "Segmentation", f"{split}.txt")
        with open(list_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        if limit is not None:
            self.ids = self.ids[:limit]
        self.img_dir = os.path.join(seg_root, "JPEGImages")
        self.mask_dir = os.path.join(seg_root, "SegmentationClass")
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image = Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, f"{img_id}.png"))
        orig_w, orig_h = image.size
        image_t = self.transform(image)
        mask_np = np.array(mask, dtype=np.int64)  # HxW, values 0-20 (class) or 255 (ignore)
        return image_t, mask_np, (orig_h, orig_w)


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = [b[1] for b in batch]
    sizes = [b[2] for b in batch]
    return images, masks, sizes


def compute_class_scores(model, visual_proj, text_proj, weighting):
    """visual_proj: [B, N, 512], text_proj: [C, 512] -> per-class per-patch scores [B, C, N]."""
    B = visual_proj.shape[0]
    v = F.normalize(visual_proj, dim=-1)          # [B, N, 512]
    t = F.normalize(text_proj, dim=-1)            # [C, 512]
    cos = torch.einsum("bnd,cd->bcn", v, t)        # [B, C, N] cosine similarity in [-1, 1]
    if weighting == "sigmoid":
        return torch.sigmoid(cos * 10)             # matches model.patch_alignment exactly
    elif weighting == "softmax":
        return F.softmax(cos * 10, dim=-1)         # softmax over patches, per class
    raise ValueError(weighting)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = open_clip_pacl(weighting=args.weighting, activation=args.activation,
                            train_text_projection=not args.freeze_text_projection)
    load_weights(model, args.weights)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device).eval()

    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.no_grad():
        if args.prompt_ensemble:
            # paper (Appendix A.2.2): average the text embedding across all 7 prompts per class.
            per_template_embeds = []
            for template in PAPER_PROMPT_TEMPLATES:
                prompts = [template.format(c) for c in VOC_CLASSES[1:]]
                toks = tokenizer(prompts).to(device)
                per_template_embeds.append(model.forward_text(toks))  # [20, 512]
            text_proj = torch.stack(per_template_embeds, dim=0).mean(dim=0)  # [20, 512]
        else:
            prompts = [args.template.format(c) for c in VOC_CLASSES[1:]]  # 20 foreground classes
            toks = tokenizer(prompts).to(device)
            text_proj = model.forward_text(toks)  # [20, 512]

    dataset = VOCSegDataset(args.voc_root, split=args.split, img_size=args.img_size, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_fn)
    grid = args.img_size // 16  # 25 for the default 400px input

    num_classes = len(VOC_CLASSES)  # 21 (background + 20 foreground)
    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)

    bg_desc = (f"entropy_threshold={args.entropy_threshold}" if args.bg_method == "entropy"
               else f"bg_threshold={args.bg_threshold}")
    print(f"Evaluating {len(dataset)} images from VOC2012 '{args.split}' split "
          f"({num_classes} classes, weighting={args.weighting}, bg_method={args.bg_method}, {bg_desc}, "
          f"prompt_ensemble={args.prompt_ensemble})")

    with torch.no_grad():
        for batch_idx, (images, masks, sizes) in enumerate(loader):
            images = images.to(device)
            visual_proj = model.forward_visual(images)  # [B, N, 512]
            scores = compute_class_scores(model, visual_proj, text_proj, args.weighting)  # [B, 20, N]
            B = images.shape[0]
            scores = scores.reshape(B, 20, grid, grid)

            for i in range(B):
                H, W = sizes[i]
                up = F.interpolate(scores[i:i + 1], size=(H, W), mode="bilinear",
                                    align_corners=False)[0]  # [20, H, W], raw trained score scale

                if args.bg_method == "entropy":
                    # paper: "mask predictions with entropy above 1.5 as background". Convert the
                    # 20 raw class scores at each pixel into a classification distribution (softmax
                    # over classes, not the patch-weighting softmax), then threshold its entropy.
                    class_probs = F.softmax(up, dim=0)  # [20, H, W], sums to 1 over classes per pixel
                    entropy = -(class_probs * (class_probs.clamp(min=1e-12)).log()).sum(dim=0)  # [H, W]
                    argmax_cls = class_probs.argmax(dim=0)  # [H, W], 0..19
                    pred = torch.where(entropy <= args.entropy_threshold, argmax_cls + 1,
                                        torch.zeros_like(argmax_cls))
                else:
                    max_score, argmax_cls = up.max(dim=0)  # [H, W] each, argmax_cls in 0..19
                    pred = torch.where(max_score >= args.bg_threshold, argmax_cls + 1,
                                        torch.zeros_like(argmax_cls))
                pred = pred.cpu().numpy()

                gt = masks[i]
                valid = gt != 255
                for cls in range(num_classes):
                    p_c = (pred == cls) & valid
                    g_c = (gt == cls) & valid
                    intersection[cls] += np.logical_and(p_c, g_c).sum()
                    union[cls] += np.logical_or(p_c, g_c).sum()

            if (batch_idx + 1) % 10 == 0:
                print(f"  ...{min((batch_idx + 1) * args.batch_size, len(dataset))}/{len(dataset)} images", flush=True)

    iou = intersection / np.maximum(union, 1)
    print("\nPer-class IoU:")
    for cls in range(num_classes):
        print(f"  {VOC_CLASSES[cls]:14s} {iou[cls]:.4f}")
    print(f"\nmIoU = {iou.mean():.4f}  (paper reports 72.3 at full ~30M-pair scale)")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PACL mIoU on Pascal VOC 2012 segmentation")
    p.add_argument("--weights", default="pacl_ft_full.pth", help="path to trained PACL weights")
    p.add_argument("--voc_root", default="/home/orensultan_google_com/VOC",
                   help="directory containing VOCdevkit/VOC2012")
    p.add_argument("--split", default="val", help="VOC ImageSets/Segmentation split to evaluate")
    p.add_argument("--img_size", type=int, default=400, help="model input resolution (must be a multiple of 16)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--template", default="a picture of a {}.", help="prompt template applied to each class name")
    p.add_argument("--prompt_ensemble", action="store_true",
                   help="paper-faithful: average text embeddings over the paper's 7 CLIP prompts "
                        "(Appendix A.2.2) per class instead of a single --template")
    p.add_argument("--weighting", choices=["sigmoid", "softmax"], default="sigmoid",
                   help="patch weighting function; MUST match how --weights was trained "
                        "(both for correct scoring and, for 'activation'/'freeze_text_projection' "
                        "below, for the checkpoint's state_dict to load at all)")
    p.add_argument("--activation", choices=["gelu", "relu"], default="gelu",
                   help="must match the activation --weights was trained with (see train_pacl.py)")
    p.add_argument("--freeze_text_projection", action="store_true",
                   help="must match --weights training: pass this if the checkpoint was trained "
                        "with --freeze_text_projection (no text_projection head to load)")
    p.add_argument("--bg_method", choices=["threshold", "entropy"], default="threshold",
                   help="'threshold' (this repo's original): raw max-class-score cutoff. "
                        "'entropy' (paper-faithful): per-pixel softmax-over-classes entropy cutoff "
                        "(paper: 'mask predictions with entropy above 1.5 as background')")
    p.add_argument("--bg_threshold", type=float, default=0.7,
                   help="[bg_method=threshold] raw sigmoid-score threshold below which a pixel is "
                        "predicted background. 0.7 is an empirically-calibrated default for the "
                        "'sigmoid' weighting mode -- CLIP cosine similarities are positively biased "
                        "(rarely near 0 even for non-matches), so sigmoid(10*s) runs hot; 0.5 "
                        "under-predicts background badly. Needs re-tuning for '--weighting softmax'.")
    p.add_argument("--entropy_threshold", type=float, default=1.5,
                   help="[bg_method=entropy] paper's exact value: pixels with class-distribution "
                        "entropy above this are predicted background")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images (for smoke-testing)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
