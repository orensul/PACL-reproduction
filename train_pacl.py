"""
Train PACL (Patch-Aligned Contrastive Learning) on MS-COCO.

Faithful to NMS05/Patch-Aligned-Contrastive-Learning, with these practical changes:
  * argparse-driven (paths, batch size, epochs, lr, workers, output) instead of hardcoded values
  * GPU count is auto-detected (nn.DataParallel over all visible GPUs) instead of a hardcoded 8

The default hyperparameters are this repo's original (non-paper-faithful) sanity-run
defaults: batch_size=1024, epochs=10, lr=1e-4 (Adam), ClipLoss temperature=0.1, 400px images.
The paper's OWN settings (Appendix A.2.1) are: batch_size=4096 global (1024/GPU x 4 GPUs),
epochs=10, AdamW(lr=5e-4, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) with a Cosine
Annealing schedule over all iterations -- pass --optimizer adamw --batch_size 4096
--lr 5e-4 --weight_decay 0.2 --beta1 0.9 --beta2 0.98 --eps 1e-6 --lr_schedule cosine
--weighting softmax --activation relu --freeze_text_projection --paper_faithful_prompts
to match. See PLAN_220726.md and the paper-fidelity-divergences project memory.

NOTE on the contrastive loss and batch size:
  ClipLoss uses in-batch negatives, so the *global* batch size affects the result.
  Lower it only if you OOM, and be aware that changes the effective number of negatives.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.image_caption_data import CocoDataset
from model.pacl import open_clip_pacl, ClipLoss


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device, amp=False, scheduler=None):
    epoch_loss = []
    model.train()
    for i, (images, caps) in enumerate(train_data_loader):
        images = images.to(device)
        caps = caps.to(device)

        optimizer.zero_grad()
        # bf16 autocast lets global batch 1024 fit on 40GB GPUs; bf16 needs no GradScaler.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            visual_features, text_features = model(images, caps)
            _loss = loss_fn(visual_features, text_features)
        epoch_loss.append(_loss.item())
        _loss.backward()
        optimizer.step()
        # paper's Cosine Annealing schedule is defined over total iterations, so step per batch.
        if scheduler is not None:
            scheduler.step()

        if i % 10 == 0:
            print("train_loss = ", _loss.item(), flush=True)
    return np.mean(epoch_loss)


def val_one_epoch(val_data_loader, model, loss_fn, device, amp=False):
    epoch_loss = []
    model.eval()
    with torch.no_grad():
        for i, (images, caps) in enumerate(val_data_loader):
            images = images.to(device)
            caps = caps.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                visual_features, text_features = model(images, caps)
                _loss = loss_fn(visual_features, text_features)
            epoch_loss.append(_loss.item())
    return np.mean(epoch_loss)


def train_clip(args):
    # DataLoaders
    # [summary §1 Training data] the paper uses GCC-3M + GCC-12M + YFCC15M; this reimpl uses
    # MS-COCO (image + caption per example). See README "Differences from the paper".
    train_dataset = CocoDataset(args.train_dir, args.train_anno, apply_transform=True,
                                 paper_faithful_prompts=args.paper_faithful_prompts)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = CocoDataset(args.val_dir, args.val_anno, apply_transform=False,
                               paper_faithful_prompts=args.paper_faithful_prompts)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model on all visible GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. PACL training requires GPUs.")
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))
    device = torch.device("cuda:0")

    model = open_clip_pacl(weighting=args.weighting, activation=args.activation,
                            train_text_projection=not args.freeze_text_projection)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    print(f"\n\t Model loaded on {num_gpus} GPU(s): {device_ids}")
    print("\t Total Params     = ", sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"\t Global batch size = {args.batch_size}  (~{args.batch_size // max(num_gpus,1)} per GPU)")

    # Train
    # [summary §3: "compared with the text embedding using the standard InfoNCE (CLIP) contrastive
    # loss"] -- ClipLoss is that InfoNCE objective; the optimizer only updates the trainable
    # projection head(s), since the CLIP backbone was frozen in open_clip_pacl.__init__.
    loss_fn = ClipLoss(temperature=args.temperature)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       betas=(args.beta1, args.beta2), eps=args.eps,
                                       weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      betas=(args.beta1, args.beta2), eps=args.eps)

    scheduler = None
    if args.lr_schedule == "cosine":
        # paper (Appendix A.2.1): "Cosine Annealing schedule where the maximum number of
        # iterations is set as ... number of epochs x number of iterations per epoch".
        total_iters = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    print("\n\t Started Training\n", flush=True)

    for epoch in range(args.epochs):
        begin = time.time()
        loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, amp=args.amp, scheduler=scheduler)
        val_loss = val_one_epoch(val_loader, model, loss_fn, device, amp=args.amp)

        print('\n\t Epoch....', epoch + 1)
        print("\t Training loss ......", round(float(loss), 4))
        print("\t Val loss ......", round(float(val_loss), 4))
        print('\t Time per epoch (in mins) = ', round((time.time() - begin) / 60, 2), '\n', flush=True)

        # save every epoch so a long run is checkpointed
        torch.save(model.state_dict(), args.save_path)

    torch.save(model.state_dict(), args.save_path)
    print(f"\n\t Saved weights to {args.save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train PACL on MS-COCO")
    base = "/home/Dataset/Visual_Recognition/MSCOCO"
    p.add_argument("--train_dir",  default=f"{base}/train2017/")
    p.add_argument("--train_anno", default=f"{base}/annotations/captions_train2017.json")
    p.add_argument("--val_dir",    default=f"{base}/val2017/")
    p.add_argument("--val_anno",   default=f"{base}/annotations/captions_val2017.json")
    p.add_argument("--batch_size", type=int, default=1024, help="GLOBAL batch (affects contrastive negatives)")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--amp", action="store_true",
                   help="bf16 mixed precision; needed to fit global batch 1024 on 40GB GPUs")
    p.add_argument("--save_path",  default="pacl_ft.pth")

    # Paper-fidelity flags (see PLAN_220726.md / paper-fidelity-divergences memory). Defaults
    # below all preserve this repo's ORIGINAL (non-paper-faithful) behavior.
    p.add_argument("--weighting", choices=["sigmoid", "softmax"], default="sigmoid",
                   help="patch_alignment weighting; 'softmax' matches the paper's Eq. (2)-(3)")
    p.add_argument("--activation", choices=["gelu", "relu"], default="gelu",
                   help="Patch_Projection main-branch activation; 'relu' matches paper Appendix A.1")
    p.add_argument("--freeze_text_projection", action="store_true",
                   help="paper trains ONLY the vision embedder; text side stays raw frozen CLIP "
                        "text_cls with no trainable head at all")
    p.add_argument("--paper_faithful_prompts", action="store_true",
                   help="emit the original caption AND a templated noun phrase as two separate "
                        "training examples per image (doubles epoch length), using the paper's 7 "
                        "CLIP templates, instead of this repo's original 50/50 substitute over 5 templates")
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999, help="paper uses 0.98 for AdamW")
    p.add_argument("--eps", type=float, default=1e-8, help="paper uses 1e-6 for AdamW")
    p.add_argument("--lr_schedule", choices=["none", "cosine"], default="none",
                   help="'cosine' matches the paper's Cosine Annealing over all iterations")
    return p.parse_args()


if __name__ == "__main__":
    train_clip(parse_args())
