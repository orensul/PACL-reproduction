"""
Train PACL (Patch-Aligned Contrastive Learning) on MS-COCO.

Faithful to NMS05/Patch-Aligned-Contrastive-Learning, with these practical changes:
  * argparse-driven (paths, batch size, epochs, lr, workers, output) instead of hardcoded values
  * GPU count is auto-detected (nn.DataParallel over all visible GPUs) instead of a hardcoded 8

The default hyperparameters reproduce the original run:
  batch_size=1024, epochs=10, lr=1e-4 (Adam), ClipLoss temperature=0.1, 400px images.

NOTE on the contrastive loss and batch size:
  ClipLoss uses in-batch negatives, so the *global* batch size affects the result.
  To reproduce the original run keep --batch_size 1024. Lower it only if you OOM,
  and be aware that changes the effective number of negatives.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.image_caption_data import CocoDataset
from model.pacl import open_clip_pacl, ClipLoss


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device, amp=False):
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
    train_dataset = CocoDataset(args.train_dir, args.train_anno, apply_transform=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = CocoDataset(args.val_dir, args.val_anno, apply_transform=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model on all visible GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. PACL training requires GPUs.")
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus))
    device = torch.device("cuda:0")

    model = open_clip_pacl()
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("\n\t Started Training\n", flush=True)

    for epoch in range(args.epochs):
        begin = time.time()
        loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, amp=args.amp)
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
    return p.parse_args()


if __name__ == "__main__":
    train_clip(parse_args())
