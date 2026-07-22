"""
Smoke test for the PACL plumbing: instantiates the model (triggers the open_clip weight
download), runs a forward pass on random data, and checks the shapes documented in
model/pacl.py -- without needing any real dataset. See PLAN_220726.md Phase 1.
"""
import os
import sys

import torch
import open_clip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pacl import open_clip_pacl, ClipLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = open_clip_pacl().to(device)
    model.eval()

    images = torch.randn(2, 3, 400, 400, device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    caps = tokenizer(["a picture of a cat.", "a picture of a dog."]).to(device)

    with torch.no_grad():
        visual_proj = model.forward_visual(images)
        assert visual_proj.shape == (2, 625, 512), f"forward_visual shape = {visual_proj.shape}"

        text_proj = model.forward_text(caps)
        assert text_proj.shape == (2, 512), f"forward_text shape = {text_proj.shape}"

        patch_weights = model.patch_alignment(visual_proj, text_proj)
        assert patch_weights.shape == (2, 625), f"patch_alignment shape = {patch_weights.shape}"

        visual_features, text_features = model(images, caps)
        assert visual_features.shape == (2, 512), f"model() visual shape = {visual_features.shape}"
        assert text_features.shape == (2, 512), f"model() text shape = {text_features.shape}"

        loss_fn = ClipLoss(temperature=0.1)
        loss = loss_fn(visual_features, text_features)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        print(f"ClipLoss = {loss.item()}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
