"""
Single-image PACL inference: produce a patch activation map for a text prompt.

Faithful to the original pacl_inference.py, with practical fixes:
  * weights path is a CLI arg and defaults to pacl_ft.pth (what train_pacl.py saves);
    the original hardcoded 'pacl.pth', which never matched the training output.
  * strips the DataParallel 'module.' prefix robustly (works whether or not it's present).
  * headless-safe: renders the activation map (and an overlay on the image) to a PNG
    instead of plt.show(), so it runs on a server with no display.
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from data.utils import prepare_data
from model.pacl import open_clip_pacl


def load_weights(model, weights_path):
    saved = torch.load(weights_path, map_location="cpu")
    if isinstance(saved, dict) and "state_dict" in saved:
        saved = saved["state_dict"]
    target = model.state_dict()
    new_state = {}
    for name in target:
        if name in saved:                      # plain state dict
            new_state[name] = saved[name]
        elif "module." + name in saved:        # DataParallel-wrapped state dict
            new_state[name] = saved["module." + name]
        else:
            raise KeyError(f"Missing key in checkpoint: {name}")
    model.load_state_dict(new_state)
    return model


def load_image(source):
    if source.startswith("http://") or source.startswith("https://"):
        return Image.open(requests.get(source, stream=True).raw).convert("RGB")
    return Image.open(source).convert("RGB")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = open_clip_pacl()
    model = load_weights(model, args.weights)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device).eval()

    process = prepare_data()
    pil_image = load_image(args.image)

    with torch.no_grad():
        # This reuses the SAME §3 mechanism as training, but stops at the per-patch weights instead
        # of pooling them -- those weights ARE the segmentation/activation map.
        # [summary §3: "The frozen CLIP ViT produces patch embeddings"] -> projected patch tokens
        image = process.preprocess_image(pil_image).unsqueeze(0).to(device)
        visual_proj = model.forward_visual(image)

        # [summary §3: "The frozen CLIP text encoder produces a text embedding"] -> projected text
        caption = process.preprocess_text(args.caption).to(device)
        text_proj = model.forward_text(caption)

        # [summary §3: "cosine similarity between every patch and the text embedding ... attention
        # weight for each patch"] -> one score per patch, reshaped to a 25x25 map below.
        similarity_scores = model.patch_alignment(visual_proj, text_proj)  # [625]

    # 400px / 16 = 25 patches per side
    scores = similarity_scores.reshape(1, 25, 25)
    scores = T.GaussianBlur(kernel_size=3)(scores).reshape(25, 25).cpu().numpy()

    # save the raw activation map
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pil_image)
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(scores)
    axes[1].set_title(f'"{args.caption}"')
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved activation map to {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="PACL single-image inference")
    p.add_argument("--weights", default="pacl_ft.pth", help="path to trained PACL weights")
    p.add_argument("--image",
                   default="https://assets3.thrillist.com/v1/image/3053693/516x516/"
                           "flatten;scale;matte=ffffff=center;jpeg_quality=70.jpg",
                   help="image URL or local path")
    p.add_argument("--caption", default="a picture of a cat.")
    p.add_argument("--out", default="results/activation_map.png")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
