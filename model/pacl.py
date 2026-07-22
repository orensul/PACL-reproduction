import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

"""
PACL model.

This is the core of PACL. It wraps a FROZEN CLIP ViT-B/16 and learns a small projection so
that individual VISION PATCH tokens align with the TEXT CLS token.

The comments below are cross-referenced with docs/paper_summary.txt: a tag like
[summary §3: "..."] points to a numbered section of that summary (and quotes the exact
sentence the code implements). Read the two side by side.
  §1 Training data          -> see train_pacl.py / data/image_caption_data.py
  §2 Prompt augmentation    -> see data/image_caption_data.py
  §3 Patch-aligned training -> the mechanism in THIS file: forward_visual / forward_text /
                               patch_alignment / forward, plus ClipLoss.
"""

class Patch_Projection(torch.nn.Module):
    def __init__(self):
        super(Patch_Projection, self).__init__()
        
        self.linear_projection = self.text_projection = nn.Sequential(
            nn.Linear(768, 512),
        )
        self.non_linear_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
    def forward(self, x):
        return self.linear_projection(x) + self.non_linear_projection(x)


class open_clip_pacl(torch.nn.Module):
    def __init__(self):
        super(open_clip_pacl, self).__init__()

        # [summary §3: "the CLIP image and text encoders remain frozen"] backbone stays frozen;
        # only the projection head(s) below are trained. (Paper freezes CLIP ViT-B/16; this reimpl
        # uses the open_clip laion2b weights instead of OpenAI's.)
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
        # Images are 400px (not CLIP's native 224px), so interpolate the position embeddings onto
        # the resulting 25x25 = 625 patch grid. Finer patches -> finer patch-level maps.
        self.clip_model.visual.positional_embedding = self.interpolate_pos_embed(self.clip_model.visual.positional_embedding.detach(), img_size=400)
        for p in self.clip_model.parameters(): p.requires_grad=False   # freeze the backbone

        # this makes sure that the unnormalized visual patch tokens are returned
        self.clip_model.visual.output_tokens = True
        # [summary §3: "A small learnable projection head projects each patch into CLIP's shared
        # embedding space" / "Only the small projection head is updated"] -- the ONLY trainable parts.
        # The summary describes a single small head on the patches; this reimpl trains BOTH a
        # visual_projection and a text_projection (see README "Differences from the paper").
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            Patch_Projection(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
        )

    def interpolate_pos_embed(self, pos_embed, img_size):
        cls_pos_embed, patch_pos_embed = pos_embed[0,:], pos_embed[1:,:] # torch.Size([768]) torch.Size([196, 768])
        new_num_patches = int(img_size // 16) # 25 for img_size=400
        new_patch_pos_embed = patch_pos_embed.reshape(1, 196, 768).transpose(1, 2).reshape(1, 768, 14, 14) # torch.Size([1, 768, 14, 14])
        new_patch_pos_embed = torch.nn.functional.interpolate(new_patch_pos_embed, size=(new_num_patches,new_num_patches), mode='bilinear') # torch.Size([1, 768, 25, 25])
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, 768, 625).transpose(1,2).squeeze(0) # torch.Size([625, 768])
        new_pos_embed = torch.cat((cls_pos_embed.unsqueeze(0), new_patch_pos_embed),dim=0) # torch.Size([626, 768])
        return torch.nn.Parameter(new_pos_embed)      
    
    def forward_visual(self, images):
        # [summary §3: "The frozen CLIP ViT produces patch embeddings (not the CLS token)" +
        # "A small learnable projection head projects each patch ..."] encode the image into PATCH
        # tokens (discard CLS, keep every patch), then project them into the shared space.
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        return self.visual_projection(visual_patches) # shape = [B, 196, 768]

    def forward_text(self, caps):
        # [summary §3: "The frozen CLIP text encoder produces a text embedding"] encode the text
        # into a single CLS token, then project it.
        text_cls = self.clip_model.encode_text(caps)
        return self.text_projection(text_cls) # shape = [B, 768]
    
    def patch_alignment(self, visual_patch_proj, text_cls_proj): # shapes =  [B, 196, 768], [B, 768]
        # [summary §3: "The cosine similarity between every patch and the text embedding is
        # computed. A softmax over these similarities gives an attention weight for each patch."]
        # NOTE: the summary uses softmax over tokens; this reimpl uses sigmoid(10*s) instead
        # (see README "Differences from the paper").

        # normalize visual patch tokens and then permute
        normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
        normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2,-1) # shapes =  [B, 768, 196]
        # normalize text cls token and unsqueeze (required for matmul)
        normalized_text_cls_proj = F.normalize(text_cls_proj, dim=-1)
        normalized_text_cls_proj = normalized_text_cls_proj.unsqueeze(1) # shapes =  [B, 1, 768]

        # compute dot product
        patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj # shapes =  [B, 1, 196]
        patch_activations = patch_activations.squeeze() # shapes =  [B, 196]
        # because of dot product, the range is between -1 (least similar) to +1 (most similar)
        # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        return F.sigmoid(patch_activations*10)
    
    def forward(self, images, caps):
        # [summary §3] the full training forward pass for an (image, text) pair.
        visual_proj = self.forward_visual(images)
        text_proj = self.forward_text(caps)
        # per-patch attention weights a(x,y)
        patch_activations = self.patch_alignment(visual_proj, text_proj) # shapes =  [B, 196]
        # [summary §3: "The weighted average of the patch embeddings becomes the image
        # representation for that text."]
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) # [B, 768]
        # return normalized (image-representation, text-embedding); ClipLoss compares them with
        # InfoNCE [summary §3: "compared with the text embedding using the standard InfoNCE ... loss"].
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)


"""
CLIP loss / Image-Text-Contrastive loss.
[summary §3: "This weighted image representation is compared with the text embedding using the
standard InfoNCE (CLIP) contrastive loss."] Symmetric cross-entropy over the image<->text
similarity matrix: each image must match its own caption against all other captions in the
batch (in-batch negatives), and vice versa. This objective drives §3's key intuition -- the
per-patch weights concentrate on the patches matching the text.
"""
class ClipLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = 1.0/temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        logits_per_image = self.logit_scale * image_features @ text_features.T
        logits_per_text = self.logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss