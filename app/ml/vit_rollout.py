# app/ml/vit_rollout.py
import torch
import numpy as np

class VitAttentionRollout:
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def __call__(self, input_tensor):
        # Forward pass with attentions
        outputs = self.model(input_tensor, output_attentions=True)
        attns = outputs.attentions[-1]  # last layer attentions (batch, heads, tokens, tokens)

        # Fuse attention heads
        if self.head_fusion == "mean":
            attn = attns.mean(dim=1)  # (batch, tokens, tokens)
        elif self.head_fusion == "max":
            attn = attns.max(dim=1)[0]
        else:
            attn = attns.min(dim=1)[0]

        # CLS token to patch tokens (exclude self)
        cls_attn = attn[:, 0, 1:]  # shape (batch, 196)
        cls_attn = cls_attn.reshape(1, 14, 14)  # ViT base patch16 → 14x14

        # Upsample to input image size
        cam = torch.nn.functional.interpolate(
            cls_attn.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze()

        # ✅ Fix: detach before numpy
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
