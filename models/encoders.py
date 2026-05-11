import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class LabelEncoder(nn.Module):
    """Frozen CLIP text encoder: action label -> semantic embedding L (512-dim)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

    @torch.no_grad()
    def forward(self, labels: list[str]) -> torch.Tensor:
        tokens = self.processor(text=labels, return_tensors="pt", padding=True, truncation=True)
        device = self.clip.text_model.embeddings.token_embedding.weight.device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return self.clip.get_text_features(**tokens)


class ImageEncoder(nn.Module):
    """Frozen CLIP ViT: image -> global embedding E (512-dim) + spatial patch tokens."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.register_buffer("mean", CLIP_MEAN.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", CLIP_STD.view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, return_patches: bool = False):
        """
        images: [B, 3, H, W] in [0, 1]
        Returns:
            E: [B, 512] global CLIP embedding
            patches: [B, 49, 512] spatial patch tokens (if return_patches=True)
        """
        device = self.clip.vision_model.embeddings.patch_embedding.weight.device
        images = images.to(device=device, dtype=torch.float32)
        images = (images - self.mean) / self.std

        global_emb = self.clip.get_image_features(pixel_values=images)  # [B, 512]

        if return_patches:
            vision_output = self.clip.vision_model(pixel_values=images).last_hidden_state  # [B, 50, 768]
            patches = vision_output[:, 1:, :]                          # [B, 49, 768]  skip CLS
            patches = self.clip.visual_projection(patches)             # [B, 49, 512]
            return global_emb, patches

        return global_emb
