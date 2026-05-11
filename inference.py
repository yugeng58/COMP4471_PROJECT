"""End-to-end Image-to-Video inference pipeline.

Given a static image I_0 and an action label (e.g. "walking"):
1. Encode I_0 -> E_0 (frozen CLIP ViT)
2. Encode label -> L    (frozen CLIP text)
3. Flow Model autoregressively generates (E_1, ..., E_n)
4. Renderer predicts optical flow: flow_t = Renderer(I_{t-1} | E_t, L, patches)
5. Warp I_{t-1} with flow_t to get I_t
6. Save output video

Usage:
    python inference.py --image path/to/image.jpg --label "walking" --out output.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, EMBED_DIM, FLOW_DROPOUT, FLOW_NUM_HEADS, FLOW_NUM_LAYERS,
    IMAGE_SIZE, UNET_BASE_CH, UNET_CH_MULTS, UNET_NUM_RES_BLOCKS,
)
from models.encoders import ImageEncoder, LabelEncoder
from models.flow_model import CausalFlowModel
from models.renderer import RendererUNet, warp_frame


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(), T.Resize((image_size, image_size)), T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def tensor_to_frame(t: torch.Tensor) -> np.ndarray:
    t = t.squeeze(0).clamp(0, 1).cpu()
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_video(frames: list[np.ndarray], path: str, fps: int = 8):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved video to {path} ({len(frames)} frames, {fps} fps)")


def resolve_checkpoint(name: str, fallbacks: tuple) -> Path:
    path = Path(name)
    if path.is_file():
        return path
    path = CHECKPOINT_DIR / name
    if path.is_file():
        return path
    for fb in fallbacks:
        fb_path = CHECKPOINT_DIR / fb
        if fb_path.is_file():
            return fb_path
    raise FileNotFoundError(f"No checkpoint found: {name}, fallbacks: {fallbacks}")


def run_inference(
    image_path: str,
    label: str,
    output_path: str,
    num_frames: int = 16,
    flow_ckpt: str = "flow_cos_epoch40.pt",
    renderer_ckpt: str = "renderer_flow_final.pt",
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading encoders...")
    image_encoder = ImageEncoder().to(device)
    label_encoder = LabelEncoder().to(device)

    print("Loading Flow Model...")
    flow_model = CausalFlowModel(
        embed_dim=EMBED_DIM, num_layers=FLOW_NUM_LAYERS,
        num_heads=FLOW_NUM_HEADS, dropout=FLOW_DROPOUT,
    ).to(device)
    ckpt_path = resolve_checkpoint(
        flow_ckpt, ("flow_cos_epoch40.pt", "flow_cos_final.pt", "flow_model_final.pt")
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    flow_model.load_state_dict(ckpt["model_state"])
    flow_model.eval()

    print("Loading Renderer...")
    renderer = RendererUNet(
        in_ch=3, embed_dim=EMBED_DIM,
        base_ch=UNET_BASE_CH, ch_mults=UNET_CH_MULTS,
        num_res_blocks=UNET_NUM_RES_BLOCKS,
    ).to(device)
    ckpt_path = resolve_checkpoint(
        renderer_ckpt,
        ("renderer_flow_final.pt", "renderer_flow_epoch10.pt",
         "renderer_ca_epoch20.pt", "renderer_epoch20.pt"),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    renderer.load_state_dict(ckpt["model_state"], strict=False)
    renderer.eval()

    print(f"Encoding image: {image_path}")
    I_0 = load_image(image_path, IMAGE_SIZE).to(device)

    with torch.no_grad():
        E_0, patches_0 = image_encoder(I_0, return_patches=True)
        L = label_encoder([label])

    print(f"Predicting embedding trajectory ({num_frames} frames)...")
    with torch.no_grad():
        E_seq = flow_model.generate(E_0, L, num_frames)

    print("Rendering frames...")
    frames = [tensor_to_frame(I_0)]
    I_prev = I_0
    patches_prev = patches_0

    for t in tqdm(range(num_frames)):
        E_t = E_seq[:, t, :]
        with torch.no_grad():
            flow = renderer(I_prev, E_t, L, patches_prev)
            I_t = warp_frame(I_prev, flow)
        frames.append(tensor_to_frame(I_t))
        I_prev = I_t
        with torch.no_grad():
            _, patches_prev = image_encoder(I_t, return_patches=True)

    save_video(frames, output_path)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Image-to-Video generation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--out", type=str, default="output.mp4")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--flow_ckpt", type=str, default="flow_cos_epoch40.pt")
    parser.add_argument("--renderer_ckpt", type=str, default="renderer_flow_final.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_inference(
        image_path=args.image, label=args.label, output_path=args.out,
        num_frames=args.num_frames, flow_ckpt=args.flow_ckpt,
        renderer_ckpt=args.renderer_ckpt, device=args.device,
    )


if __name__ == "__main__":
    main()
