"""Train the Renderer UNet for optical flow prediction.

The model predicts a dense flow field from I_{t-1}, conditioned on E_t, L,
and CLIP patch tokens. The flow warps I_{t-1} to reconstruct I_t.

Losses:
  - L1(warped_frame, gt_frame) — reconstruction
  - TV(flow) — spatial smoothness
  - L1(pred_flow, farneback_flow) — pseudo-supervision from Farneback
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BATCH_SIZE, CHECKPOINT_DIR, DATA_DIR, EMBED_DIM, GRAD_ACCUM_STEPS,
    IMAGE_SIZE, LOG_EVERY, NUM_FRAMES, NUM_WORKERS, RENDERER_EPOCHS,
    RENDERER_LR, SAVE_EVERY, UNET_BASE_CH, UNET_CH_MULTS, UNET_NUM_RES_BLOCKS,
)
from data.dataset import HMDB51FrameDataset
from models.encoders import ImageEncoder, LabelEncoder
from models.renderer import RendererUNet, warp_frame

RECON_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 0.05
FARNEBACK_WEIGHT = 1.0


def _to_gray(frame: torch.Tensor) -> np.ndarray:
    """[3, H, W] tensor -> HxW uint8 gray image."""
    frame = frame.detach().cpu().clamp(0, 1)
    frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def compute_farneback(prev_frames: torch.Tensor, cur_frames: torch.Tensor,
                      max_flow: float) -> torch.Tensor:
    """Compute Farneback optical flow on CPU, return [B, 2, H, W] tensor."""
    flows = []
    for prev, cur in zip(prev_frames, cur_frames):
        pg = _to_gray(prev)
        cg = _to_gray(cur)
        flow = cv2.calcOpticalFlowFarneback(
            pg, cg, None, pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow = np.clip(flow, -max_flow, max_flow)
        flows.append(flow)
    flow_np = np.stack(flows, axis=0)  # [B, H, W, 2]
    return torch.from_numpy(flow_np).permute(0, 3, 1, 2).float()


def flow_tv(flow: torch.Tensor) -> torch.Tensor:
    """Total variation smoothness loss for flow."""
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    return dx + dy


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading encoders...")
    image_encoder = ImageEncoder().to(device)
    label_encoder = LabelEncoder().to(device)

    print("Building dataset...")
    ds = HMDB51FrameDataset(
        data_dir=DATA_DIR, num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE, frame_stride=2, split="train",
    )
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    print(f"Dataset size: {len(ds)} frame pairs")

    model = RendererUNet(
        in_ch=3, embed_dim=EMBED_DIM,
        base_ch=UNET_BASE_CH, ch_mults=UNET_CH_MULTS,
        num_res_blocks=UNET_NUM_RES_BLOCKS, max_flow_px=16.0,
    ).to(device)
    print(f"Renderer params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=RENDERER_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RENDERER_EPOCHS)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(RENDERER_EPOCHS):
        model.train()
        epoch_rec, epoch_smooth, epoch_fb, epoch_total = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{RENDERER_EPOCHS}")

        for batch_idx, (prev_frames, cur_frames, labels) in enumerate(pbar):
            B = prev_frames.shape[0]
            prev_frames = prev_frames.to(device)
            cur_frames = cur_frames.to(device)

            with torch.no_grad():
                E_cur, patches = image_encoder(cur_frames, return_patches=True)
                L = label_encoder(list(labels)).to(device)

            # Predict flow, warp previous frame
            pred_flow = model(prev_frames, E_cur, L, patches)
            warped = warp_frame(prev_frames, pred_flow)

            # Reconstruction loss
            loss_rec = F.l1_loss(warped, cur_frames)

            # Flow smoothness
            loss_smooth = flow_tv(pred_flow)

            # Farneback pseudo-supervision
            fb_flow = compute_farneback(prev_frames.cpu(), cur_frames.cpu(),
                                        model.max_flow_px).to(device)
            loss_fb = F.l1_loss(pred_flow, fb_flow)

            loss = (RECON_WEIGHT * loss_rec +
                    SMOOTHNESS_WEIGHT * loss_smooth +
                    FARNEBACK_WEIGHT * loss_fb)

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_total += loss.item() * GRAD_ACCUM_STEPS
            epoch_rec += loss_rec.item()
            epoch_smooth += loss_smooth.item()
            epoch_fb += loss_fb.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}",
                "rec": f"{loss_rec.item():.4f}",
                "fb": f"{loss_fb.item():.4f}",
            })

            if global_step % LOG_EVERY == 0:
                n = batch_idx + 1
                print(f"  Step {global_step} | Total: {epoch_total/n:.6f}  "
                      f"Rec: {epoch_rec/n:.6f}  FB: {epoch_fb/n:.6f}  "
                      f"TV: {epoch_smooth/n:.6f}")

        n = len(loader)
        print(f"Epoch {epoch+1} | Total: {epoch_total/n:.6f}  "
              f"Rec: {epoch_rec/n:.6f}  FB: {epoch_fb/n:.6f}  "
              f"TV: {epoch_smooth/n:.6f}")

        scheduler.step()

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer": optimizer.state_dict()},
                CHECKPOINT_DIR / f"renderer_flow_epoch{epoch+1}.pt",
            )

    torch.save(
        {"epoch": RENDERER_EPOCHS, "model_state": model.state_dict(),
         "optimizer": optimizer.state_dict()},
        CHECKPOINT_DIR / "renderer_flow_final.pt",
    )
    print("Renderer training complete.")


if __name__ == "__main__":
    train()
