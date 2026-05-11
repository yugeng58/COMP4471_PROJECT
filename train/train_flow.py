"""Train the Causal Flow Model.

Uses pre-extracted CLIP embeddings — no CUDA models in DataLoader workers.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    EMBED_DIM,
    FLOW_DROPOUT,
    FLOW_EPOCHS,
    FLOW_LR,
    FLOW_NUM_HEADS,
    FLOW_NUM_LAYERS,
    GRAD_ACCUM_STEPS,
    LOG_EVERY,
    NUM_FRAMES,
    NUM_WORKERS,
    SAVE_EVERY,
)
from data.dataset import HMDB51EmbeddingDataset
from models.flow_model import CausalFlowModel


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset ---
    cache_path = Path(DATA_DIR) / "processed" / f"embeddings_n{NUM_FRAMES}.pt"
    if not cache_path.exists():
        print(f"Embedding cache not found: {cache_path}")
        print("Run: python -m data.preprocess")
        return

    print(f"Loading embeddings from {cache_path}...")
    emb_ds = HMDB51EmbeddingDataset(cache_path)
    loader = DataLoader(
        emb_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    print(f"Dataset size: {len(emb_ds)} videos")

    # --- Flow Model ---
    model = CausalFlowModel(
        embed_dim=EMBED_DIM,
        num_layers=FLOW_NUM_LAYERS,
        num_heads=FLOW_NUM_HEADS,
        dropout=FLOW_DROPOUT,
    ).to(device)
    print(f"Flow Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=FLOW_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FLOW_EPOCHS)

    def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1)).mean()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    global_step = 0
    for epoch in range(FLOW_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{FLOW_EPOCHS}")

        for batch_idx, (e_seq, L) in enumerate(pbar):
            e_seq = e_seq.to(device)  # [B, n+1, D]
            L = L.to(device)          # [B, D]

            gt = e_seq[:, 1:, :]      # [B, n, D]

            preds = model(e_seq, L)   # [B, n, D]
            loss = cosine_loss(preds, gt)

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += loss_val
            global_step += 1

            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            if global_step % LOG_EVERY == 0:
                print(f"  Step {global_step} | Avg Loss: {epoch_loss / (batch_idx + 1):.6f}")

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_epoch_loss:.6f}")

        scheduler.step()

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "optimizer": optimizer.state_dict()},
                CHECKPOINT_DIR / f"flow_cos_epoch{epoch+1}.pt",
            )

    torch.save(
        {"epoch": FLOW_EPOCHS, "model_state": model.state_dict(), "optimizer": optimizer.state_dict()},
        CHECKPOINT_DIR / "flow_cos_final.pt",
    )
    print("Flow Model training complete.")


if __name__ == "__main__":
    train()
