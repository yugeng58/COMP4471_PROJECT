"""Causal Transformer Flow Model.

Predicts embedding trajectory (E_1, ..., E_n) from E_0 and label embedding L.

Architecture: GPT-style decoder with causal mask.
Input:  [L, E_0, E_1, ..., E_{n-1}]
Output: [E_1_pred, E_2_pred, ..., E_n_pred]  (shifted by one)

At inference time, we can autoregressively generate beyond n frames.
"""

import torch
import torch.nn as nn
import math


class CausalFlowModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask (upper triangular, True = masked)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
            ),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        for pn, p in self.named_parameters():
            if p.dim() < 2:
                nn.init.zeros_(p)
            elif "out_proj" in pn:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * len(self.transformer.layers)))

    def forward(self, e_seq: torch.Tensor, label_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_seq: ground-truth embedding sequence [B, n+1, D]  (E_0, ..., E_n)
            label_embed: semantic label embedding L [B, D]

        Returns:
            preds: predicted next embeddings [B, n, D]
                   preds[:, t] is the prediction for E_{t+1}
        """
        B, n_plus_1, D = e_seq.shape
        n = n_plus_1 - 1

        # Input sequence: [L, E_0, E_1, ..., E_{n-1}]
        # Causal mask ensures position t sees only [L, E_0, ..., E_{t-1}]
        seq_len = 1 + n
        assert seq_len <= self.max_seq_len

        tokens = torch.zeros(B, seq_len, D, device=e_seq.device, dtype=e_seq.dtype)
        tokens[:, 0] = label_embed            # L
        tokens[:, 1:] = e_seq[:, :n, :]       # E_0 .. E_{n-1}

        x = tokens + self.pos_embed[:, :seq_len, :]
        mask = self.causal_mask[:seq_len, :seq_len]
        h = self.transformer(src=x, mask=mask, is_causal=True)

        # Position 0 (L) ignored; positions 1..n predict E_1..E_n
        preds = self.out_proj(h[:, 1:, :])  # [B, n, D]
        return preds

    @torch.no_grad()
    def generate(
        self, e0: torch.Tensor, label_embed: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Autoregressive generation of embedding trajectory."""
        B, D = e0.shape
        device = e0.device

        generated = [e0]
        current_e = e0

        for _ in range(num_frames):
            # Build sequence so far: L, E_0, E_1, ..., E_t
            seq_len = 1 + len(generated)
            tokens = torch.zeros(B, seq_len, D, device=device, dtype=e0.dtype)
            tokens[:, 0] = label_embed
            for i, emb in enumerate(generated):
                tokens[:, 1 + i] = emb

            x = tokens + self.pos_embed[:, :seq_len, :]
            mask = self.causal_mask[:seq_len, :seq_len]
            h = self.transformer(src=x, mask=mask, is_causal=True)

            next_e = self.out_proj(h[:, -1, :])  # last position predicts E_{t+1}
            generated.append(next_e)
            current_e = next_e

        # Stack E_1..E_n
        return torch.stack(generated[1:], dim=1)  # [B, n, D]
