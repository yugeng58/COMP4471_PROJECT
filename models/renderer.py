"""UNet Renderer that predicts optical flow from I_{t-1}, conditioned on E_t and L.

Output: dense flow field [B, 2, H, W] — warp I_{t-1} with this flow to get I_t.
The embedding condition provides semantic guidance about the action type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feat_dim)
        self.beta_proj = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(cond)[:, :, None, None]
        beta = self.beta_proj(cond)[:, :, None, None]
        return gamma * x + beta


class CrossAttnBlock(nn.Module):
    """Cross-attention from feature map to CLIP patch tokens."""

    def __init__(self, ch: int, patch_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.norm_q = nn.GroupNorm(8, ch)
        self.norm_kv = nn.LayerNorm(patch_dim)
        self.q_proj = nn.Linear(ch, ch)
        self.k_proj = nn.Linear(patch_dim, ch)
        self.v_proj = nn.Linear(patch_dim, ch)
        self.out_proj = nn.Linear(ch, ch)
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

    def forward(self, x: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x_norm = self.norm_q(x).reshape(B, C, H * W).transpose(1, 2)
        q = self.q_proj(x_norm)
        patches_norm = self.norm_kv(patches)
        k = self.k_proj(patches_norm)
        v = self.v_proj(patches_norm)

        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, 49, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, 49, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out).transpose(1, 2).reshape(B, C, H, W)
        return residual + out


class ResBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.film = FiLMBlock(cond_dim, ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        x = self.film(x, cond)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_res: int = 2):
        super().__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, cond_dim) for _ in range(num_res)])

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.downsample(x)
        for block in self.res_blocks:
            x = block(x, cond)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int,
                 patch_dim: int = 512, num_res: int = 2, use_cross_attn: bool = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_in = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(out_ch, cond_dim) for _ in range(num_res)])
        self.cross_attn = CrossAttnBlock(out_ch, patch_dim) if use_cross_attn else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor,
                patches: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x, cond)
        if self.cross_attn is not None:
            x = self.cross_attn(x, patches)
        return x


class FlowHead(nn.Module):
    """Predict dense flow from decoder features, with multi-scale refinement."""

    def __init__(self, in_ch: int, max_flow_px: float = 16.0):
        super().__init__()
        self.max_flow_px = max_flow_px
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 2, 3, padding=1)
        # Small random init so flow starts non-zero
        nn.init.normal_(self.conv3.weight, std=0.01)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        flow = self.conv3(x)
        flow = torch.tanh(flow) * self.max_flow_px
        return flow


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp frame using optical flow. flow: [B, 2, H, W] (dx, dy in pixel units)."""
    B, _, H, W = frame.shape
    # Build sampling grid: identity + normalized flow
    gy, gx = torch.meshgrid(
        torch.arange(H, device=frame.device, dtype=frame.dtype),
        torch.arange(W, device=frame.device, dtype=frame.dtype),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
    # flow: [B, 2, H, W] in pixel units -> normalized [-1, 1]
    flow_norm = flow.clone()
    flow_norm[:, 0] = flow_norm[:, 0] / (W / 2.0)
    flow_norm[:, 1] = flow_norm[:, 1] / (H / 2.0)
    sample_grid = (grid + flow_norm).permute(0, 2, 3, 1)  # [B, H, W, 2]
    # Normalize to [-1, 1]
    sample_grid[:, :, :, 0] = 2.0 * sample_grid[:, :, :, 0] / (W - 1) - 1.0
    sample_grid[:, :, :, 1] = 2.0 * sample_grid[:, :, :, 1] / (H - 1) - 1.0
    return F.grid_sample(frame, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


class RendererUNet(nn.Module):
    """UNet that predicts optical flow from I_{t-1}, conditioned on E_t, L, and CLIP patches."""

    def __init__(
        self,
        in_ch: int = 3,
        embed_dim: int = 512,
        base_ch: int = 64,
        ch_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        max_flow_px: float = 16.0,
    ):
        super().__init__()
        self.cond_dim = embed_dim * 2
        self.max_flow_px = max_flow_px

        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.patch_proj = nn.Linear(embed_dim, embed_dim)

        chs = [base_ch]
        self.down_blocks = nn.ModuleList()
        cur_ch = base_ch
        for mult in ch_mults:
            enc_ch = base_ch * mult
            self.down_blocks.append(DownBlock(cur_ch, enc_ch, self.cond_dim, num_res_blocks))
            chs.append(enc_ch)
            cur_ch = enc_ch

        self.bottleneck = nn.ModuleList(
            [ResBlock(cur_ch, self.cond_dim) for _ in range(num_res_blocks)]
        )
        self.bottleneck_cross = CrossAttnBlock(cur_ch, embed_dim)

        self.up_blocks = nn.ModuleList()
        use_ca = [True, True, False, False]
        for i in range(len(ch_mults)):
            skip_ch = chs[-(i + 2)]
            self.up_blocks.append(
                UpBlock(cur_ch, skip_ch, skip_ch, self.cond_dim, embed_dim,
                        num_res_blocks, use_cross_attn=use_ca[i])
            )
            cur_ch = skip_ch

        self.flow_head = FlowHead(cur_ch, max_flow_px)

    def forward(
        self,
        prev_frame: torch.Tensor,
        embed_t: torch.Tensor,
        label_embed: torch.Tensor,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_frame:  I_{t-1}  [B, 3, H, W]
            embed_t:     E_t      [B, 512]
            label_embed: L        [B, 512]
            patches:     CLIP spatial patches [B, 49, 512]

        Returns:
            flow: [B, 2, H, W]  optical flow in pixel units; warp I_{t-1} to get I_t
        """
        B = prev_frame.shape[0]

        cond = torch.cat([embed_t, label_embed], dim=-1)
        cond = self.cond_proj(cond)
        patches = self.patch_proj(patches)

        x = self.input_conv(prev_frame)
        skips = [x]
        for down in self.down_blocks:
            x = down(x, cond)
            skips.append(x)

        for block in self.bottleneck:
            x = block(x, cond)
        x = self.bottleneck_cross(x, patches)

        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[-(i + 2)], cond, patches)

        flow = self.flow_head(x)
        return flow
