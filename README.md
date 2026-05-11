# Decoupled I2V via Latent Flow Prediction

COMP4471 Deep Learning for Computer Vision — course project.

Decoupled image-to-video generation: predict motion in CLIP embedding space (Flow
Model), then render to pixels (Renderer). **Flow Model works; Renderer fails across
all tested architectures. The primary root cause is that frozen CLIP ViT-B/32
embeddings lack the spatial information needed for pixel-level frame rendering.**

See `final_report.pdf` for the full paper with detailed failure analysis.

## Architecture

```
  I_0 ──→ [CLIP ViT] ──→ E_0 ──┐
                                  │
  label ──→ [CLIP Text] ──→ L ──┤
                                  │
              ┌───────────────────┘
              ▼
       [Causal Transformer]   ← Flow Model (cosine similarity loss)
              │
              ▼
      Ê_1, Ê_2, ..., Ê_n      ← predicted embedding trajectory
              │
              ▼
       [Conditioned UNet]      ← Renderer (FiLM / Cross-Attn / Flow pred)
              │
              ▼
      I_1, I_2, ..., I_n       ← output video frames
```

- **Image Encoder**: frozen CLIP ViT-B/32 → 512-dim global + 49×512 patch tokens
- **Label Encoder**: frozen CLIP Text → 512-dim semantic embedding
- **Flow Model**: 4-layer causal Transformer (12.9M params), shifted teacher forcing
- **Renderer**: UNet with multiple conditioning variants tested (40–43M params)

## Setup

```bash
pip install -r requirements.txt   # Python 3.12+, PyTorch 2.6+ CUDA
```

## Dataset

HMDB51 (51 classes, 6,754 videos) from HuggingFace mirror.

```bash
python -m data.download_hmdb51     # ~2GB extracted
python -m data.preprocess          # pre-extract CLIP embeddings (242MB)
```

## Commands

```bash
python -m train.train_flow         # Train Flow Model (cosine loss)
python -m train.train_renderer     # Train Renderer

# Inference
python inference.py --image test_walk.jpg --label "walking" \
    --flow_ckpt flow_cos_epoch40.pt --out output.mp4
```

Checkpoints saved to `checkpoints/` every 10 epochs. Flow training ~4h, Renderer training ~7h on RTX 4060 Laptop 8GB.

## What Worked

### Flow Model — Trajectory Prediction

| Metric | MSE Loss | Cosine Loss |
|---|---|---|
| Final loss | 0.0116 | 0.0264 |
| Inter-frame 1-cos (predicted) | 0.0006 | 0.0239 |
| % of GT motion preserved | 2.0% | **82.5%** |

MSE loss collapsed to near-constant predictions. Cosine similarity loss fixed the
collapse. Shifted teacher forcing (GPT-style next-token prediction) aligned
training with autoregressive inference.

Best checkpoint: `checkpoints/flow_cos_epoch40.pt`

## What Failed

### Renderer — Embedding Condition Ignored

Four Renderer variants tested; all converged to ignoring the conditioning signal:

| Variant | Conditioning Sensitivity | Result |
|---|---|---|
| 1. FiLM + L1 residual | 1.03x | no response to embedding |
| 2. + Cross-Attention (CLIP patches) | 0.98x | no response |
| 3. + Contrastive conditioning loss | 0.98x | no response |
| 4. + Optical flow prediction + Farneback | 1.01x | flow collapses to zero |

### Primary Root Cause: CLIP Embeddings Lack Spatial Information

CLIP ViT-B/32 is trained for image-text semantic alignment. It encodes *what* is in
an image but discards *where* things are — spatial configuration, pose, dense
correspondence. Adjacent frames differ by only 1-cos ≈ 0.024 in CLIP space.
This tiny difference cannot carry enough information for pixel-level rendering,
regardless of architecture or loss function.

Supporting evidence: purely autoregressive Transformer video generation (e.g.,
VideoGPT, TATS) works when using *learned* VQ-VAE tokens trained jointly for
reconstruction, but fails when using *frozen* CLIP embeddings.

### Contributing Factors

- **L1 loss rewards near-identity output**: ΔI ≈ 0 already gives low loss on
  adjacent-frame pairs (mean residual ≈ 0.005 at 224×224)
- **FiLM cannot route spatial information**: channel-wise modulation applies
  uniformly across all positions — can adjust brightness, not move a hand

## Project Structure

```
├── config.py
├── inference.py
├── final_report.tex / .pdf
├── data/
│   ├── dataset.py          # HMDB51FrameDataset + EmbeddingDataset
│   ├── download_hmdb51.py  # HuggingFace mirror
│   └── preprocess.py       # CLIP embedding pre-extraction
├── models/
│   ├── encoders.py         # ImageEncoder (global + patches) + LabelEncoder
│   ├── flow_model.py       # CausalFlowModel (TransformerEncoder)
│   └── renderer.py         # RendererUNet (FiLM + CrossAttn + FlowHead)
└── train/
    ├── train_flow.py       # Flow Model training
    └── train_renderer.py   # Renderer training
```
