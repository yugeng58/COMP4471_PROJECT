# Image-to-Video Generation via Decoupled Latent Flow

Decoupled I2V: predict motion in CLIP embedding space (Flow Model), then render to
pixels (Renderer). Current status: **Flow Model works, Renderer does not generate
meaningful motion.** See "What Went Wrong" below.

## Architecture (current)

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
       [FiLM + Cross-Attn UNet] ← Renderer (I_{t-1} + Ê_t + L + patch tokens → I_t)
              │
              ▼
      I_1, I_2, ..., I_n       ← output video frames
```

- **Image Encoder**: frozen CLIP ViT-B/32, image → 512-dim global + 49×512 patch tokens
- **Label Encoder**: frozen CLIP Text, action label → 512-dim semantic embedding
- **Flow Model**: GPT-style causal Transformer, `[L, E_0, ..., E_{t-1}]` → `Ê_t`
- **Renderer**: lightweight UNet with FiLM (global) + cross-attention (spatial) conditioning

## Setup

```bash
# Python 3.12+, PyTorch 2.6+ with CUDA
pip install -r requirements.txt
```

## Dataset

[HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
(51 classes, 6,754 videos) downloaded from HuggingFace mirror.

```bash
python -m data.download_hmdb51     # ~2GB extracted
python -m data.preprocess          # pre-extract CLIP embeddings (242MB .pt)
```

## Training

```bash
python -m train.train_flow         # Flow Model (cosine loss)
python -m train.train_renderer     # Renderer (L1 + contrastive)
```

Checkpoints saved to `checkpoints/` every 10 epochs.

## Inference

```bash
python inference.py --image test_walk.jpg --label "walking" --num_frames 16 \
    --flow_ckpt flow_cos_epoch40.pt --renderer_ckpt renderer_ca_epoch20.pt --out output.mp4
```

Available labels: `brush hair`, `cartwheel`, `catch`, `chew`, `clap`, `climb`,
`climb stairs`, `dive`, `draw sword`, `dribble`, `drink`, `eat`, `fall floor`,
`fencing`, `flic flac`, `golf`, `handstand`, `hit`, `hug`, `jump`, `kick`,
`kick ball`, `kiss`, `laugh`, `pick`, `pour`, `pullup`, `punch`, `push`,
`pushup`, `ride bike`, `ride horse`, `run`, `shake hands`, `shoot ball`,
`shoot bow`, `shoot gun`, `sit`, `situp`, `smile`, `smoke`, `somersault`,
`stand`, `swing baseball`, `sword`, `sword exercise`, `talk`, `throw`,
`turn`, `walk`, `wave`

## Project structure

```
├── config.py                  # All hyperparameters
├── inference.py               # End-to-end I2V inference
├── final_report.tex/pdf       # Course report (CVPR 2026 template)
├── requirements.txt
├── data/
│   ├── dataset.py             # HMDB51FrameDataset + HMDB51EmbeddingDataset
│   ├── download_hmdb51.py     # HF mirror download
│   ├── preprocess.py          # CLIP embedding pre-extraction
│   └── hmdb51/                # Raw videos + processed/embeddings_n16.pt
├── models/
│   ├── encoders.py            # ImageEncoder (global + patches) + LabelEncoder
│   ├── flow_model.py          # CausalFlowModel (TransformerEncoder, causal mask)
│   └── renderer.py            # RendererUNet (FiLM + CrossAttn blocks)
└── train/
    ├── train_flow.py          # Flow Model training (cosine similarity loss)
    └── train_renderer.py      # Renderer training (L1 + contrastive conditioning loss)
```

## What works

### Flow Model

- **Shifted teacher forcing** (GPT-style next-token prediction) eliminates the
  train-inference gap vs zero-placeholder inputs.
- **Cosine similarity loss** (vs MSE) — MSE loss causes the Flow Model to collapse:
  predicted embeddings differ by 1-cos ≈ 0.0005, only 2% of ground-truth motion.
  Cosine loss restores this to 82.5% (1-cos ≈ 0.024 vs GT 0.029).
- Best checkpoint: `checkpoints/flow_cos_epoch40.pt`

```
                    MSE loss         Cosine loss
Inter-frame 1-cos   0.0006 (2.0%)    0.0239 (82.5% of GT)
```

## What went wrong

### Renderer — embedding is ignored

The Renderer (trained to predict frame-to-frame residuals) produces near-identical
outputs regardless of the conditioning embedding. All three architectures tried
show the same behavior:

| Renderer variant | mag(E_t) / mag(E_0) | Result |
|---|---|---|
| FiLM-only + L1 | ~1.0x | no response to embedding |
| FiLM + Cross-Attn + L1 | ~1.0x | no response to embedding |
| FiLM + Cross-Attn + L1 + Contrastive | ~0.98x | no response to embedding |

**Root cause diagnosis:**

1. **CLIP embeddings barely change between adjacent frames** — 1-cos ≈ 0.024 for
   frame-to-frame transitions. Flow Model predicts embedding trajectories with
   82.5% of this tiny variation, but the absolute signal is still very small.

2. **L1 residual loss encourages "predict zero"** — the optimal strategy to
   minimize L1(ΔI_pred, I_t - I_{t-1}) is to output a near-zero residual, since
   the ground-truth residual is tiny (mean ≈ 0.005 at 224×224).

3. **FiLM can't express spatial guidance** — FiLM applies channel-wise
   gamma/beta modulation globally (same for all spatial positions). This can
   adjust brightness/contrast but cannot tell the model "move the hand up-right
   and keep the background still."

4. **Cross-attention doesn't help** — replacing/addding cross-attention to CLIP
   patch tokens (7×7 spatial grid) didn't fix the collapse because the training
   objective still penalizes any output larger than the tiny GT residual.

5. **Contrastive conditioning loss doesn't fix the magnitude problem** — adding a
   term that matches `cos(res_real, res_fake)` to `cos(E_real, E_fake)` only
   controls the *direction* of residuals, not their magnitude. The model learns to
   produce subtly different near-zero outputs for different conditions.

### Possible next directions

- **Train with larger frame gaps** — use $(I_t, I_{t+k})$ with $k=4,8,16$ so
  the GT residual is naturally larger and the embedding difference more meaningful
- **Perceptual loss** — VGG or CLIP feature matching instead of pixel L1
- **Direct frame generation** — predict $I_t$ directly (not residual) with a
  GAN or diffusion-like objective
- **Optical flow supervision** — predict an explicit motion field as an
  intermediate representation
- **Spatial embedding features** — use DINOv2 or other dense-prediction
  embeddings instead of CLIP global vectors
- **Joint end-to-end training** — backprop through Flow Model into Renderer
