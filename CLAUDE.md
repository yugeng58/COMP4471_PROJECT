# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

COMP4471 (Deep Learning for Computer Vision) course project — Image-to-Video generation via
decoupled latent flow. Given a static image + action label, generate a short video by:
1. Predicting CLIP embedding trajectory in semantic space (Flow Model — causal Transformer)
2. Rendering each predicted embedding to a pixel frame via image-to-image UNet (Renderer)

Both stages conditioned on frozen CLIP text embedding of the action label.

## Commands

```bash
# Python: C:\Users\PhilipZhu\AppData\Local\Programs\Python\Python312\python.exe
# GPU: RTX 4060 Laptop (8GB), PyTorch 2.6+cu124

# Data preparation
python -m data.download_hmdb51      # HF mirror → data/hmdb51/
python -m data.preprocess           # Pre-extract CLIP embeddings → data/hmdb51/processed/

# Training (order matters — Flow Model first, then Renderer)
python -m train.train_flow          # Loads pre-extracted embeddings, no CLIP in workers
python -m train.train_renderer      # Uses HMDB51FrameDataset, CLIP called in main process

# Inference
python inference.py --image path/to/frame.jpg --label "walking" --out output.mp4

# Compile report
pdflatex final_report.tex
```

## Architecture

**Models (all in `models/`):**
- `encoders.py` — `ImageEncoder` (CLIP ViT-B/32, image→512) and `LabelEncoder` (CLIP text, label→512), both frozen
- `flow_model.py` — `CausalFlowModel`, 4-layer TransformerEncoder, shifted teacher forcing: `[L, E_0..E_{n-1}]` → `[Ê_1..Ê_n]`. Has both `forward()` (training with real history) and `generate()` (autoregressive inference)
- `renderer.py` — `RendererUNet`, 4 down/up blocks with FiLM conditioning on `[E_t; L]`, predicts residual ΔI_t added to I_{t-1}

**Training pipeline warts:**
- Flow Model training loads pre-extracted embeddings from disk (`data/hmdb51/processed/embeddings_n16.pt`, 242MB) — the old `HMDB51EmbeddingDataset` that ran CLIP in `__getitem__` was scrapped because CUDA models can't be serialized to DataLoader workers
- Renderer training uses `HMDB51FrameDataset` (loads raw frames, no CUDA), encodes with CLIP in the main training loop — this is fine
- Both models frozen at init; separate training scripts, no joint fine-tuning (yet)
- Checkpoints are ~149MB each (include optimizer state)

**Data pipeline:**
- `data/dataset.py` — `HMDB51FrameDataset` (random frame pairs) and `HMDB51EmbeddingDataset` (loads pre-extracted .pt cache)
- `data/preprocess.py` — iterates all 6,754 videos, encodes with CLIP, saves to disk
- Videos expected at `data/hmdb51/{class_name}/{video_name}.avi` (51 class dirs)

**Key config values (in `config.py`):**
- `NUM_FRAMES=16`, `IMAGE_SIZE=224`, `EMBED_DIM=512`
- `FLOW_EPOCHS=100` (final loss ~0.0116), `RENDERER_EPOCHS=100`
- `BATCH_SIZE=8`, `GRAD_ACCUM_STEPS=2` (effective batch 16)
- `CHECKPOINT_DIR` = `checkpoints/`

**Report:** `final_report.tex` uses CVPR 2026 template, requires `cvpr.sty` (downloaded to project root).
