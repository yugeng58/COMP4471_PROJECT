from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "hmdb51"
CHECKPOINT_DIR = ROOT / "checkpoints"

# --- Dataset ---
NUM_FRAMES = 16
IMAGE_SIZE = 224
FRAME_STRIDE = 2  # sample every N-th frame

# --- CLIP encoders (frozen) ---
CLIP_MODEL = "openai/clip-vit-base-patch32"
EMBED_DIM = 512  # CLIP ViT-B/32 output dim

# --- Flow Model ---
FLOW_NUM_LAYERS = 4
FLOW_NUM_HEADS = 8
FLOW_DROPOUT = 0.1
FLOW_LR = 1e-4
FLOW_EPOCHS = 100

# --- Renderer UNet ---
UNET_BASE_CH = 64
UNET_CH_MULTS = (1, 2, 4, 8)
UNET_NUM_RES_BLOCKS = 2
RENDERER_LR = 1e-4
RENDERER_EPOCHS = 100

# --- Training ---
BATCH_SIZE = 8
NUM_WORKERS = 4
GRAD_ACCUM_STEPS = 2
LOG_EVERY = 50
SAVE_EVERY = 10
