"""Pre-extract CLIP embeddings for all HMDB51 videos and save to disk.

This avoids passing CUDA models through DataLoader workers.
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from config import DATA_DIR, EMBED_DIM, IMAGE_SIZE, NUM_FRAMES
from data.dataset import HMDB51FrameDataset, class_to_label
from models.encoders import ImageEncoder, LabelEncoder


def preprocess():
    device = torch.device("cuda")
    output_dir = Path(DATA_DIR) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    img_enc = ImageEncoder().to(device)
    lbl_enc = LabelEncoder().to(device)

    ds = HMDB51FrameDataset(DATA_DIR, num_frames=NUM_FRAMES, image_size=IMAGE_SIZE)

    # Group by (class_name, video_name) to process each video once
    videos: dict[tuple[str, str], str] = {}
    for class_name, video_name, video_path in ds.samples:
        videos[(class_name, video_name)] = video_path

    print(f"Processing {len(videos)} unique videos...")
    all_embeddings = []

    for (class_name, video_name), video_path in tqdm(videos.items()):
        frames_raw = ds._read_frames(video_path)
        if len(frames_raw) < NUM_FRAMES + 1:
            while len(frames_raw) < NUM_FRAMES + 1:
                frames_raw.append(frames_raw[-1])

        # Sample NUM_FRAMES+1 consecutive frames from the middle
        max_start = max(0, len(frames_raw) - (NUM_FRAMES + 1))
        start = max_start // 2  # take from middle for consistency
        segment = frames_raw[start : start + NUM_FRAMES + 1]

        tensor_frames = torch.stack([ds.transform(f) for f in segment]).to(device)

        with torch.no_grad():
            e_seq = img_enc(tensor_frames)  # [n+1, 512]
            L = lbl_enc([class_to_label(class_name)])  # [1, 512]

        all_embeddings.append({
            "class": class_name,
            "video": video_name,
            "label_embed": L.squeeze(0).cpu(),
            "e_seq": e_seq.cpu(),
        })

    # Save
    output_path = output_dir / f"embeddings_n{NUM_FRAMES}.pt"
    torch.save(all_embeddings, output_path)
    print(f"Saved {len(all_embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    preprocess()
