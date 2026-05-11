"""HMDB51 dataset loader for Image-to-Video generation.

Each sample: a video clip → consecutive frame pairs + action label.
Flow Model receives: embedding sequence from encoder.
Renderer receives: (I_{t-1}, I_t) pairs with E_t and L as condition.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm


HMDB51_CLASSES = sorted(
    [
        "brush_hair", "cartwheel", "catch", "chew", "clap", "climb",
        "climb_stairs", "dive", "draw_sword", "dribble", "drink",
        "eat", "fall_floor", "fencing", "flic_flac", "golf",
        "handstand", "hit", "hug", "jump", "kick", "kick_ball",
        "kiss", "laugh", "pick", "pour", "pullup", "punch",
        "push", "pushup", "ride_bike", "ride_horse", "run",
        "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun",
        "sit", "situp", "smile", "smoke", "somersault",
        "stand", "swing_baseball", "sword", "sword_exercise",
        "talk", "throw", "turn", "walk", "wave",
    ]
)


def class_to_label(name: str) -> str:
    """Convert directory name to readable label, e.g. 'brush_hair' → 'brush hair'."""
    return name.replace("_", " ")


class HMDB51FrameDataset(Dataset):
    """
    Extracts frames from HMDB51 videos and returns consecutive frame pairs
    with associated action labels.

    Returns:
        prev_frame: I_{t-1}  [3, H, W] tensor
        cur_frame:  I_t      [3, H, W] tensor
        label:      action name string
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_frames: int = 16,
        image_size: int = 224,
        frame_stride: int = 2,
        pair_gap: int = 1,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.pair_gap = max(1, pair_gap)
        self.split = split

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

        self.samples = self._build_index()

    def _build_index(self) -> list[tuple[str, str, str]]:
        """Build list of (class_name, video_name, video_path) entries."""
        entries = []
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for video_file in class_dir.iterdir():
                if video_file.suffix in (".avi", ".mp4", ".mkv", ".webm"):
                    entries.append((class_dir.name, video_file.stem, str(video_file)))
        return entries

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, video_path: str) -> list[np.ndarray]:
        """Read all frames from a video, returning list of BGR numpy arrays."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            idx += 1
        cap.release()
        return frames

    def __getitem__(self, idx: int):
        class_name, video_name, video_path = self.samples[idx]
        label = class_to_label(class_name)

        frames = self._read_frames(video_path)
        needed = self.pair_gap + 1
        if len(frames) < needed:
            # Fallback for very short videos: duplicate last frame until enough
            while len(frames) < needed:
                frames.append(frames[-1])

        # Sample a consecutive pair from the video
        max_start = max(0, len(frames) - needed)
        start = np.random.randint(0, max_start + 1)
        prev = self.transform(frames[start])
        cur = self.transform(frames[start + self.pair_gap])

        return prev, cur, label


class HMDB51EmbeddingDataset(Dataset):
    """Loads pre-extracted embeddings from disk. No CUDA model in workers."""

    def __init__(self, cache_path: str | Path):
        self.data = torch.load(cache_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        return item["e_seq"].float(), item["label_embed"].float()
