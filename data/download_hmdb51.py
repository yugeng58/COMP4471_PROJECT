"""Download and extract HMDB51 from HuggingFace mirror.

Source: https://huggingface.co/datasets/jili5044/hmdb51
"""

import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

HF_REPO = "jili5044/hmdb51"
HF_FILE = "hmdb51.zip"


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract zip and organize into class/video_name.avi structure."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, dest_dir)

    # Flatten nested structure: move everything from inner dirs up
    avi_files = list(dest_dir.rglob("*.avi"))
    if avi_files and avi_files[0].parent != dest_dir:
        # Find the deepest common ancestor that contains class dirs
        for item in list(dest_dir.iterdir()):
            if item.is_dir() and item.name != dest_dir.name:
                inner_avis = list(item.rglob("*.avi"))
                if inner_avis:
                    print(f"Moving {len(inner_avis)} files from '{item.name}/' up...")
                    for sub in item.iterdir():
                        target = dest_dir / sub.name
                        if not target.exists():
                            shutil.move(str(sub), str(target))
                    if not list(item.iterdir()):
                        item.rmdir()

    avi_files = list(dest_dir.rglob("*.avi"))
    print(f"Total videos: {len(avi_files)}")
    classes = set()
    for f in avi_files:
        rel = f.relative_to(dest_dir)
        if len(rel.parts) > 1:
            classes.add(rel.parts[0])
    print(f"Classes: {len(classes)}")


def main():
    data_dir = Path(__file__).parent / "hmdb51"

    # Check if already extracted
    existing_avis = list(data_dir.rglob("*.avi"))
    if existing_avis:
        classes = set()
        for f in existing_avis:
            rel = f.relative_to(data_dir)
            if len(rel.parts) > 1:
                classes.add(rel.parts[0])
        print(f"Already have {len(existing_avis)} videos across {len(classes)} classes.")
        return

    # Download from HuggingFace
    print(f"Downloading {HF_FILE} from {HF_REPO}...")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
    )
    zip_path = Path(zip_path)
    print(f"Downloaded to {zip_path}")

    # Extract
    print("Extracting...")
    extract_zip(zip_path, data_dir)

    # Clean up zip
    zip_path.unlink()
    print(f"Done! Removed {zip_path.name}")


if __name__ == "__main__":
    main()
