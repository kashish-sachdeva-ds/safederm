"""PyTorch Dataset for SafeDerm.

Wraps a split CSV (train/val/test) and the consolidated images folder into
something a DataLoader can batch. Label encoding is fixed by ALL_CLASSES in
src/labels.py, so the same integer always means the same diagnosis across
every notebook and the eventual serving code.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.labels import ALL_CLASSES

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


class SkinLesionDataset(Dataset):
    """One row per image. Returns (transformed_image_tensor, label_index)."""

    def __init__(self, split_csv_path: Path, images_dir: Path, transform=None):
        self.df = pd.read_csv(split_csv_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img_path = self.images_dir / f"{row['image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = CLASS_TO_IDX[row["dx"]]
        return image, label

    def get_image_path(self, idx: int) -> Path:
        row = self.df.iloc[idx]
        return self.images_dir / f"{row['image_id']}.jpg"
