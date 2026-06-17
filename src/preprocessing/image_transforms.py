from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image_tensor(path: str | Path, image_size: int = 224) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
        offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
        canvas.paste(image, offset)
        array = np.asarray(canvas, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std
