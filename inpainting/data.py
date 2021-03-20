import os

import numpy as np
from skimage import io, color

import torch
from torch.utils.data import Dataset


class BBox:
    def __init__(self, top, left, bottom, right):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right


def random_bbox(height: int, width: int) -> BBox:
    """Generates random bounding box by choosing (left, right) and (top, bottom) pairs uniformly
    """
    bottom, top = np.sort(np.random.choice(height, size=2, replace=False))
    left, right = np.sort(np.random.choice(width, size=2, replace=False))

    return BBox(top, left, bottom, right)


def random_bbox_fixed(height: int, width: int, input_shape: (int, int)) -> BBox:
    """Generates random bounding box of fixed size by choosing center point first
    """
    center_x = np.random.randint(height // 2, input_shape[0] - height // 2)
    center_y = np.random.randint(width // 2, input_shape[1] - width // 2)

    return BBox(center_x + height // 2, center_y - width // 2, center_x - height // 2, center_y + width // 2)


def bbox2mask(input_shape: (int, int), bbox: BBox) -> torch.FloatTensor:
    """Converts bounding box to torch tensor
    """
    out = torch.FloatTensor(input_shape)
    out[bbox.bottom:bbox.top, bbox.left:bbox.right] += 1

    return out


class ImageDataset(Dataset):
    def __init__(self, path: str, image_shape: (int, int) = (256, 256)):
        self.path = path
        self.filenames = []

        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.jpg'):
                    self.filenames.append(
                        os.path.join(root, filename)
                    )
        self.image_shape = image_shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = io.imread(self.filenames[idx]) / 255

        if len(image.shape) == 2:
            image = color.gray2rgb(image)

        return np.moveaxis(image, -1, 0).astype(float)
