import os

import numpy as np
from skimage import io

import torch
from torch.utils.data import Dataset, DataLoader


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
    center_x = np.random.choice(input_shape[0] - height // 2)
    center_y = np.random.choice(input_shape[1] - width // 2)

    return BBox(center_x + height // 2, center_y - width // 2, center_x - height // 2, center_y + width // 2)


def bbox2mask(input_shape: (int, int), bbox: BBox) -> torch.FloatTensor:
    """Converts bounding box to torch tensor
    """
    out = torch.FloatTensor(input_shape)
    out[bbox.bottom:bbox.top, bbox.left:bbox.right] += 1

    return out


class ImageDataset(Dataset):
    def __init__(self, path: str, image_shape: (int, int) = (256, 256), max_mask_size: int = 100):
        self.path = path
        self.filenames = os.listdir(path)
        self.image_shape = image_shape
        self.max_mask_size = max_mask_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.path, self.filenames[idx]))
        mask_size = np.random.choice(self.max_mask_size)
        mask = bbox2mask(self.image_shape, random_bbox_fixed(mask_size, mask_size, self.image_shape))
        sample = {
            'image': image,
            'mask': mask
        }
        return sample
