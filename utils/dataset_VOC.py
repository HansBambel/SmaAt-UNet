import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


class VOCSegmentation(Dataset):
    CLASS_NAMES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted-plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    def __init__(self, root: Path, image_set="train", transformations=None, augmentations=False):
        super().__init__()
        assert image_set in ["train", "val", "trainval"]

        voc_root = root / "VOC2012"
        image_dir = voc_root / "JPEGImages"
        mask_dir = voc_root / "SegmentationClass"

        splits_dir = voc_root / "ImageSets" / "Segmentation"

        split_f = splits_dir / (image_set.rstrip("\n") + ".txt")
        with open(split_f) as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [image_dir / (x + ".jpg") for x in file_names]
        self.masks = [mask_dir / (x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

        self.transformations = transformations
        self.augmentations = augmentations

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transformations is not None:
            img = self.transformations(img)
            target = self.transformations(target)
        # Apply augmentations to both input and target
        if self.augmentations:
            img, target = self.apply_augmentations(img, target)

        # Convert the RGB image to a tensor
        toTensorTransform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        img = toTensorTransform(img)
        # Convert target to long tensor
        target = torch.from_numpy(np.array(target)).long()
        target[target == 255] = 0

        return img, target

    def __len__(self):
        return len(self.images)

    def apply_augmentations(self, img, target):
        # Horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            target = TF.hflip(target)
        # Random Rotation (clockwise and counterclockwise)
        if random.random() > 0.5:
            degrees = 10
            if random.random() > 0.5:
                degrees *= -1
            img = TF.rotate(img, degrees)
            target = TF.rotate(target, degrees)
        # Brighten or darken image (only applied to input image)
        if random.random() > 0.5:
            brightness = 1.2
            if random.random() > 0.5:
                brightness -= 0.4
            img = TF.adjust_brightness(img, brightness)
        return img, target
