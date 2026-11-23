import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.degradations import apply_degradation, random_degradation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepfakeDataset(Dataset):
    def __init__(self, manifest, size=224, train=False, degrade="none",
                 train_degrade=True, degrade_prob=0.5):
        self.rows = []
        with open(manifest, newline="") as f:
            import csv

            self.rows = list(csv.DictReader(f))
        self.size = size
        self.train = train
        self.degrade = degrade
        self.train_degrade = train_degrade
        self.degrade_prob = degrade_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        if self.train:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            if self.train_degrade and np.random.rand() < self.degrade_prob:
                img = random_degradation(img)
        elif self.degrade != "none":
            img = apply_degradation(img, self.degrade)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        x = torch.from_numpy(img.transpose(2, 0, 1)).float()
        y = int(row["label"])
        return x, y
