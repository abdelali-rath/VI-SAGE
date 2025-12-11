import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class UTKFaceGender(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.img_paths = []
        self.labels = []

        for f in os.listdir(root):
            if not f.endswith(".jpg"):
                continue

            parts = f.split("_")

            if len(parts) < 4:
                continue

            try:
                gender = int(parts[1])  # 0 = male, 1 = female
            except:
                continue

            self.img_paths.append(os.path.join(root, f))
            self.labels.append(gender)

        assert len(self.img_paths) == len(self.labels), "Mismatch img_paths / labels length!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
