import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class UTKFaceGender(Dataset):
    """
    Dataset for gender classification based on the UTKFace filename scheme.

    Expected filenames:
        age_gender_race_*.jpg

    Only the second field (gender) is used as the label.
    """

    def __init__(self, root, transform=None):
        """
        Args:
            root: Directory containing the UTKFace images.
            transform: Optional torchvision transform for the images.
        """
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

        assert len(self.img_paths) == len(
            self.labels
        ), "Mismatch img_paths / labels length!"

    def __len__(self):
        """Total number of samples in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Load a single sample.

        Returns:
            tuple[Tensor, Tensor]: (image tensor, label as long tensor).
        """
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
