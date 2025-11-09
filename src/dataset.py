"""
Smaaaaaaaall Dataset loader for faces. Expects pre-cropped face images in folders.
Not for actual training
"""

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class FaceFolderDataset(Dataset):
    """
    Simple dataset: images in a folder. Labels must be provided as an optional dict filename->label.
    For now labels=None.
    """
    def __init__(self, img_dir: str, labels: dict = None, transform=DEFAULT_TRANSFORMS):
        self.img_dir = img_dir
        self.files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform
        self.labels = labels or {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        label = self.labels.get(fname, {})  # could be dict(age, gender, ethnicity)
        return x, label
