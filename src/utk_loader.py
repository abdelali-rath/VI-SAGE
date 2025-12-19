import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class UTKFaceMultiTask(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.img_paths = []
        self.gender_labels = []
        self.ethnicity_labels = []

        # UTKFace filename format: [age]_[gender]_[race]_[date].jpg
        for f in os.listdir(root):
            if not f.endswith(".jpg"):
                continue

            parts = f.split("_")

            # Ensure filename has enough parts to extract age, gender, and race
            if len(parts) < 4:
                continue

            try:
                # Index 1 is Gender (0=Male, 1=Female)
                gender = int(parts[1])
                
                # Index 2 is Ethnicity (0=White, 1=Black, 2=Asian, 3=Indian, 4=Other)
                race = int(parts[2])
                
                # Filter invalid ranges if necessary (e.g. race should be 0-4)
                if race < 0 or race > 4:
                    continue
                    
            except ValueError:
                continue

            self.img_paths.append(os.path.join(root, f))
            self.gender_labels.append(gender)
            self.ethnicity_labels.append(race)

        print(f"Loaded {len(self.img_paths)} images from {root}")
        assert len(self.img_paths) == len(self.gender_labels) == len(self.ethnicity_labels), \
            "Mismatch in data lists length!"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Load and convert image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Get labels
        gender_label = self.gender_labels[idx]
        ethnicity_label = self.ethnicity_labels[idx]

        # Return image and a dictionary of targets
        targets = {
            "gender": torch.tensor(gender_label, dtype=torch.long),
            "ethnicity": torch.tensor(ethnicity_label, dtype=torch.long)
        }

        return img, targets