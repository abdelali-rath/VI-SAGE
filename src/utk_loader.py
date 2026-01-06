import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class UTKFaceMultiTask(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []

        # check if directory exists
        if not os.path.exists(root):
            raise RuntimeError(f"Dataset path {root} does not exist!")

        # Read filenames
        for filename in os.listdir(root):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                # filename parsen ?
                parts = filename.split('_')
                if len(parts) >= 4:
                    self.images.append(filename)

        print(f"-> Loaded {len(self.images)} images from {root}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)

        # Load images
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            # Fallback
            return self.__getitem__((idx + 1) % len(self.images))

        if self.transform:
            image = self.transform(image)

        # Labels parsen
        # Format: [age]_[gender]_[race]_[date].jpg
        parts = filename.split('_')
        
        try:
            age = float(parts[0])       # float for regression
            gender = int(parts[1])      # 0=Male, 1=Female
            ethnicity = int(parts[2])   # 0..4
        except ValueError:
            # skip wrong filenames
            return self.__getitem__((idx + 1) % len(self.images))

        # Give dictionary back
        targets = {
            'age': torch.tensor(age, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'ethnicity': torch.tensor(ethnicity, dtype=torch.long)
        }

        return image, targets