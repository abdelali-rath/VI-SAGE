"""
Small training script (debug / quickstart).
Just constructs the MultiTaskModel and runs a few epochs on the provided dataset folder (FaceFolderDataset).
"""

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import MultiTaskModel
from dataset import FaceFolderDataset
import os


def train_loop(model, dataloader, opt, device):
    model.train()
    total = 0
    for images, _ in dataloader:
        images = images.to(device)
        # dummy targets for quick debug
        age_target = torch.zeros(images.size(0), device=device)
        gender_target = torch.zeros(images.size(0), dtype=torch.long, device=device)
        eth_target = torch.zeros(images.size(0), dtype=torch.long, device=device)

        out = model(images)
        loss_age = F.l1_loss(out["age"], age_target)
        loss_gender = F.cross_entropy(out["gender_logits"], gender_target)
        loss_eth = F.cross_entropy(out["ethnicity_logits"], eth_target)

        loss = loss_age + loss_gender + loss_eth
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += images.size(0)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/faces")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    ds = FaceFolderDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5)
    model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(args.epochs):
        n = train_loop(model, dl, opt, args.device)
        print(f"Epoch {ep+1}/{args.epochs} trained on {n} images.")

    ck_dir = "checkpoints"
    os.makedirs(ck_dir, exist_ok=True)
    from src.models import save_checkpoint
    save_checkpoint(model, os.path.join(ck_dir, "visage_debug.pt"))
    print("Saved checkpoint to checkpoints/visage_debug.pt")


if __name__ == "__main__":
    main()
