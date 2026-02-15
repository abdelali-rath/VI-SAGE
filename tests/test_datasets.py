"""Tests for dataset classes (UTK age/gender parsing, etc.)."""

import pytest

try:
    from PIL import Image
    from src.training.train_age.train_age import UTKAgeDataset
except Exception:
    pytest.skip("torch/PIL or train_age not available", allow_module_level=True)


def test_utk_age_dataset_parses_filenames(tmp_path):
    """UTKAgeDataset should list only .jpg files with underscore and parse age from first segment."""
    # Create minimal valid JPEGs so Image.open does not fail
    tiny = Image.new("RGB", (1, 1), color=(0, 0, 0))
    tiny.save(tmp_path / "25_1_0_0.jpg")
    tiny.save(tmp_path / "30_0_1_0.jpg")
    (tmp_path / "plain.jpg").touch()  # excluded: no underscore
    (tmp_path / "40_1_0_1.png").touch()  # excluded: not .jpg

    ds = UTKAgeDataset(str(tmp_path), transform=None)
    assert len(ds) == 2
    _, age1 = ds[0]
    _, age2 = ds[1]
    ages = {age1.item(), age2.item()}
    assert ages == {25.0, 30.0}


def test_utk_age_dataset_getitem_returns_tensor_and_age(tmp_path):
    """Create a real image so transform can be applied; check return types."""
    from torchvision import transforms

    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    img.save(tmp_path / "20_1_0_0.jpg")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    ds = UTKAgeDataset(str(tmp_path), transform=transform)
    assert len(ds) == 1
    img_t, age_t = ds[0]
    assert img_t.shape == (3, 224, 224)
    assert age_t.item() == 20.0
