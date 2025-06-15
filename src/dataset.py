from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root = Path(root_dir)
        self.vanilla_dir = self.root / "before"
        self.shader_dir = self.root / "after"
        # Dataset structure
        # generator_bot/screenshots/before/before_0001.png
        # generator_bot/screenshots/after/after_0001.png

        prefix_len = len("before_")
        suffix_len = len(".png")

        self.files = sorted([f.name[prefix_len:-suffix_len] for f in self.vanilla_dir.iterdir() if f.suffix in [".png", ".jpg"]])
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        van = Image.open(self.vanilla_dir / f"before_{fname}.png").convert("RGB")
        shd = Image.open(self.shader_dir / f"after_{fname}.png").convert("RGB")
        if self.transforms:
            van = self.transforms(van)
            shd = self.transforms(shd)
        return {"vanilla": van, "shader": shd}
