import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


class SSIMLoss(nn.Module):
    def __init__(self, device, data_range=1.0):
        super().__init__()
        # using torchmetrics SSIM
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    def forward(self, pred, target):
        # torchmetrics returns similarity; we want a loss:
        return 1 - self.ssim(pred, target)
