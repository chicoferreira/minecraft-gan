import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchmetrics.image import StructuralSimilarityIndexMeasure, FrechetInceptionDistance


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


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device, layer_weights=None):
        super().__init__()
        vgg = models.vgg16().features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        # choose layers to extract from
        self.layer_ids = [3, 8, 15, 22]
        self.weights = layer_weights or [1.0, 1.0, 1.0, 1.0]
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        x = torch.cat([pred, target], dim=0)
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layer_ids:
                feats.append(x)
        # split back into pred / target
        for i, feat in enumerate(feats):
            p_feat, t_feat = torch.chunk(feat, 2, dim=0)
            loss += self.weights[i] * self.criterion(p_feat, t_feat)
        return loss
