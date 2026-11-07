"""Optional stretch model: RGB backbone + FFT log-magnitude branch, fused by a linear head."""

from __future__ import annotations

import timm
import torch
from torch import nn


def fft_log_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Bx3xHxW normalized RGB -> Bx1xHxW centred log-magnitude spectrum, standardized."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        gray = x.float().mean(dim=1, keepdim=True)
        spec = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(spec.abs())
        mean = mag.mean(dim=(-2, -1), keepdim=True)
        std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (mag - mean) / std


class FFTBranch(nn.Module):
    def __init__(self, width: int = 32, out_features: int = 128) -> None:
        super().__init__()
        layers, c = [], 1
        for w in (width, width * 2, width * 4, out_features):
            layers += [
                nn.Conv2d(c, w, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
            ]
            c = w
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(fft_log_magnitude(x))


class RGBFFTNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True, drop_rate: float = 0.2) -> None:
        super().__init__()
        self.rgb = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.fft = FFTBranch()
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.rgb.num_features + self.fft.out_features, 1)
        self.pretrained_cfg = self.rgb.pretrained_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([self.rgb(x), self.fft(x)], dim=1)
        return self.head(self.dropout(feats))
