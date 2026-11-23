import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SRMFilter(nn.Module):
    """固定的高通/噪声残差滤波器，不参与训练。"""

    def __init__(self):
        super().__init__()
        k = torch.tensor(
            [
                [[0, 0, 0, 0, 0],
                 [0, -1, 2, -1, 0],
                 [0, 2, -4, 2, 0],
                 [0, -1, 2, -1, 0],
                 [0, 0, 0, 0, 0]],
                [[-1, 2, -2, 2, -1],
                 [2, -6, 8, -6, 2],
                 [-2, 8, -12, 8, -2],
                 [2, -6, 8, -6, 2],
                 [-1, 2, -2, 2, -1]],
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 1, -2, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)  # (3, 1, 5, 5)
        self.register_buffer("weight", k)

    def forward(self, gray):
        return F.conv2d(gray, self.weight, padding=2)


class FreqBranch(nn.Module):
    def __init__(self, in_ch=4, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, 2, 1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class DualStream(nn.Module):
    def __init__(self, pretrained=True, use_freq=True, use_gate=True, dropout=0.3):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.rgb_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.rgb = backbone

        self.use_freq = use_freq
        self.use_gate = use_gate
        if use_freq:
            self.srm = SRMFilter()
            self.freq = FreqBranch(4, 256)
            self.gate = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 1)
            )
            feat_dim = self.rgb_dim + 256
        else:
            feat_dim = self.rgb_dim

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x, return_parts=False):
        r = self.rgb(x)

        if not self.use_freq:
            logits = self.head(r)
            return (logits, {"gate": None}) if return_parts else logits

        gray = x.mean(dim=1, keepdim=True)

        spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        amp = torch.log1p(torch.abs(spec))
        amp = (amp - amp.mean(dim=(-2, -1), keepdim=True)) / (
            amp.std(dim=(-2, -1), keepdim=True) + 1e-6
        )

        res = self.srm(gray)
        f = self.freq(torch.cat([amp, res], dim=1))

        g = None
        if self.use_gate:
            hf = (gray - F.avg_pool2d(gray, 3, 1, 1)).abs().mean(
                dim=(1, 2, 3), keepdim=True
            )
            # 压成 (B,1)，避免与 (B,256) 相乘时广播成 4 维
            q = torch.log1p(hf.flatten(1) * 10.0)
            g = torch.sigmoid(self.gate(q))
            f = f * g

        logits = self.head(torch.cat([r, f], dim=1))
        if return_parts:
            return logits, {"gate": g}
        return logits
