"""
U-Net 2D estándar para predicción de dosis.
- Bilinear upsample + conv 3×3 en lugar de transposed convolutions
  para evitar artefactos de cuadriculado (checkerboard).
- Sin BatchNorm (los volúmenes de validación van uno a uno).
- GroupNorm como alternativa estable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Bloque conv-norm-act × 2."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample con maxpool + DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsample con bilinear + conv 1×1 + DoubleConv. Evita checkerboard."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # Ajuste por si las dimensiones difieren por 1 pixel
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    U-Net 2D para predicción de dosis.
    Entrada: (B, in_channels, H, W)
    Salida:  (B, 1, H, W)  → dosis en % de prescripción (sin activación final)
    """
    def __init__(self, in_channels: int = 5, out_channels: int = 1,
                 base_features: int = 32, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        assert depth >= 2
        self.depth = depth

        # Encoder
        self.inc = DoubleConv(in_channels, base_features, dropout=dropout)
        self.downs = nn.ModuleList()
        ch = base_features
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, dropout=dropout))
            ch *= 2

        # Decoder
        self.ups = nn.ModuleList()
        for i in range(depth):
            in_ch  = ch
            skip_ch = ch // 2
            out_ch = ch // 2
            self.ups.append(Up(in_ch, skip_ch, out_ch, dropout=dropout))
            ch = out_ch

        self.outc = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # skips[-1] es el bottleneck; usamos skips[:-1] en orden inverso
        bottleneck = skips[-1]
        x = bottleneck
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = up(x, skip)
        return self.outc(x)


def build_model(cfg) -> nn.Module:
    """Factory que construye el modelo según la config."""
    arch = cfg.model.arch
    if arch == "unet2d":
        return UNet2D(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            base_features=cfg.model.base_features,
            depth=cfg.model.depth,
            dropout=cfg.model.dropout,
        )
    if arch == "unet3d":
        from src.models.unet3d import build_model_3d
        return build_model_3d(cfg)
    raise NotImplementedError(f"Arch {arch} no implementada todavía")
