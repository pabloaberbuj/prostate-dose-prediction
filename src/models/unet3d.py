"""
U-Net 3D para predicción de dosis — análogo volumétrico de unet2d.py.
Mismas decisiones de diseño que la serie 2D (ver CLAUDE_CODE_CONTEXT):
- Trilinear upsample + conv 1x1x1 en lugar de transposed convolutions
  (evita artefactos de cuadriculado).
- GroupNorm en vez de BatchNorm (batch=1 por VRAM).
- Gradient checkpointing opcional en cada bloque (activaciones no se
  guardan, se recomputan en el backward) — para entrar en 12GB de VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DoubleConv3D(nn.Module):
    """Bloque conv-norm-act x2, 3D."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """Downsample con maxpool 3D + DoubleConv3D."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Upsample con trilinear + conv 1x1x1 + DoubleConv3D. Evita checkerboard."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = DoubleConv3D(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    U-Net 3D para predicción de dosis.
    Entrada: (B, in_channels, D, H, W)  [D = Z, cortes axiales]
    Salida:  (B, 1, D, H, W)  → dosis en % de prescripción (sin activación final)

    grad_checkpointing=True envuelve cada bloque (inc, downs, ups) en
    torch.utils.checkpoint — reduce memoria de activaciones a costa de
    recomputar el forward de cada bloque durante el backward (~overhead
    de tiempo, no de memoria de parámetros/gradientes).
    """
    def __init__(self, in_channels: int = 5, out_channels: int = 1,
                 base_features: int = 16, depth: int = 4, dropout: float = 0.0,
                 grad_checkpointing: bool = False):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        self.grad_checkpointing = grad_checkpointing

        self.inc = DoubleConv3D(in_channels, base_features, dropout=dropout)
        self.downs = nn.ModuleList()
        ch = base_features
        for _ in range(depth):
            self.downs.append(Down3D(ch, ch * 2, dropout=dropout))
            ch *= 2

        self.ups = nn.ModuleList()
        for _ in range(depth):
            in_ch = ch
            skip_ch = ch // 2
            out_ch = ch // 2
            self.ups.append(Up3D(in_ch, skip_ch, out_ch, dropout=dropout))
            ch = out_ch

        self.outc = nn.Conv3d(ch, out_channels, kernel_size=1)

    def _run_block(self, block, *inputs):
        if self.grad_checkpointing and self.training:
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, x):
        skips = []
        x = self._run_block(self.inc, x)
        skips.append(x)
        for down in self.downs:
            x = self._run_block(down, x)
            skips.append(x)
        bottleneck = skips[-1]
        x = bottleneck
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            x = self._run_block(up, x, skip)
        return self.outc(x)


def build_model_3d(cfg) -> nn.Module:
    """Factory análoga a unet2d.build_model, para configs con model.arch == 'unet3d'."""
    return UNet3D(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_features=cfg.model.base_features,
        depth=cfg.model.depth,
        dropout=cfg.model.dropout,
        grad_checkpointing=getattr(cfg.model, 'grad_checkpointing', False),
    )
