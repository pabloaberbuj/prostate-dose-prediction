"""
Sanity check: entrenamiento mínimo de un MLP sobre datos sintéticos
para verificar que el setup (Lightning + W&B + GPU + mixed precision)
funciona en la máquina actual.

Uso:
    python scripts/hello_world.py
    python scripts/hello_world.py --no-wandb   # sin logging a W&B

Esperado: completa 5 epochs sin errores, GPU detectada, loss baja.
"""

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DummyMLP(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def make_loaders(n_train=1024, n_val=256, batch_size=32):
    torch.manual_seed(0)
    x_tr = torch.randn(n_train, 32)
    y_tr = (x_tr.sum(dim=1, keepdim=True) * 0.1 + torch.randn(n_train, 1) * 0.01)
    x_va = torch.randn(n_val, 32)
    y_va = (x_va.sum(dim=1, keepdim=True) * 0.1 + torch.randn(n_val, 1) * 0.01)
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Sanity check del setup")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Lightning: {pl.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    train_loader, val_loader = make_loaders()
    model = DummyMLP()

    logger = False
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            logger = WandbLogger(
                project="prostate-dose-prediction",
                name="hello_world_sanity_check",
                save_dir="lightning_logs",
            )
            print("✓ W&B logger activo")
        except Exception as e:
            print(f"✗ W&B no disponible ({e}), seguir sin logger")
            logger = False

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=logger,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    print("\n✓ Sanity check completado")


if __name__ == "__main__":
    main()
