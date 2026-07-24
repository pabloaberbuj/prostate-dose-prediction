"""
LightningDataModule para predicción de dosis 2D.

Estrategia de batching: SAMPLING POR PACIENTE.
- Cada item del dataset es un volumen completo de paciente.
- DataLoader entrega batches de N volúmenes.
- El LightningModule itera internamente sobre los cortes axiales y agrega.
- Esto permite calcular DVH/MomentLoss por paciente en cada paso.

Para entrenamiento, los volúmenes se paddean a una cantidad fija de cortes
(la mediana del dataset + margen) para poder hacer batches > 1.
Los cortes de padding se ignoran vía la máscara BODY (que está toda en 0 allí).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DosePatientDataset(Dataset):
    """
    Cada __getitem__ devuelve un volumen completo de un paciente como dict de tensores.
    Con cache_in_ram=True carga todos los NPZ al inicio → elimina el I/O durante training.
    """

    def __init__(self, npz_paths: list, augment: bool = False,
                 flip_lr_prob: float = 0.5,
                 rotation_degrees: float = 10.0,
                 intensity_jitter: float = 0.05,
                 fixed_n_slices: Optional[int] = None,
                 cache_in_ram: bool = True):
        self.npz_paths = [Path(p) for p in npz_paths]
        self.augment = augment
        self.flip_lr_prob = flip_lr_prob
        self.rotation_degrees = rotation_degrees
        self.intensity_jitter = intensity_jitter
        self.fixed_n_slices = fixed_n_slices
        self.cache_in_ram = cache_in_ram
        self._cache = {}

        if cache_in_ram:
            print(f"  Cargando {len(self.npz_paths)} pacientes en RAM...")
            for i, path in enumerate(self.npz_paths):
                sample = self._load_npz(path)
                # Aplicar padding en cache (operación fija, no depende de augmentation)
                if self.fixed_n_slices is not None:
                    sample = self._pad_or_crop_z(sample, self.fixed_n_slices)
                self._cache[i] = sample
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(self.npz_paths)}")
            print(f"  Cache completo.")

    def _load_npz(self, path: Path) -> dict:
        try:
            data = np.load(str(path), allow_pickle=True)
        except Exception:
            # Fallback con memory mapping si falla la carga directa
            data = np.load(str(path), allow_pickle=True, mmap_mode='r')
        meta = json.loads(str(data['meta'][0]))
        # CT, dose, PSDMs en float16 (~50% vs float32).
        # Masks en uint8 (originalmente ya son uint8 en disco — no hay pérdida de información).
        # Todo se convierte a float32 en __getitem__ antes de pasarlo al modelo.
        result = {
            'ct':           torch.from_numpy(np.array(data['ct'],           dtype=np.float32)).half(),
            'dose':         torch.from_numpy(np.array(data['dose'],         dtype=np.float32)).half(),
            'ptv_mask':     torch.from_numpy(np.array(data['ptv_mask'],     dtype=np.uint8)),
            'body_mask':    torch.from_numpy(np.array(data['body_mask'],    dtype=np.uint8)),
            'rectum_mask':  torch.from_numpy(np.array(data['rectum_mask'],  dtype=np.uint8)),
            'bladder_mask': torch.from_numpy(np.array(data['bladder_mask'], dtype=np.uint8)),
            'psdm_ptv':     torch.from_numpy(np.array(data['psdm_ptv'],     dtype=np.float32)).half(),
            'psdm_rectum':  torch.from_numpy(np.array(data['psdm_rectum'],  dtype=np.float32)).half(),
            'psdm_bladder': torch.from_numpy(np.array(data['psdm_bladder'], dtype=np.float32)).half(),
            'anonid':       meta.get('anonid', path.stem),
            'factor_norm':  float(meta.get('factor_norm', 1.0)),
        }
        data.close()
        return result

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict:
        if self.cache_in_ram:
            # Convertir a float32: cache usa float16 (CT/dose/PSDMs) y uint8 (masks)
            sample = {k: (v.float() if torch.is_tensor(v) else v)
                      for k, v in self._cache[idx].items()}
        else:
            # Cargar desde disco y convertir a float32 directamente
            sample = self._load_npz(self.npz_paths[idx])
            sample = {k: (v.float() if torch.is_tensor(v) else v)
                      for k, v in sample.items()}

        if self.augment:
            sample = self._apply_augmentation(sample)

        # Padding solo si no se aplicó en cache
        if self.fixed_n_slices is not None and not self.cache_in_ram:
            sample = self._pad_or_crop_z(sample, self.fixed_n_slices)

        return sample

    def _apply_augmentation(self, sample: dict) -> dict:
        # Flip LR
        if torch.rand(1).item() < self.flip_lr_prob:
            for key in ['ct', 'dose', 'ptv_mask', 'body_mask', 'rectum_mask',
                        'bladder_mask', 'psdm_ptv', 'psdm_rectum', 'psdm_bladder']:
                sample[key] = torch.flip(sample[key], dims=[-1])

        # Rotación pequeña (en el plano axial)
        if self.rotation_degrees > 0:
            angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
            sample = self._rotate_volume(sample, angle)

        # Intensity jitter en CT (rango [-1,1] → ±intensity_jitter)
        if self.intensity_jitter > 0:
            jitter = (torch.rand(1).item() * 2 - 1) * self.intensity_jitter
            sample['ct'] = torch.clamp(sample['ct'] + jitter, -1.0, 1.0)

        return sample

    def _rotate_volume(self, sample: dict, angle_deg: float) -> dict:
        """Rota cada corte axial por el mismo ángulo. Volumen: (Z, H, W)."""
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        # Matriz afín 2D: [Z, 2, 3]
        Z = sample['ct'].shape[0]
        theta = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a,  cos_a, 0]], dtype=torch.float32)
        theta = theta.unsqueeze(0).expand(Z, -1, -1)

        for key, mode in [
            ('ct', 'bilinear'), ('dose', 'bilinear'),
            ('psdm_ptv', 'bilinear'), ('psdm_rectum', 'bilinear'), ('psdm_bladder', 'bilinear'),
            ('ptv_mask', 'nearest'), ('body_mask', 'nearest'),
            ('rectum_mask', 'nearest'), ('bladder_mask', 'nearest'),
        ]:
            vol = sample[key].unsqueeze(1)  # (Z, 1, H, W)
            grid = F.affine_grid(theta, vol.shape, align_corners=False)
            rotated = F.grid_sample(vol, grid, mode=mode, align_corners=False,
                                    padding_mode='zeros')
            sample[key] = rotated.squeeze(1)
        return sample

    def _pad_or_crop_z(self, sample: dict, n_target: int) -> dict:
        z_actual = sample['ct'].shape[0]
        if z_actual == n_target:
            return sample
        if z_actual > n_target:
            # Crop centrado en Z
            start = (z_actual - n_target) // 2
            end   = start + n_target
            for key in ['ct', 'dose', 'ptv_mask', 'body_mask', 'rectum_mask',
                        'bladder_mask', 'psdm_ptv', 'psdm_rectum', 'psdm_bladder']:
                sample[key] = sample[key][start:end]
        else:
            # Pad simétrico en Z (con ceros)
            pad_total = n_target - z_actual
            pad_before = pad_total // 2
            pad_after  = pad_total - pad_before
            for key in ['ct', 'dose', 'ptv_mask', 'body_mask', 'rectum_mask',
                        'bladder_mask', 'psdm_ptv', 'psdm_rectum', 'psdm_bladder']:
                # F.pad espera padding al revés: (left, right, top, bottom, front, back)
                # Para tensor (Z, H, W), padear en Z = primer eje
                pad_value = 0.0 if 'mask' not in key else 0
                # CT en padding = -1 (HU mínimo normalizado)
                if key == 'ct':
                    pad_value = -1.0
                # PSDM en padding = 1 (lejos de la estructura)
                if 'psdm' in key:
                    pad_value = 1.0
                sample[key] = F.pad(sample[key], (0,0, 0,0, pad_before, pad_after),
                                    value=pad_value)
        return sample


# ─── DataModule ──────────────────────────────────────────────────────────────

class DoseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.processed_dir = Path(cfg.data.processed_dir)
        self.splits_file   = Path(cfg.data.splits_file)
        self.batch_size    = cfg.data.batch_size
        self.num_workers   = cfg.data.num_workers
        self.pin_memory    = cfg.data.pin_memory
        self.cache_train   = getattr(cfg.data, 'cache_train', True)
        self.cache_val     = getattr(cfg.data, 'cache_val', False)

    def setup(self, stage: Optional[str] = None):
        with open(self.splits_file) as f:
            splits = json.load(f)

        def to_paths(anonid_list):
            return [self.processed_dir / f"{a}.npz" for a in anonid_list
                    if (self.processed_dir / f"{a}.npz").exists()]

        train_paths = to_paths(splits['train'])
        val_paths   = to_paths(splits['val'])
        test_paths  = to_paths(splits['test'])

        # Calcular n_slices fijo a partir del train set
        n_target = self._calcular_n_slices_target(train_paths)
        print(f"[DataModule] n_slices fijo para padding: {n_target}")

        if self.cache_train:
            n_float16 = 5  # ct, dose, psdm_ptv, psdm_rectum, psdm_bladder
            n_uint8   = 4  # ptv_mask, body_mask, rectum_mask, bladder_mask
            bytes_pp  = (n_float16 * 2 + n_uint8 * 1) * n_target * 256 * 256
            est_gb = len(train_paths) * bytes_pp / 1e9
            print(f"[DataModule] Cache train estimado: {est_gb:.1f} GB "
                  f"({len(train_paths)} pac × {n_target} slices, float16+uint8)")
        else:
            print(f"[DataModule] Sin cache — carga desde disco por batch (compute-bound, <3% overhead)")

        self.train_ds = DosePatientDataset(
            train_paths, augment=True,
            flip_lr_prob     = self.cfg.augmentation.flip_lr_prob,
            rotation_degrees = self.cfg.augmentation.rotation_degrees,
            intensity_jitter = self.cfg.augmentation.intensity_jitter_hu / 1000.0,
            fixed_n_slices   = n_target,
            cache_in_ram     = self.cache_train,
        )
        self.val_ds  = DosePatientDataset(val_paths,  augment=False,
                                          fixed_n_slices=n_target, cache_in_ram=self.cache_val)
        self.test_ds = DosePatientDataset(test_paths, augment=False,
                                          fixed_n_slices=n_target, cache_in_ram=False)

        print(f"[DataModule] Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}")

    def _calcular_n_slices_target(self, paths: list) -> int:
        """Devuelve max n_slices del train set (redondeado a múltiplo de 8)."""
        n_slices_list = []
        for p in paths:
            data = np.load(str(p), allow_pickle=True)
            n_slices_list.append(data['ct'].shape[0])
            data.close()
        n_max = max(n_slices_list)
        # Redondear al múltiplo de 8 más cercano por arriba
        n_target = ((n_max + 7) // 8) * 8
        return n_target

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=False,
        )