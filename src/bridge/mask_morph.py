"""Morfologia 3D en distancia fisica real (mm), no en voxels -- para mascaras con
spacing anisotropico (tipico en CT: ~1mm in-plane, 2.5-3mm entre cortes). Usado por
`scripts/substruct_derisk_mask.py` (Bloque 0) y `scripts/split_oar_by_dose.py`
(Bloque 1, HANDOFF_substructuras_dosis.md).

Un structuring element definido en voxels (p.ej. `scipy.ndimage.binary_erosion` con
`iterations=N`) erosiona/dilata distinto en cada eje si el spacing no es isotropico.
Estas funciones usan `distance_transform_edt(..., sampling=spacing)` para que
`radius_mm` sea una distancia euclidea real, igual en las 3 direcciones.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


def erode_mm(mask: np.ndarray, spacing: tuple, radius_mm: float) -> np.ndarray:
    """Erosiona `mask` (3D bool) `radius_mm` en todas direcciones. `spacing` debe
    tener un valor por eje de `mask`, en el mismo orden que sus ejes."""
    if mask.sum() == 0:
        raise ValueError("Mascara de entrada vacia -- no hay nada para erosionar")
    dist = distance_transform_edt(mask, sampling=spacing)
    return dist > radius_mm


def dilate_mm(mask: np.ndarray, spacing: tuple, radius_mm: float) -> np.ndarray:
    """Dilata `mask` (3D bool) `radius_mm` en todas direcciones."""
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm


def close_mm(mask: np.ndarray, spacing: tuple, radius_mm: float) -> np.ndarray:
    """Closing morfologico (dilata y despues erosiona el mismo radio) -- rellena
    huecos/muescas mas chicas que `radius_mm` sin cambiar apreciablemente el
    contorno externo."""
    if radius_mm <= 0:
        return mask
    return erode_mm(dilate_mm(mask, spacing, radius_mm), spacing, radius_mm)
