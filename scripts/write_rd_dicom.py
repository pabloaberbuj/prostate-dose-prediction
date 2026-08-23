"""
Piloto KBP — escribe un RD (RTDOSE) DICOM sintético a partir de dose_pred (exp002)
y lo compara contra el RD real del plan clínico, para intentar importarlo en Eclipse.

Geometría in-plane (X/Y) — CALIBRADA empíricamente, no asumida a partir de la CT:
preprocess.py NO recorta en X/Y (confirmado leyendo el código y las 3 versiones en
git — sólo hace downsample_inplane 512->256 con `scipy.ndimage.zoom` calculando fy,
fx por separado). Pero fy≠fx cada vez que la imagen de entrada no es cuadrada, y
hay evidencia de que la CT que preprocess.py vio originalmente (para este dataset,
generado hace meses) no era la misma serie que la CT exportada hoy desde Eclipse
para este piloto (meta['spacing_mm']=2.5mm no coincide con la CT real de hoy,
0.976563mm — ver hallazgo documentado en CLAUDE_CODE_CONTEXT). Sea cual sea la
causa exacta (serie distinta, matriz no cuadrada, u otra cosa ya perdida de esa
corrida original), el resultado medible es el mismo: el grid 256x256 de este NPZ
tiene spacing distinto en X e Y (sp_x/sp_y ~1.47 para PT_003fe2bb84986507, medido
abajo), y ni preprocess.py ni el NPZ guardan el tamaño nativo real para derivarlo
analíticamente.
En vez de reconstruir esa transformación perdida, se CALIBRA empíricamente contra
la única fuente de verdad física disponible siempre (para cualquier paciente con
RS real): el propio contorno del PTV.
  - spacing por eje: extensión real del PTV en mm (min/max de los puntos de
    contorno del RS, por eje) ÷ su extensión en píxeles en el array 256x256.
  - origen por eje: se resuelve para que el centro de la extensión del PTV en
    píxeles caiga exactamente sobre el centro de su extensión real en mm.
Validado empíricamente: con esta calibración el RD sintético importó en Eclipse
alineado correctamente con la anatomía (confirmado por Pablo, ver captura en
results/pilot_rd/). Al no depender de ningún supuesto sobre la CT original (ni su
tamaño, ni si hubo o no un crop), esta calibración generaliza a cualquier paciente
sin necesitar recuperar esa transformación perdida.
El recorte en Z sigue siendo por índice de corte nativo (meta['z_range']),
verificado por separado contra la CT/RD reales (alineación exacta, sin bug).

Uso:
    python scripts/write_rd_dicom.py \
        --patient-id PT_003fe2bb84986507 \
        --dicom-dir  ../dicom_pilot/PT_003fe2bb84986507 \
        --pred-npz   results/pilot_rd/pred_PT_003fe2bb84986507.npz \
        --output     results/pilot_rd/RD_UNet_PT_003fe2bb84986507.dcm
"""

import argparse
import copy
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bridge.unet_to_target import (
    calibrar_grid_256_inplane,
    leer_geometria_ct,
    puntos_ptv_desde_rs,
)


# ──────────────────────────────────────────────────────────────────────────────
# Lectura de geometría real (RD template) — leer_geometria_ct/puntos_ptv_desde_rs/
# calibrar_grid_256_inplane se promovieron a src/bridge/unet_to_target.py (las
# reusa también el bridge de mimicking), se importan arriba.
# ──────────────────────────────────────────────────────────────────────────────

def leer_rd_template(rd_path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(rd_path))


# ──────────────────────────────────────────────────────────────────────────────
# Puente: dose_pred (%, 256x256xN recortado) → dosis Gy en el grid del RD template
# ──────────────────────────────────────────────────────────────────────────────

def construir_dosis_en_grid_rd(dose_pred_pct: np.ndarray, meta: dict,
                               ct_geom: dict, rd_template: pydicom.Dataset,
                               rx_gy: float, ptv_mask_256: np.ndarray,
                               dicom_dir: Path) -> np.ndarray:
    z_min, z_max = meta["z_range"]
    n_npz = dose_pred_pct.shape[0]
    assert n_npz == (z_max - z_min), (
        f"n_slices del npz ({n_npz}) no coincide con z_range {meta['z_range']}"
    )

    dose_gy_256 = dose_pred_pct.astype(np.float64) * rx_gy / 100.0

    # Footprint físico real del grid 256x256, calibrado por eje (X e Y por
    # separado) vía la extensión real del PTV en el RS -- NO asumido a partir
    # de la CT exportada hoy, ver docstring del módulo.
    sp256_x, sp256_y, origin_x, origin_y = calibrar_grid_256_inplane(meta, ptv_mask_256, dicom_dir)

    rows_rd = int(rd_template.Rows)
    cols_rd = int(rd_template.Columns)
    n_frames = int(rd_template.NumberOfFrames)
    px_rd = [float(v) for v in rd_template.PixelSpacing]
    ipp_rd = [float(v) for v in rd_template.ImagePositionPatient]
    gfov = np.array([float(v) for v in rd_template.GridFrameOffsetVector])
    rd_z_positions = ipp_rd[2] + gfov

    # Cada frame RD -> índice de corte CT nativo más cercano (por posición física)
    ct_idx_for_frame = np.array([
        int(np.argmin(np.abs(ct_geom["ipp_z_per_slice"] - z))) for z in rd_z_positions
    ])
    dist = np.abs(ct_geom["ipp_z_per_slice"][ct_idx_for_frame] - rd_z_positions)
    if dist.max() > 1.0:
        raise ValueError(
            f"Grid Z del RD template no alinea con la CT (offset máx {dist.max():.2f}mm) "
            "— este paciente necesita interpolación en Z, no soportada todavía."
        )

    # Coordenadas destino (grid RD, físicas) -> coordenadas fuente (grid 256, fraccionarias)
    row_idx, col_idx = np.meshgrid(np.arange(rows_rd), np.arange(cols_rd), indexing="ij")
    x_phys = ipp_rd[0] + col_idx * px_rd[1]
    y_phys = ipp_rd[1] + row_idx * px_rd[0]
    col_256 = (x_phys - origin_x) / sp256_x
    row_256 = (y_phys - origin_y) / sp256_y

    out = np.zeros((n_frames, rows_rd, cols_rd), dtype=np.float64)
    n_fuera_rango = 0
    for f in range(n_frames):
        npz_idx = ct_idx_for_frame[f] - z_min
        if 0 <= npz_idx < n_npz:
            out[f] = map_coordinates(dose_gy_256[npz_idx], [row_256, col_256],
                                     order=1, mode="constant", cval=0.0)
        else:
            n_fuera_rango += 1
    print(f"Frames fuera del z_range recortado (dosis=0): {n_fuera_rango}/{n_frames}")
    return out.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Escritura del RD sintético (clona el template real)
# ──────────────────────────────────────────────────────────────────────────────

def escribir_rd_sintetico(rd_template: pydicom.Dataset, dose_gy: np.ndarray,
                          output_path: Path, referenced_plan_uid: str = None) -> pydicom.Dataset:
    ds = copy.deepcopy(rd_template)

    now = datetime.now()
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S.%f")

    nuevo_sop = generate_uid()
    ds.SOPInstanceUID = nuevo_sop
    ds.SeriesInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = nuevo_sop
    ds.SeriesDescription = "UNet Predicted Dose - PILOT"

    if referenced_plan_uid is not None:
        # El RD real referencia el plan tratado (con dosis ya asociada e intocable
        # en Eclipse). Para poder importar, apuntar a un plan vacío nuevo creado
        # para este piloto.
        ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = referenced_plan_uid

    scaling = float(ds.DoseGridScaling)
    # El modelo no tiene activación final -> ruido negativo cerca de 0 (ver dose_pred
    # min~-1% en predict_one.py). uint32 no tiene signo: castear un negativo sin clip
    # da wraparound a ~4.3e9 (visto empíricamente: dosis "sintética" de 337783 Gy).
    n_negativos = int((dose_gy < 0).sum())
    if n_negativos:
        print(f"Clipeando {n_negativos} voxels con dosis negativa (ruido del modelo) a 0.")
    dose_gy_clipped = np.clip(dose_gy, 0.0, None)
    pixel_raw = np.round(dose_gy_clipped / scaling)
    max_uint32 = np.iinfo(np.uint32).max
    if pixel_raw.max() > max_uint32:
        raise ValueError(
            f"Dosis predicha ({dose_gy_clipped.max():.1f} Gy) excede el rango representable "
            f"con el DoseGridScaling del template ({scaling}) — recalcular escala."
        )
    ds.PixelData = pixel_raw.astype(np.uint32).tobytes()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# Validación
# ──────────────────────────────────────────────────────────────────────────────

CAMPOS_DEBEN_COINCIDIR = [
    "PatientID", "PatientName", "StudyInstanceUID", "FrameOfReferenceUID",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "GridFrameOffsetVector", "Rows", "Columns", "NumberOfFrames",
    "DoseUnits", "DoseType", "DoseSummationType", "DoseGridScaling",
    "BitsAllocated", "PixelRepresentation",
]


def validar(ds_sintetico: pydicom.Dataset, rd_template: pydicom.Dataset):
    print("\n=== Validación tag-por-tag vs RD real ===")
    ok = True
    for campo in CAMPOS_DEBEN_COINCIDIR:
        v1, v2 = getattr(ds_sintetico, campo, None), getattr(rd_template, campo, None)
        same = str(v1) == str(v2)
        ok &= same
        if not same:
            print(f"  ✗ {campo}: sintético={v1!r}  real={v2!r}")
    print("  Todos los campos de geometría/identidad coinciden." if ok else "  Hay diferencias — revisar arriba.")

    reread = pydicom.dcmread(ds_sintetico.filename)
    dose_gy = reread.pixel_array.astype(np.float64) * float(reread.DoseGridScaling)
    dose_real_gy = rd_template.pixel_array.astype(np.float64) * float(rd_template.DoseGridScaling)
    print(f"\nDosis sintética: min={dose_gy.min():.2f} Gy  max={dose_gy.max():.2f} Gy")
    print(f"Dosis real:      min={dose_real_gy.min():.2f} Gy  max={dose_real_gy.max():.2f} Gy")
    return dose_gy, dose_real_gy


def figura_comparativa(dose_pred_gy: np.ndarray, dose_real_gy: np.ndarray, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz = dose_real_gy.shape[0]
    z_mid = nz // 2
    # corte con mayor dosis real cerca del medio, más informativo que el geométrico
    doses_por_corte = dose_real_gy.reshape(nz, -1).max(axis=1)
    z_mid = int(np.argmax(doses_por_corte[nz // 4: 3 * nz // 4]) + nz // 4)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = max(dose_real_gy.max(), dose_pred_gy.max())
    im0 = axes[0].imshow(dose_real_gy[z_mid], cmap="jet", vmin=0, vmax=vmax)
    axes[0].set_title(f"RD real (z={z_mid})")
    im1 = axes[1].imshow(dose_pred_gy[z_mid], cmap="jet", vmin=0, vmax=vmax)
    axes[1].set_title("RD sintético (U-Net)")
    diff = dose_pred_gy[z_mid] - dose_real_gy[z_mid]
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-10, vmax=10)
    axes[2].set_title("Diferencia (pred - real), Gy")
    for ax, im in zip(axes, [im0, im1, im2]):
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=110)
    plt.close(fig)
    print(f"\nFigura comparativa guardada en {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--dicom-dir", required=True, help="Carpeta con CT.*.dcm, RS*.dcm, RP*.dcm, RD*.dcm reales")
    parser.add_argument("--pred-npz", required=True, help="Salida de predict_one.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rx-gy", type=float, default=78.0)
    parser.add_argument("--referenced-plan-uid", default=None,
                        help="ReferencedSOPInstanceUID a usar en vez del RP real "
                             "(necesario si el RP real ya tiene dosis tratada asociada "
                             "y Eclipse no permite reimportar sobre él)")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    rd_files = glob.glob(str(dicom_dir / "RD.*.dcm"))
    if not rd_files:
        raise FileNotFoundError(f"No hay RD*.dcm real en {dicom_dir}")

    print("Leyendo geometría CT real...")
    ct_geom = leer_geometria_ct(dicom_dir)
    print(f"  CT: {ct_geom['n_slices']} cortes, {ct_geom['rows']}x{ct_geom['columns']}, "
          f"spacing={ct_geom['pixel_spacing']}, FrameOfReferenceUID={ct_geom['frame_of_reference_uid']}")

    print("Leyendo RD template real...")
    rd_template = leer_rd_template(Path(rd_files[0]))
    if str(rd_template.FrameOfReferenceUID) != ct_geom["frame_of_reference_uid"]:
        raise ValueError("FrameOfReferenceUID del RD no coincide con el de la CT — carpeta inconsistente")

    print("Cargando predicción...")
    data = np.load(args.pred_npz, allow_pickle=True)
    dose_pred_pct = data["dose_pred"]
    ptv_mask_256 = data["ptv_mask"]
    meta = json.loads(str(data["meta"][0]))

    print("Resampleando dosis predicha al grid del RD real...")
    dose_gy = construir_dosis_en_grid_rd(dose_pred_pct, meta, ct_geom, rd_template, args.rx_gy,
                                         ptv_mask_256, dicom_dir)

    print("Escribiendo RD sintético...")
    output_path = Path(args.output)
    ds_sintetico = escribir_rd_sintetico(rd_template, dose_gy, output_path,
                                         referenced_plan_uid=args.referenced_plan_uid)
    ds_sintetico.filename = str(output_path)
    print(f"Guardado: {output_path}")
    print(f"ReferencedRTPlanSequence -> {ds_sintetico.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID}")

    dose_pred_gy, dose_real_gy = validar(ds_sintetico, rd_template)
    figura_comparativa(dose_pred_gy, dose_real_gy, output_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
