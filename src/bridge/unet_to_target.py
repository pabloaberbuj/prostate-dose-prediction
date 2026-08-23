"""Puente: dosis U-Net (grid 256x256xN, % de prescripción) -> dosis en Gy en
la MISMA grilla física que usa PyDoseRT (`patient.density_image`/`patient.dose`
de `pydosert.data.loaders.load_dicom`), para usar como target del mimicking
(PIPELINE_KBP_PDRT.md, sección A2/B paso 2).

La calibración in-plane (spacing/origen del grid 256, por eje) se resuelve
contra el contorno real del PTV en el RS, sin depender de ningún supuesto
sobre la CT original (tamaño, si hubo crop, qué serie se usó) — validada
empíricamente en el piloto RD DICOM (`scripts/write_rd_dicom.py`, ver
CLAUDE_CODE_CONTEXT_070826.md sección "Piloto..."). Las funciones de esa
calibración se promovieron a este módulo tal como recomendaba ese hallazgo;
`write_rd_dicom.py` las importa de acá.

A diferencia del piloto (que resampleaba al grid nativo del RD real, para
poder importar el resultado en Eclipse), esta versión resamplea a la grilla
que arma `load_dicom` (crop+resample con SimpleITK). `load_dicom` no expone
el `ct_resampled` de referencia con el que resamplea la dosis AAA real
internamente — se reconstruye acá con las mismas funciones que usa por
dentro (`load_ct_series`/`resample_image_to_spacing`/`center_crop_axial`,
mismos parámetros) para garantizar exactamente la misma grilla de salida
(mismo shape/spacing que `patient.dose`).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

from pydosert.data.loaders import center_crop_axial, resample_image_to_spacing
from pydosert.data.utils.dicom_utils import load_ct_series


def leer_geometria_ct(dicom_dir: Path) -> dict:
    """Lee todos los CT.*.dcm de la carpeta, ordenados por posición Z física."""
    dicom_dir = Path(dicom_dir)
    ct_files = sorted(glob.glob(str(dicom_dir / "CT.*.dcm")))
    if not ct_files:
        raise FileNotFoundError(f"No hay archivos CT.*.dcm en {dicom_dir}")

    slices = [pydicom.dcmread(f, stop_before_pixels=True) for f in ct_files]
    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    orient = [round(float(v), 3) for v in slices[0].ImageOrientationPatient]
    if orient != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
        raise NotImplementedError(
            f"ImageOrientationPatient no estándar ({orient}) — este módulo asume "
            "axial puro, sin gantry tilt."
        )

    rows, cols = int(slices[0].Rows), int(slices[0].Columns)
    px = [float(v) for v in slices[0].PixelSpacing]
    ipp_xy = (float(slices[0].ImagePositionPatient[0]),
              float(slices[0].ImagePositionPatient[1]))
    ipp_z = np.array([float(s.ImagePositionPatient[2]) for s in slices])

    return {
        "rows": rows, "columns": cols, "pixel_spacing": px, "ipp_xy": ipp_xy,
        "ipp_z_per_slice": ipp_z, "n_slices": len(slices),
        "frame_of_reference_uid": str(slices[0].FrameOfReferenceUID),
        "patient_id": str(slices[0].PatientID),
    }


def puntos_ptv_desde_rs(dicom_dir: Path, nombre_ptv: str) -> np.ndarray:
    """Todos los puntos de contorno (x, y) en mm del PTV real, desde el RS
    (independiente de qué CT se haya usado en preprocess.py)."""
    dicom_dir = Path(dicom_dir)
    rs_files = glob.glob(str(dicom_dir / "RS.*.dcm"))
    if not rs_files:
        raise FileNotFoundError(f"No hay RS*.dcm en {dicom_dir}")
    ds = pydicom.dcmread(rs_files[0])

    numero = None
    for roi in ds.StructureSetROISequence:
        if roi.ROIName == nombre_ptv:
            numero = roi.ROINumber
            break
    if numero is None:
        disponibles = [roi.ROIName for roi in ds.StructureSetROISequence]
        raise ValueError(f"'{nombre_ptv}' no encontrado en RS. Disponibles: {disponibles}")

    puntos = []
    for roi_contour in ds.ROIContourSequence:
        if roi_contour.ReferencedROINumber != numero:
            continue
        for contorno in roi_contour.ContourSequence:
            pts = np.array(contorno.ContourData, dtype=np.float64).reshape(-1, 3)
            puntos.append(pts[:, :2])
    return np.concatenate(puntos, axis=0)


def calibrar_grid_256_inplane(meta: dict, ptv_mask_256: np.ndarray,
                              dicom_dir: Path) -> tuple:
    """Spacing y origen reales del grid 256x256, calibrados por eje por separado
    (NO isotrópico -- el downsample_inplane de preprocess.py calcula fy/fx por
    separado, así que si la CT nativa que vio en su momento no era cuadrada el
    spacing resultante en X y en Y difiere; visto empíricamente: sp_x/sp_y ~1.47
    para PT_003fe2bb84986507).

    Devuelve (sp_x, sp_y, origin_x, origin_y).
    """
    _, rr, cc = np.where(ptv_mask_256 > 0)
    row_min, row_max = int(rr.min()), int(rr.max())
    col_min, col_max = int(cc.min()), int(cc.max())

    puntos = puntos_ptv_desde_rs(dicom_dir, meta["nombre_ptv"])
    x_min, x_max = puntos[:, 0].min(), puntos[:, 0].max()
    y_min, y_max = puntos[:, 1].min(), puntos[:, 1].max()

    sp_x = float((x_max - x_min) / (col_max - col_min))
    sp_y = float((y_max - y_min) / (row_max - row_min))

    col_c, row_c = (col_min + col_max) / 2.0, (row_min + row_max) / 2.0
    x_c, y_c = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    origin_x = x_c - col_c * sp_x
    origin_y = y_c - row_c * sp_y

    print(f"[unet_to_target] Calibración in-plane 256-grid: sp_x={sp_x:.4f}mm sp_y={sp_y:.4f}mm "
          f"(ratio {sp_x/sp_y:.3f}) origin=({origin_x:.2f},{origin_y:.2f}) "
          f"[footprint {256*sp_x:.1f}x{256*sp_y:.1f}mm]")
    return sp_x, sp_y, origin_x, origin_y


def dose_pred_a_grid_nativo(dose_pred_pct: np.ndarray, meta: dict, ct_geom: dict,
                             ptv_mask_256: np.ndarray, dicom_dir: Path) -> np.ndarray:
    """Resamplea `dose_pred_pct` (grid 256x256xN, en % -- clip+renormalizado a
    D95(PTV)=100% ya aplicado por el caller) a la grilla NATIVA completa de la CT
    (n_slices_nativo, Rows, Columns), usando la misma calibracion in-plane empirica
    que `construir_dosis_en_grid_rd` (write_rd_dicom.py) -- por eje, contra el
    contorno real del PTV. Cortes nativos fuera de `meta['z_range']` (donde el
    modelo no predijo nada) quedan en 0.

    Usado por `scripts/split_oar_by_dose.py` (Bloque 1, HANDOFF_substructuras_dosis.md)
    para poder umbralar dose_pred con una mascara de OAR leida en resolucion nativa
    (via rt-utils), no la de 256x256.

    Devuelve un array (n_slices_nativo, Rows, Columns) -- convencion numpy estandar
    de `pixel_array` (igual a `ct_geom`: axis1=fila->Y usando row_spacing,
    axis2=columna->X usando col_spacing), NO la de rt-utils.

    ATENCION -- gotcha verificado empiricamente (PT_44a81316c45f64f5, comparando
    contra PTV_High: bounding box de los contornos del RS vs. bounding box de
    `rtstruct.get_roi_mask_by_name` convertido a mm): pese a que rt-utils llama a
    sus ejes "(Columns, Rows, n_slices)", el eje que rt-utils llama "Columns"
    (axis0 de su mascara) es en realidad el eje **Y** (usar row_spacing), y el que
    llama "Rows" (axis1) es el eje **X** (usar col_spacing) -- lo opuesto a lo que
    sugieren esos nombres. Confirmado tambien contra las Dose Statistics de Eclipse
    (PTV_High Mean/Min/Max coincidieron solo con esta convencion, no con la
    "naive"). Por eso, para combinar este array con una mascara de rt-utils hay
    que transponer con `.transpose(1, 2, 0)` (que lleva axis1(fila/Y)->axis0 y
    axis2(columna/X)->axis1) -- **NO** `.transpose(2, 1, 0)`, que fue el bug
    original de `split_oar_by_dose.py` (corregido).
    """
    z_min, z_max = meta["z_range"]
    n_npz = dose_pred_pct.shape[0]
    if n_npz != (z_max - z_min):
        raise ValueError(f"n_slices del npz ({n_npz}) no coincide con z_range {meta['z_range']}")

    sp_x, sp_y, origin_x, origin_y = calibrar_grid_256_inplane(meta, ptv_mask_256, dicom_dir)

    rows, cols = ct_geom["rows"], ct_geom["columns"]
    row_spacing, col_spacing = ct_geom["pixel_spacing"]  # DICOM: [row_spacing, column_spacing]
    ipp_x, ipp_y = ct_geom["ipp_xy"]

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    x_phys = ipp_x + col_idx * col_spacing
    y_phys = ipp_y + row_idx * row_spacing
    col_256 = (x_phys - origin_x) / sp_x
    row_256 = (y_phys - origin_y) / sp_y

    n_slices_nativo = ct_geom["n_slices"]
    out = np.zeros((n_slices_nativo, rows, cols), dtype=np.float64)
    n_fuera = 0
    for i in range(n_slices_nativo):
        npz_idx = i - z_min
        if 0 <= npz_idx < n_npz:
            out[i] = map_coordinates(dose_pred_pct[npz_idx], [row_256, col_256],
                                     order=1, mode="constant", cval=0.0)
        else:
            n_fuera += 1
    if n_fuera:
        print(f"[dose_pred_a_grid_nativo] {n_fuera}/{n_slices_nativo} cortes nativos "
              f"fuera de z_range (dosis=0).")
    return out


def unet_dose_to_pdrt_grid(dicom_dir, pred_npz_path, rx_gy_total: float, n_fracciones: int,
                            new_spacing: tuple = (2.0, 2.0, 2.0),
                            crop_volume: bool = True) -> tuple:
    """Resamplea la dosis U-Net (NPZ de `predict_one.py`, % de prescripción,
    grid 256x256xN) a la grilla física que usa PyDoseRT para este paciente.

    Args:
        dicom_dir: carpeta con CT.*.dcm y RS.*.dcm reales del paciente.
        pred_npz_path: salida de `scripts/predict_one.py` (claves `dose_pred`,
            `ptv_mask`, `meta`).
        rx_gy_total: dosis de prescripción TOTAL del curso (Gy) — la que usa
            el proyecto para normalizar el % (D95(PTV)=100% de esta dosis).
        n_fracciones: fracciones planeadas. `load_dicom` (pydosert) carga la
            dosis del RD (DoseSummationType=PLAN, dosis TOTAL del curso) y la
            divide por las fracciones para que `patient.dose` sea dosis POR
            FRACCIÓN (así calza con el `compute_dose` del motor, que usa las
            MU por fracción del RP). Acá se hace la misma división para que
            el target quede en la misma convención — si no, la loss de
            mimicking compara total contra por-fracción y queda ~N veces mal
            escalada sin que salte ningún error.
        new_spacing, crop_volume: DEBEN coincidir con los que se use al cargar
            el mismo paciente con `load_dicom(...)` en el resto del pipeline
            de mimicking, para que el resultado tenga el mismo shape que
            `patient.dose`/`patient.density_image`.

    Returns:
        tuple[np.ndarray, sitk.Image]: dosis U-Net en Gy POR FRACCIÓN, shape
            (z, y, x) igual al de `patient.dose`; y la imagen CT resampleada
            de referencia (misma grilla), por si hace falta para debug/plots.
    """
    dicom_dir = Path(dicom_dir)
    ct_geom = leer_geometria_ct(dicom_dir)

    data = np.load(pred_npz_path, allow_pickle=True)
    dose_pred_pct = data["dose_pred"]
    ptv_mask_256 = data["ptv_mask"]
    meta = json.loads(str(data["meta"][0]))

    z_min, z_max = meta["z_range"]
    n_npz = dose_pred_pct.shape[0]
    if n_npz != (z_max - z_min):
        raise ValueError(f"n_slices del npz ({n_npz}) no coincide con z_range {meta['z_range']}")

    sp_x, sp_y, origin_x, origin_y = calibrar_grid_256_inplane(meta, ptv_mask_256, dicom_dir)
    dose_gy_256 = (dose_pred_pct.astype(np.float64) * rx_gy_total / 100.0) / n_fracciones

    z_positions_native = ct_geom["ipp_z_per_slice"][z_min:z_max]
    z_steps = np.diff(z_positions_native)
    z_step = float(np.median(z_steps)) if z_steps.size else float(ct_geom["ipp_z_per_slice"][1] - ct_geom["ipp_z_per_slice"][0])
    if z_steps.size and np.max(np.abs(z_steps - z_step)) > 0.05:
        print(f"[unet_to_target] Aviso: paso en Z no perfectamente uniforme "
              f"(min={z_steps.min():.3f} max={z_steps.max():.3f}mm) — se usa la mediana ({z_step:.3f}mm).")

    # sitk.GetImageFromArray espera (z,y,x); SetSpacing/SetOrigin en orden (x,y,z).
    dose_img = sitk.GetImageFromArray(dose_gy_256.astype(np.float32))
    dose_img.SetSpacing((sp_x, sp_y, z_step))
    dose_img.SetOrigin((origin_x, origin_y, float(z_positions_native[0])))
    dose_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    ct_series, _ = load_ct_series(str(dicom_dir))
    new_spacing_sitk = (new_spacing[2], new_spacing[1], new_spacing[0])  # sitk usa (x,y,z)
    ct_resampled = resample_image_to_spacing(ct_series, new_spacing=new_spacing_sitk, interpolator=sitk.sitkLinear)
    if crop_volume:
        ct_resampled = center_crop_axial(ct_resampled, max_size_cm=40.0)

    dose_resampled = sitk.Resample(
        dose_img, ct_resampled, sitk.Transform(), sitk.sitkLinear, 0.0, dose_img.GetPixelID()
    )
    dose_gy_pdrt_grid = sitk.GetArrayFromImage(dose_resampled)  # (z, y, x), Gy
    return dose_gy_pdrt_grid, ct_resampled


def rd_real_a_grid_nativo(rd_path, ct_geom: dict) -> np.ndarray:
    """Resamplea un RD (RTDOSE) real de Eclipse a la grilla NATIVA completa de
    la CT (n_slices, Rows, Columns) -- misma convencion numpy estandar de
    `pixel_array` que `dose_pred_a_grid_nativo` (axis1=fila->Y, axis2=columna->X).
    No requiere calibracion in-plane empirica (el RD trae su propia geometria
    DICOM completa, a diferencia del grid 256x256 del modelo) -- solo hace
    falta reproyectar dosis-por-frame al plano XY de la CT y emparejar Z por
    posicion fisica mas cercana.

    Usado para comparar dosis REAL (RD) contra dose_pred, o para calcular
    DVH real de subestructuras cuando no hay npz de prediccion (ej. planes ya
    optimizados en Eclipse) -- ver conversacion del piloto de substructuras
    (HANDOFF_substructuras_dosis.md), validado contra las Dose Statistics de
    Eclipse para PTV_High (match dentro de 0.5pp en mean/min/max).
    """
    rd = pydicom.dcmread(str(rd_path))
    scaling = float(rd.DoseGridScaling)
    rd_pixels = rd.pixel_array.astype(np.float64) * scaling
    ipp_rd = [float(v) for v in rd.ImagePositionPatient]
    px_rd = [float(v) for v in rd.PixelSpacing]
    gfov = np.array([float(v) for v in rd.GridFrameOffsetVector])
    rd_z = ipp_rd[2] + gfov

    rows, cols = ct_geom["rows"], ct_geom["columns"]
    row_sp, col_sp = ct_geom["pixel_spacing"]
    ipp_x, ipp_y = ct_geom["ipp_xy"]
    ct_z = ct_geom["ipp_z_per_slice"]
    n_slices = ct_geom["n_slices"]

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    x_phys = ipp_x + col_idx * col_sp
    y_phys = ipp_y + row_idx * row_sp
    col_rd = (x_phys - ipp_rd[0]) / px_rd[1]
    row_rd = (y_phys - ipp_rd[1]) / px_rd[0]

    dose_gy_native = np.zeros((n_slices, rows, cols), dtype=np.float64)
    n_fuera = 0
    for i in range(n_slices):
        fi = int(np.argmin(np.abs(rd_z - ct_z[i])))
        if abs(rd_z[fi] - ct_z[i]) > 2.0:
            n_fuera += 1
            continue
        dose_gy_native[i] = map_coordinates(rd_pixels[fi], [row_rd, col_rd],
                                            order=1, mode="constant", cval=0.0)
    if n_fuera:
        print(f"[rd_real_a_grid_nativo] {n_fuera}/{n_slices} cortes nativos sin frame RD "
              f"cercano (dosis=0).")
    return dose_gy_native
