"""
Etapa 2 — Preprocesado DICOM → NPZ

Por cada paciente en el dataset:
  1. Lee CT, RS (estructuras) y RD (dosis) desde su carpeta DICOM.
  2. Remuestrea la dosis a la grilla de la CT.
  3. Extrae máscaras binarias (PTV, BODY, Bladder, Rectum) desde RS.
  4. Calcula PSDM (Physical Signed Distance Map) en cm, normalizado a [-1, 1].
  5. Normaliza la CT a [-1, 1].
  6. Normaliza la dosis a dosis relativa: D95(PTV) = 100%.
  7. Recorta axialmente: conserva cortes con presencia de PTV/Bladder/Rectum ± margen.
  8. Downsamplea in-plane a 256×256.
  9. Guarda como NPZ por paciente con todos los arrays y metadatos.

Uso:
    python data/preprocess.py \
        --dicom-root /path/to/dicoms \
        --output-dir data/processed \
        --splits    data/splits/splits_v1.json \
        --workers   4

    # Solo procesar una lista de AnonIDs (útil para re-procesar casos sueltos)
    python data/preprocess.py \
        --dicom-root /path/to/dicoms \
        --output-dir data/processed \
        --splits    data/splits/splits_v1.json \
        --only PT_7f36fc43bef06c52 PT_9b0b147e01c09f2d
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Constantes ───────────────────────────────────────────────────────────────
CT_HU_MIN      = -1000.0
CT_HU_MAX      =  1500.0
INPLANE_SIZE   = 256
Z_MARGIN       = 5        # cortes adicionales arriba/abajo del ROI
PSDM_NORM_CM   = 15.0     # normalización PSDM: divide por este valor → [-1, 1] aprox

OPCIONES_PTV   = ['PTV', 'PTV_High', 'PTV_High2', 'PTV_Abr22', 'PTVp']
OPCIONES_BODY  = ['BODY', 'body', 'Body', 'External', 'Body1', 'BODY1']
OAR_NAMES      = ['Rectum', 'Bladder']

# ─── Helpers DICOM ────────────────────────────────────────────────────────────

def cargar_ct(carpeta: Path) -> sitk.Image:
    """Lee la serie CT desde la carpeta y devuelve sitk.Image 3D (HU)."""
    reader = sitk.ImageSeriesReader()
    archivos = reader.GetGDCMSeriesFileNames(str(carpeta))
    if not archivos:
        raise FileNotFoundError(f"No se encontró serie CT en {carpeta}")
    reader.SetFileNames(archivos)
    reader.MetaDataDictionaryArrayUpdateOn()
    imagen = reader.Execute()
    return imagen


def cargar_rd(carpeta: Path) -> sitk.Image:
    """Lee el archivo RD (dosis DICOM) y devuelve sitk.Image 3D en cGy."""
    rds = list(carpeta.glob("RD*.dcm")) or list(carpeta.glob("rd*.dcm"))
    if not rds:
        # Buscar por SOPClassUID
        for f in carpeta.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if "1.2.840.10008.5.1.4.1.1.481.2" in str(getattr(ds, 'SOPClassUID', '')):
                    rds = [f]
                    break
            except Exception:
                continue
    if not rds:
        raise FileNotFoundError(f"No se encontró RD en {carpeta}")
    ds = pydicom.dcmread(str(rds[0]))
    escala = float(getattr(ds, 'DoseGridScaling', 1.0))
    array_dosis = ds.pixel_array.astype(np.float32) * escala  # cGy
    # Convertir a sitk.Image con geometría correcta
    imagen = sitk.GetImageFromArray(array_dosis)
    origen = list(map(float, ds.ImagePositionPatient))
    imagen.SetOrigin(origen)
    spacings = list(map(float, ds.PixelSpacing)) + [float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0]) if len(ds.GridFrameOffsetVector) > 1 else 3.0]
    imagen.SetSpacing([spacings[1], spacings[0], spacings[2]])
    # Dirección: asumir ejes estándar
    imagen.SetDirection([1,0,0, 0,1,0, 0,0,1])
    return imagen


def cargar_estructuras(carpeta: Path) -> dict:
    """
    Lee el archivo RS y devuelve dict {nombre_estructura: lista_de_contornos_por_slice}.
    Formato: {nombre: [(z_mm, array_Nx2_puntos_xy), ...]}
    """
    rss = list(carpeta.glob("RS*.dcm")) or list(carpeta.glob("rs*.dcm"))
    if not rss:
        for f in carpeta.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if "1.2.840.10008.5.1.4.1.1.481.3" in str(getattr(ds, 'SOPClassUID', '')):
                    rss = [f]
                    break
            except Exception:
                continue
    if not rss:
        raise FileNotFoundError(f"No se encontró RS en {carpeta}")

    ds = pydicom.dcmread(str(rss[0]))
    estructuras = {}

    # Mapa ROINumber → nombre
    nombre_por_numero = {
        roi.ROINumber: roi.ROIName
        for roi in ds.StructureSetROISequence
    }

    for roi_contour in ds.ROIContourSequence:
        numero = roi_contour.ReferencedROINumber
        nombre = nombre_por_numero.get(numero, f"ROI_{numero}")
        contornos = []
        if hasattr(roi_contour, 'ContourSequence'):
            for contorno in roi_contour.ContourSequence:
                puntos = np.array(contorno.ContourData, dtype=np.float64).reshape(-1, 3)
                z = puntos[0, 2]
                xy = puntos[:, :2]
                contornos.append((z, xy))
        estructuras[nombre] = contornos

    return estructuras


def contornos_a_mascara(contornos: list, imagen_ref: sitk.Image) -> np.ndarray:
    """
    Convierte lista de contornos [(z_mm, array_xy)] a máscara binaria 3D
    con la misma geometría que imagen_ref (ZYX order para numpy).
    """
    from skimage.draw import polygon

    size_z, size_y, size_x = (
        imagen_ref.GetSize()[2],
        imagen_ref.GetSize()[1],
        imagen_ref.GetSize()[0],
    )
    mascara = np.zeros((size_z, size_y, size_x), dtype=np.uint8)

    origen  = np.array(imagen_ref.GetOrigin())    # [x0, y0, z0]
    spacing = np.array(imagen_ref.GetSpacing())   # [sx, sy, sz]

    for z_mm, xy in contornos:
        # Índice Z en la imagen
        iz = int(round((z_mm - origen[2]) / spacing[2]))
        if iz < 0 or iz >= size_z:
            continue
        # Convertir XY físico → índices de píxel
        col = (xy[:, 0] - origen[0]) / spacing[0]  # x → col
        row = (xy[:, 1] - origen[1]) / spacing[1]  # y → row
        rr, cc = polygon(row, col, shape=(size_y, size_x))
        mascara[iz, rr, cc] = 1

    return mascara


# ─── Cálculo de PSDM ─────────────────────────────────────────────────────────

def calcular_psdm(mascara: np.ndarray, spacing_zyx: tuple) -> np.ndarray:
    """
    Physical Signed Distance Map en cm, normalizado por PSDM_NORM_CM.
    - Positivo fuera de la estructura (distancia al borde más cercano).
    - Negativo dentro de la estructura.
    - Cero en el borde.
    Devuelve array float32 con rango aproximado [-1, 1].
    """
    # Distancia fuera → positiva
    dist_fuera = distance_transform_edt(1 - mascara, sampling=spacing_zyx)
    # Distancia dentro → negativa
    dist_dentro = distance_transform_edt(mascara, sampling=spacing_zyx)
    # SDM en mm (spacing está en mm)
    sdm_mm = dist_fuera - dist_dentro
    # Convertir a cm y normalizar
    sdm_cm = sdm_mm / 10.0
    sdm_norm = (sdm_cm / PSDM_NORM_CM).astype(np.float32)
    return sdm_norm


# ─── Normalización de dosis ───────────────────────────────────────────────────

def normalizar_dosis(dosis_cgy: np.ndarray, mascara_ptv: np.ndarray,
                     prescripcion_cgy: float = 7800.0) -> tuple:
    """
    Normaliza la dosis para que D95(PTV) = 100%.
    Retorna (dosis_normalizada_pct, factor_escala).
    factor_escala: multiplicar dosis_normalizada por (prescripcion_cgy / 100) para volver a cGy.
    """
    dosis_ptv = dosis_cgy[mascara_ptv > 0]
    if len(dosis_ptv) == 0:
        raise ValueError("Máscara PTV vacía — no se puede calcular D95")
    d95_cgy = float(np.percentile(dosis_ptv, 5))  # D95 = percentil 5 del histograma acumulativo inverso
    if d95_cgy <= 0:
        raise ValueError(f"D95 = {d95_cgy:.2f} cGy — valor inválido")
    factor = prescripcion_cgy / d95_cgy
    dosis_norm_pct = (dosis_cgy * factor / prescripcion_cgy) * 100.0
    return dosis_norm_pct.astype(np.float32), float(factor)


# ─── Remuestreo ──────────────────────────────────────────────────────────────

def remuestrear_a_ct(imagen_fuente: sitk.Image, imagen_ref: sitk.Image,
                     interpolador=sitk.sitkLinear) -> np.ndarray:
    """Remuestrea imagen_fuente al espacio de imagen_ref. Devuelve array numpy ZYX."""
    resampleado = sitk.Resample(
        imagen_fuente,
        imagen_ref,
        sitk.Transform(),
        interpolador,
        0.0,
        imagen_fuente.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampleado)


def downsample_inplane(array_zyx: np.ndarray, target: int = INPLANE_SIZE) -> np.ndarray:
    """
    Downsamplea las dimensiones YX a target×target usando zoom de scipy.
    Mantiene la dimensión Z intacta.
    """
    from scipy.ndimage import zoom
    sz, sy, sx = array_zyx.shape
    if sy == target and sx == target:
        return array_zyx
    fy, fx = target / sy, target / sx
    return zoom(array_zyx, (1.0, fy, fx), order=1).astype(array_zyx.dtype)


# ─── Recorte axial ───────────────────────────────────────────────────────────

def recortar_axial(arrays: dict, margen: int = Z_MARGIN) -> dict:
    """
    Recorta en Z conservando solo los cortes donde hay presencia de
    PTV, Bladder o Rectum, más margen cortes arriba y abajo.
    Aplica el mismo recorte a todos los arrays del dict.
    """
    roi_union = np.zeros_like(arrays['ptv_mask'])
    for key in ['ptv_mask', 'bladder_mask', 'rectum_mask']:
        if key in arrays:
            roi_union = np.maximum(roi_union, arrays[key])

    slices_con_roi = np.where(roi_union.sum(axis=(1, 2)) > 0)[0]
    if len(slices_con_roi) == 0:
        raise ValueError("No se encontraron cortes con ROIs")

    z_min = max(0, int(slices_con_roi[0])  - margen)
    z_max = min(roi_union.shape[0], int(slices_con_roi[-1]) + margen + 1)

    recortados = {}
    for key, arr in arrays.items():
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            recortados[key] = arr[z_min:z_max]
        else:
            recortados[key] = arr
    recortados['z_range'] = (z_min, z_max)
    return recortados


# ─── Procesado de un paciente ─────────────────────────────────────────────────

def procesar_paciente(anonid: str, carpeta_dicom: Path, output_dir: Path) -> dict:
    """
    Pipeline completo para un paciente. Devuelve dict con métricas de QC.
    Lanza excepción si algo falla.
    """
    out_path = output_dir / f"{anonid}.npz"
    if out_path.exists():
        return {'anonid': anonid, 'status': 'skipped (ya existe)'}

    # 1. Cargar DICOM
    ct_sitk  = cargar_ct(carpeta_dicom)
    rd_sitk  = cargar_rd(carpeta_dicom)
    estructuras = cargar_estructuras(carpeta_dicom)

    # 2. Identificar estructuras por nombre
    def buscar(opciones):
        for op in opciones:
            if op in estructuras:
                return op
        # Búsqueda case-insensitive
        for nombre in estructuras:
            for op in opciones:
                if nombre.lower() == op.lower():
                    return nombre
        return None

    nombre_ptv    = buscar(OPCIONES_PTV)
    nombre_body   = buscar(OPCIONES_BODY)
    nombre_rectum = buscar(['Rectum'])
    nombre_bladder = buscar(['Bladder'])

    if nombre_ptv is None:
        raise ValueError(f"PTV no encontrado. Estructuras disponibles: {list(estructuras.keys())}")

    # 3. Construir máscaras en espacio CT
    def hacer_mascara(nombre):
        if nombre is None:
            sz = ct_sitk.GetSize()
            return np.zeros((sz[2], sz[1], sz[0]), dtype=np.uint8)
        return contornos_a_mascara(estructuras[nombre], ct_sitk)

    ptv_mask     = hacer_mascara(nombre_ptv)
    body_mask    = hacer_mascara(nombre_body)
    rectum_mask  = hacer_mascara(nombre_rectum)
    bladder_mask = hacer_mascara(nombre_bladder)

    # 4. Remuestrear dosis al espacio CT
    dosis_cgy = remuestrear_a_ct(rd_sitk, ct_sitk, sitk.sitkLinear)

    # 5. CT array y normalización HU
    ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    ct_norm  = np.clip(ct_array, CT_HU_MIN, CT_HU_MAX)
    ct_norm  = (ct_norm - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)  # [0, 1]
    ct_norm  = ct_norm * 2.0 - 1.0                               # [-1, 1]

    # 6. Normalizar dosis a D95(PTV) = 100%
    dosis_norm_pct, factor_norm = normalizar_dosis(dosis_cgy, ptv_mask)

    # 7. Calcular PSDM (en espacio CT, antes de downsample para máxima precisión)
    spacing_mm = ct_sitk.GetSpacing()  # (sx, sy, sz) en mm
    spacing_zyx_mm = (spacing_mm[2], spacing_mm[1], spacing_mm[0])

    psdm_ptv     = calcular_psdm(ptv_mask,     spacing_zyx_mm)
    psdm_rectum  = calcular_psdm(rectum_mask,  spacing_zyx_mm)
    psdm_bladder = calcular_psdm(bladder_mask, spacing_zyx_mm)

    # 8. Agrupar arrays y recortar axialmente
    arrays = {
        'ct':           ct_norm.astype(np.float32),
        'dose':         dosis_norm_pct,
        'ptv_mask':     ptv_mask,
        'body_mask':    body_mask,
        'rectum_mask':  rectum_mask,
        'bladder_mask': bladder_mask,
        'psdm_ptv':     psdm_ptv,
        'psdm_rectum':  psdm_rectum,
        'psdm_bladder': psdm_bladder,
    }
    arrays = recortar_axial(arrays, margen=Z_MARGIN)
    z_range = arrays.pop('z_range')

    # 9. Downsample in-plane a 256×256
    for key in arrays:
        if isinstance(arrays[key], np.ndarray) and arrays[key].ndim == 3:
            arrays[key] = downsample_inplane(arrays[key], INPLANE_SIZE)

    # 10. QC mínimo
    n_slices  = arrays['ct'].shape[0]
    vol_ptv   = float(ptv_mask.sum()) * np.prod(spacing_mm) / 1000.0  # cc
    dose_max  = float(arrays['dose'].max())
    dose_ptv_d95 = float(np.percentile(arrays['dose'][arrays['ptv_mask'] > 0], 5))

    # 11. Guardar NPZ
    meta = {
        'anonid':         anonid,
        'spacing_mm':     list(spacing_mm),          # (sx, sy, sz)
        'z_range':        list(z_range),
        'factor_norm':    factor_norm,
        'nombre_ptv':     nombre_ptv or '',
        'nombre_rectum':  nombre_rectum or '',
        'nombre_bladder': nombre_bladder or '',
        'vol_ptv_cc':     round(vol_ptv, 2),
        'n_slices':       n_slices,
    }

    np.savez_compressed(
        str(out_path),
        ct          = arrays['ct'],
        dose        = arrays['dose'],
        ptv_mask    = arrays['ptv_mask'],
        body_mask   = arrays['body_mask'],
        rectum_mask = arrays['rectum_mask'],
        bladder_mask = arrays['bladder_mask'],
        psdm_ptv    = arrays['psdm_ptv'],
        psdm_rectum = arrays['psdm_rectum'],
        psdm_bladder = arrays['psdm_bladder'],
        meta        = np.array([json.dumps(meta)]),
    )

    return {
        'anonid':      anonid,
        'status':      'ok',
        'n_slices':    n_slices,
        'vol_ptv_cc':  round(vol_ptv, 2),
        'dose_max':    round(dose_max, 2),
        'dose_d95_ptv': round(dose_ptv_d95, 2),
        'factor_norm': round(factor_norm, 4),
    }


# ─── Worker para multiprocessing ─────────────────────────────────────────────

def _worker(args):
    anonid, dicom_root, output_dir = args
    carpeta = Path(dicom_root) / anonid
    try:
        return procesar_paciente(anonid, carpeta, Path(output_dir))
    except Exception as e:
        return {'anonid': anonid, 'status': f'ERROR: {e}'}


# ─── CLI principal ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocesado DICOM → NPZ")
    parser.add_argument("--dicom-root",  required=True, help="Carpeta raíz con subcarpetas por AnonID")
    parser.add_argument("--output-dir",  required=True, help="Carpeta de salida para NPZs")
    parser.add_argument("--splits",      required=True, help="JSON con splits train/val/test")
    parser.add_argument("--workers",     type=int, default=1, help="Número de procesos paralelos")
    parser.add_argument("--only",        nargs="+",     help="Procesar solo estos AnonIDs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar lista de pacientes desde splits (excluye los descartados)
    with open(args.splits) as f:
        splits = json.load(f)

    excluidos = set(splits.get('excluidos', []))
    todos = splits['train'] + splits['val'] + splits['test']
    anonids = [a for a in todos if a not in excluidos]

    if args.only:
        anonids = [a for a in anonids if a in set(args.only)]
        log.info(f"Modo --only: procesando {len(anonids)} pacientes")

    log.info(f"Total a procesar: {len(anonids)} pacientes")
    log.info(f"Output: {output_dir}")
    log.info(f"Workers: {args.workers}")

    worker_args = [(a, args.dicom_root, args.output_dir) for a in anonids]

    resultados = []
    if args.workers == 1:
        for wa in tqdm(worker_args, desc="Preprocesando"):
            resultados.append(_worker(wa))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, wa): wa[0] for wa in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocesando"):
                resultados.append(future.result())

    # Resumen
    ok  = [r for r in resultados if r['status'] == 'ok']
    skip = [r for r in resultados if 'skipped' in r['status']]
    err = [r for r in resultados if 'ERROR' in r['status']]

    log.info(f"\n=== Resumen ===")
    log.info(f"OK:      {len(ok)}")
    log.info(f"Skipped: {len(skip)}")
    log.info(f"Errores: {len(err)}")

    if err:
        log.warning("Errores:")
        for r in err:
            log.warning(f"  {r['anonid']}: {r['status']}")

    # Guardar log de resultados
    log_path = output_dir / "preprocess_log.json"
    with open(log_path, 'w') as f:
        json.dump(resultados, f, indent=2)
    log.info(f"Log guardado en {log_path}")

    # QC rápido sobre los procesados
    if ok:
        n_slices = [r['n_slices'] for r in ok]
        vol_ptv  = [r['vol_ptv_cc'] for r in ok]
        d95      = [r['dose_d95_ptv'] for r in ok]
        log.info(f"\n=== QC ===")
        log.info(f"N cortes/paciente: {np.mean(n_slices):.1f} ± {np.std(n_slices):.1f} [{min(n_slices)}-{max(n_slices)}]")
        log.info(f"Vol PTV (cc):      {np.mean(vol_ptv):.1f} ± {np.std(vol_ptv):.1f}")
        log.info(f"D95 PTV (% pres):  {np.mean(d95):.2f} ± {np.std(d95):.2f} (debe ser ~100)")


if __name__ == "__main__":
    main()
