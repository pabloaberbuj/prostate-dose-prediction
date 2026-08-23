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
INPLANE_CROP_MM = 500.0   # lado de la caja de recorte en el plano, centrada en centroide PTV
                          # (medido sobre 44 pacientes 2026-08-23: garantiza PTV/Rectum/Bladder
                          # completos siempre; BODY completo en 97.7%, resto clipea lateral)

OPCIONES_PTV   = ['PTV', 'PTV_High', 'PTV_High2', 'PTV_Abr22', 'PTVp']
OPCIONES_BODY  = ['BODY', 'body', 'Body', 'External', 'Body1', 'BODY1']
OAR_NAMES      = ['Rectum', 'Bladder']

# ─── Helpers DICOM ────────────────────────────────────────────────────────────

def cargar_ct(carpeta: Path) -> sitk.Image:
    """Lee la serie CT desde la carpeta y devuelve sitk.Image 3D (HU).

    La carpeta mezcla CT + RD + RS + RP, así que hay que elegir explícitamente
    la serie con Modality == 'CT': GetGDCMSeriesFileNames() sin seriesID devuelve
    la primera serie que encuentra por orden de SeriesInstanceUID, que en este
    dataset resulta ser la serie RD de 1 archivo (dosis), no la CT real. Esto es
    lo que antes causaba el falso bug de "CT leída en 4D" — era el RD mal leído,
    no un problema real de la CT (confirmado por auditoría, 2026-08-23).
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(carpeta))
    if not series_ids:
        raise FileNotFoundError(f"No se encontraron series DICOM en {carpeta}")

    candidatos_ct = []
    for sid in series_ids:
        archivos_sid = reader.GetGDCMSeriesFileNames(str(carpeta), sid)
        if not archivos_sid:
            continue
        try:
            modality = str(pydicom.dcmread(archivos_sid[0], stop_before_pixels=True).Modality)
        except Exception:
            continue
        if modality == 'CT':
            candidatos_ct.append(archivos_sid)

    if not candidatos_ct:
        raise FileNotFoundError(f"No se encontró ninguna serie con Modality=='CT' en {carpeta}")

    # Si hay más de una serie CT, quedarse con la de más archivos (la CT de planificación completa)
    archivos = max(candidatos_ct, key=len)

    reader.SetFileNames(archivos)
    reader.MetaDataDictionaryArrayUpdateOn()
    imagen = reader.Execute()

    if imagen.GetDimension() != 3 or len(imagen.GetDirection()) != 9:
        raise ValueError(
            f"CT de {carpeta} no es 3D tras cargar la serie CT "
            f"(dirección de longitud {len(imagen.GetDirection())})"
        )

    # Rango HU plausible: tiene que haber aire (<-500) y hueso/tejido denso (>200).
    # Si no, casi seguro se coló la serie equivocada otra vez.
    array_hu = sitk.GetArrayFromImage(imagen)
    hu_min, hu_max = float(array_hu.min()), float(array_hu.max())
    if hu_min > -500.0 or hu_max < 200.0:
        raise ValueError(
            f"CT de {carpeta} con rango HU no plausible (min={hu_min:.1f}, max={hu_max:.1f}) "
            f"— posible serie equivocada"
        )

    return imagen


def cargar_rd(carpeta: Path) -> sitk.Image:
    """
    Lee el archivo RD (dosis DICOM) y devuelve sitk.Image 3D en cGy.

    DICOM RT Dose almacena los pixels como enteros sin signo.
    La dosis real = pixel_array × DoseGridScaling, donde DoseGridScaling
    está en Gy por unidad de pixel. El resultado está en Gy; multiplicamos
    por 100 para obtener cGy.
    sitk.ReadImage NO aplica DoseGridScaling automáticamente.
    """
    rd_path = None
    for f in carpeta.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            sop = str(getattr(ds, 'SOPClassUID', ''))
            modality = str(getattr(ds, 'Modality', ''))
            if '1.2.840.10008.5.1.4.1.1.481.2' in sop or modality == 'RTDOSE':
                rd_path = f
                break
        except Exception:
            continue

    if rd_path is None:
        raise FileNotFoundError(f"No se encontró RD en {carpeta}")

    # Leer con pydicom para obtener pixels crudos y el factor de escala
    ds = pydicom.dcmread(str(rd_path))
    escala_gy = float(getattr(ds, 'DoseGridScaling', 1.0))  # Gy / pixel_value
    array_cgy = ds.pixel_array.astype(np.float32) * escala_gy * 100.0  # → cGy

    # Verificación: dosis máxima debe ser razonable (entre 50 y 200 Gy → 5000-20000 cGy)
    dmax = array_cgy.max()
    if dmax < 100 or dmax > 20000:
        raise ValueError(
            f"Dosis máxima fuera de rango esperado: {dmax:.1f} cGy "
            f"(escala={escala_gy}, pixel_max={ds.pixel_array.max()})"
        )

    # Leer geometría con SimpleITK (origen, spacing, dirección)
    imagen_sitk = sitk.ReadImage(str(rd_path))

    # Construir imagen final con array correcto y geometría de SimpleITK
    imagen_cgy = sitk.GetImageFromArray(array_cgy)
    imagen_cgy.SetOrigin(imagen_sitk.GetOrigin())
    imagen_cgy.SetSpacing(imagen_sitk.GetSpacing())
    imagen_cgy.SetDirection(imagen_sitk.GetDirection())

    return imagen_cgy


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


# ─── Recorte en el plano ──────────────────────────────────────────────────────
# Asumen imagen_ref con dirección axis-aligned (identidad) — verificado empíricamente
# en este dataset (2026-08-23): sin gantry tilt, mapeo índice→físico simple es correcto.

def centroide_fisico(mascara: np.ndarray, imagen_ref: sitk.Image) -> tuple:
    """Centroide de masa (x,y) en mm físicos de una máscara binaria ZYX."""
    idx = np.argwhere(mascara > 0)
    if idx.size == 0:
        raise ValueError("Máscara vacía — no se puede calcular centroide")
    origen, spacing = imagen_ref.GetOrigin(), imagen_ref.GetSpacing()
    cx = origen[0] + idx[:, 2].mean() * spacing[0]
    cy = origen[1] + idx[:, 1].mean() * spacing[1]
    return cx, cy


def extensiones_mascara(mascara: np.ndarray, imagen_ref: sitk.Image, cx: float, cy: float) -> dict:
    """+X,-X,+Y,-Y en mm desde (cx,cy) hasta el vóxel más lejano de la máscara en cada dirección."""
    idx = np.argwhere(mascara > 0)
    if idx.size == 0:
        return {"+X": 0.0, "-X": 0.0, "+Y": 0.0, "-Y": 0.0}
    origen, spacing = imagen_ref.GetOrigin(), imagen_ref.GetSpacing()
    dx = origen[0] + idx[:, 2] * spacing[0] - cx
    dy = origen[1] + idx[:, 1] * spacing[1] - cy
    return {
        "+X": float(max(dx.max(), 0.0)), "-X": float(max(-dx.min(), 0.0)),
        "+Y": float(max(dy.max(), 0.0)), "-Y": float(max(-dy.min(), 0.0)),
    }


def calcular_recorte_plano(imagen_ref: sitk.Image, cx: float, cy: float, lado_mm: float) -> dict:
    """Índices [row_min:row_max, col_min:col_max] sobre la grilla nativa de imagen_ref para
    una caja cuadrada de lado_mm centrada en (cx,cy), más el padding necesario por lado si la
    caja pedida cae fuera del FOV real de la CT (paciente ancho)."""
    origen, spacing, size = imagen_ref.GetOrigin(), imagen_ref.GetSpacing(), imagen_ref.GetSize()
    half = lado_mm / 2.0

    col_min = int(round((cx - half - origen[0]) / spacing[0]))
    col_max = int(round((cx + half - origen[0]) / spacing[0]))
    row_min = int(round((cy - half - origen[1]) / spacing[1]))
    row_max = int(round((cy + half - origen[1]) / spacing[1]))

    return {
        "row_min": max(0, row_min), "row_max": min(size[1], row_max),
        "col_min": max(0, col_min), "col_max": min(size[0], col_max),
        "pad_top":    max(0, -row_min), "pad_bottom": max(0, row_max - size[1]),
        "pad_left":   max(0, -col_min), "pad_right":  max(0, col_max - size[0]),
    }


def aplicar_recorte_plano(array_zyx: np.ndarray, rec: dict, fill_value: float) -> np.ndarray:
    """Recorta en Y,X según `rec` y rellena con fill_value donde la caja pedida se sale del FOV."""
    recortado = array_zyx[:, rec["row_min"]:rec["row_max"], rec["col_min"]:rec["col_max"]]
    if any(rec[k] > 0 for k in ("pad_top", "pad_bottom", "pad_left", "pad_right")):
        recortado = np.pad(
            recortado,
            ((0, 0), (rec["pad_top"], rec["pad_bottom"]), (rec["pad_left"], rec["pad_right"])),
            mode="constant", constant_values=fill_value,
        )
    return recortado.astype(array_zyx.dtype)


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
    Normaliza la dosis para que D95(PTV) = 100% de la prescripción.

    La dosis de salida está en % de prescripción:
        100% = prescripcion_cgy = 78 Gy

    factor_norm: factor multiplicativo aplicado a la dosis original.
        dosis_norm_pct = dosis_cgy * factor_norm / prescripcion_cgy * 100
        factor_norm ≈ 1.0 cuando el plan ya alcanza D95 ≈ prescripción.
        factor_norm > 1.0 cuando D95 < prescripción (plan subóptimo).

    Retorna (dosis_normalizada_pct, factor_norm).
    """
    dosis_ptv = dosis_cgy[mascara_ptv > 0]
    if len(dosis_ptv) == 0:
        raise ValueError("Máscara PTV vacía — no se puede calcular D95")
    # D95 = percentil 5 (5% del volumen recibe MENOS que este valor)
    d95_cgy = float(np.percentile(dosis_ptv, 5))
    if d95_cgy <= 0:
        raise ValueError(f"D95 = {d95_cgy:.2f} cGy — valor inválido")

    # factor_norm ≈ 1.0: escala la dosis para que D95(PTV) = prescripción
    factor_norm = prescripcion_cgy / d95_cgy

    # Dosis normalizada en % de prescripción
    # Con factor_norm aplicado, D95(PTV) queda exactamente en 100%
    dosis_norm_pct = (dosis_cgy * factor_norm / prescripcion_cgy * 100.0)

    return dosis_norm_pct.astype(np.float32), float(factor_norm)


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

    # 5. CT array y normalización HU (sobre grilla nativa completa, antes de recortar)
    ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    ct_norm  = np.clip(ct_array, CT_HU_MIN, CT_HU_MAX)
    ct_norm  = (ct_norm - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)  # [0, 1]
    ct_norm  = ct_norm * 2.0 - 1.0                               # [-1, 1]

    # 6. Normalizar dosis a D95(PTV) = 100% (sobre máscara PTV nativa completa, sin recortar:
    #    el PTV siempre entra en la caja de recorte, así que el D95 no cambia por recortar)
    dosis_norm_pct, factor_norm = normalizar_dosis(dosis_cgy, ptv_mask)

    spacing_mm = ct_sitk.GetSpacing()  # (sx, sy, sz) en mm

    # 7. Recorte en el plano: caja cuadrada de INPLANE_CROP_MM centrada en el centroide del
    #    PTV. PTV/Rectum/Bladder deben entrar siempre completos (medido sobre 44 pacientes,
    #    2026-08-23) — si no, es un caso anómalo y se aborta el paciente para revisión manual
    #    en vez de guardar una máscara de OAR/PTV truncada en silencio. BODY sí puede clipear
    #    en lateral (esperado en pacientes anchos) — eso queda como warning, no error.
    cx, cy = centroide_fisico(ptv_mask, ct_sitk)
    half_mm = INPLANE_CROP_MM / 2.0

    clip_oar = {}
    for nombre, mascara in [('PTV', ptv_mask), ('Rectum', rectum_mask), ('Bladder', bladder_mask)]:
        ext = extensiones_mascara(mascara, ct_sitk, cx, cy)
        excedente = {d: round(v - half_mm, 1) for d, v in ext.items() if v > half_mm}
        if excedente:
            clip_oar[nombre] = excedente
    if clip_oar:
        raise ValueError(
            f"{anonid}: recorte de {INPLANE_CROP_MM/10:.0f}cm corta OAR/PTV — revisar caso "
            f"a mano (excedentes mm: {clip_oar})"
        )

    ext_body = extensiones_mascara(body_mask, ct_sitk, cx, cy)
    clip_body = {d: round(v - half_mm, 1) for d, v in ext_body.items() if v > half_mm}
    if clip_body:
        log.warning(
            f"{anonid}: BODY lateral clipeado {clip_body} mm por el recorte de "
            f"{INPLANE_CROP_MM/10:.0f}cm, sin afectar OARs/PTV — impacto esperado en piel/grasa "
            f"de zona de dosis baja"
        )

    rec = calcular_recorte_plano(ct_sitk, cx, cy, INPLANE_CROP_MM)
    ct_norm        = aplicar_recorte_plano(ct_norm,        rec, fill_value=-1.0)
    dosis_norm_pct = aplicar_recorte_plano(dosis_norm_pct, rec, fill_value=0.0)
    ptv_mask       = aplicar_recorte_plano(ptv_mask,       rec, fill_value=0)
    body_mask      = aplicar_recorte_plano(body_mask,      rec, fill_value=0)
    rectum_mask    = aplicar_recorte_plano(rectum_mask,    rec, fill_value=0)
    bladder_mask   = aplicar_recorte_plano(bladder_mask,   rec, fill_value=0)

    # 8. Calcular PSDM sobre la máscara nativa YA RECORTADA (mismo spacing nativo — el recorte
    #    no resamplea, solo acota el FOV, así que la distancia física a cada estructura no
    #    cambia salvo muy cerca del borde nuevo de la caja).
    spacing_zyx_mm = (spacing_mm[2], spacing_mm[1], spacing_mm[0])

    psdm_ptv     = calcular_psdm(ptv_mask,     spacing_zyx_mm)
    psdm_rectum  = calcular_psdm(rectum_mask,  spacing_zyx_mm)
    psdm_bladder = calcular_psdm(bladder_mask, spacing_zyx_mm)

    # 9. Agrupar arrays y recortar axialmente
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

    # 10. Downsample in-plane a 256×256 (500mm/256 = 1.953mm/px isotrópico, fijo entre pacientes)
    for key in arrays:
        if isinstance(arrays[key], np.ndarray) and arrays[key].ndim == 3:
            arrays[key] = downsample_inplane(arrays[key], INPLANE_SIZE)

    # 11. QC mínimo
    n_slices  = arrays['ct'].shape[0]
    vol_ptv   = float(ptv_mask.sum()) * np.prod(spacing_mm) / 1000.0  # cc
    dose_max  = float(arrays['dose'].max())
    dose_ptv_d95 = float(np.percentile(arrays['dose'][arrays['ptv_mask'] > 0], 5))

    # 12. Guardar NPZ
    meta = {
        'anonid':         anonid,
        'spacing_mm':     list(spacing_mm),          # spacing NATIVO (sx, sy, sz) — no confundir
                                                       # con el spacing efectivo del array guardado
        'effective_inplane_spacing_mm': INPLANE_CROP_MM / INPLANE_SIZE,  # fijo: 1.953mm/px
        'crop_lado_mm':        INPLANE_CROP_MM,
        'centroide_ptv_xy_mm': [round(cx, 2), round(cy, 2)],
        'body_clip_mm':        clip_body,
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
        'body_clip_mm': clip_body,
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
        clipeados = [r for r in ok if r.get('body_clip_mm')]
        log.info(f"\n=== QC ===")
        log.info(f"N cortes/paciente: {np.mean(n_slices):.1f} ± {np.std(n_slices):.1f} [{min(n_slices)}-{max(n_slices)}]")
        log.info(f"Vol PTV (cc):      {np.mean(vol_ptv):.1f} ± {np.std(vol_ptv):.1f}")
        log.info(f"D95 PTV (% pres):  {np.mean(d95):.2f} ± {np.std(d95):.2f} (debe ser ~100)")
        log.info(f"BODY clipeado (lateral, esperado): {len(clipeados)}/{len(ok)} pacientes")
        for r in clipeados:
            log.info(f"  {r['anonid']}: {r['body_clip_mm']}")


if __name__ == "__main__":
    main()