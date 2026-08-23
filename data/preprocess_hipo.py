"""
Preprocesado DICOM → NPZ — dataset HIPOFRACCIONADO (28 fx x 2.5 Gy = 70 Gy).

Reusa los helpers DICOM/geometría de data/preprocess.py (CT, RD, RS, PSDM, resampleo,
downsample, recorte axial) y ajusta lo especifico del protocolo hipofraccionado:
  - Prescripcion_Gy = 70.0 (D95(PTV) = 100%).
  - Busqueda de PTV por lista de prioridad ampliada (ver PTV_ALIAS_PRIORITY).
  - Dos carpetas fuente de DICOMs: la mayoria de los pacientes esta en
    --dicom-root-primario; los 21 pacientes que ademas tienen un plan
    normofraccionado (pasados a normo) tienen su plan hipo en
    --dicom-root-fallback (mismo AnonID, RD/RP distintos).
  - meta enriquecido: s_D95, PTV_name_usado, D95_delivered_Gy, Status (del CSV).

Ver CLAUDE_CODE_CONTEXT.md / HIPOFX_KICKOFF.md para contexto de diseño.

Uso:
    python data/preprocess_hipo.py \
        --dicom-root-primario  "../dicoms hipofx" \
        --dicom-root-fallback  "../20260709_1834_Ptta_Hipo_filtrados" \
        --csv        "../dicoms hipofx/metricas_planes_hipofx_D95norm.csv" \
        --output-dir "../processed_hipo" \
        --splits     data/splits/splits_hipo_v1.json \
        --workers    1
"""

import argparse
import csv
import json
import logging
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import (  # noqa: E402
    cargar_ct,
    cargar_rd,
    cargar_estructuras,
    contornos_a_mascara,
    calcular_psdm,
    normalizar_dosis,
    remuestrear_a_ct,
    downsample_inplane,
    recortar_axial,
    centroide_fisico,
    extensiones_mascara,
    calcular_recorte_plano,
    aplicar_recorte_plano,
    INPLANE_SIZE,
    INPLANE_CROP_MM,
    Z_MARGIN,
    CT_HU_MIN,
    CT_HU_MAX,
    OPCIONES_BODY,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Constantes especificas hipofraccionado ──────────────────────────────────
PRESCRIPCION_GY  = 70.0
PRESCRIPCION_CGY = PRESCRIPCION_GY * 100.0

PTV_ALIAS_PRIORITY = ['PTV_High', 'PTV', 'PTV_High1', 'PTV_High2', 'PTV_Highrecor', 'PTV_Ptta']


# ─── CSV de referencia (Status, cross-check) ─────────────────────────────────

def cargar_csv(csv_path: Path) -> dict:
    """Devuelve dict {AnonID: fila_dict} leyendo el CSV con separador ';'."""
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {row["AnonID"]: row for row in reader}


def _tiene_plan_completo(carpeta: Path) -> bool:
    """Chequeo rapido: la carpeta tiene RD (dosis) ademas de CT/RS.
    Alguna carpeta de --dicom-root-primario esta incompleta (solo CT, sin
    RD/RS/RP) para pacientes que en realidad viven en --dicom-root-fallback."""
    return bool(list(carpeta.glob("RD*.dcm")) or list(carpeta.glob("rd*.dcm")))


def resolver_carpeta(anonid: str, root_primario: Path, root_fallback: Path) -> Path:
    """Busca la carpeta DICOM del paciente: primero en root_primario, luego en
    root_fallback (pacientes con plan hipo calculado tras ser pasados a normo).
    Si la carpeta primaria existe pero esta incompleta (sin RD), tambien cae
    a root_fallback."""
    carpeta_primaria = root_primario / anonid
    if carpeta_primaria.is_dir() and _tiene_plan_completo(carpeta_primaria):
        return carpeta_primaria
    carpeta_fallback = root_fallback / anonid
    if carpeta_fallback.is_dir() and _tiene_plan_completo(carpeta_fallback):
        return carpeta_fallback
    if carpeta_primaria.is_dir():
        return carpeta_primaria  # dejar que cargar_rd() lance el error original
    raise FileNotFoundError(
        f"{anonid}: no encontrado en {root_primario} ni en {root_fallback}"
    )


def buscar_ptv_por_prioridad(estructuras: dict, prioridad=PTV_ALIAS_PRIORITY):
    """Devuelve el nombre de estructura PTV a usar, siguiendo la lista de
    prioridad. Case-insensitive como fallback. None si ninguna aparece."""
    for op in prioridad:
        if op in estructuras:
            return op
    for nombre in estructuras:
        for op in prioridad:
            if nombre.lower() == op.lower():
                return nombre
    return None


def resolver_nombre_ptv(estructuras: dict, nombre_ptv_csv: str):
    """Determina el PTV a usar para este paciente.

    Prioridad 1: el nombre exacto que uso el C#/ESAPI para este paciente
    (columna NombrePTV del CSV) — es la fuente de verdad por-paciente, porque
    algunos RS tienen VARIAS estructuras PTV_* simultaneas (ej. 'PTV_High' Y
    'PTV_High1' en el mismo set) y una lista de prioridad global puede elegir
    la estructura equivocada para un paciente puntual (verificado en
    PT_5b54e7add30325f0: la lista global elegia 'PTV_High', pero ESAPI uso
    'PTV_High1' — factor de normalizacion completamente distinto: s=1.83 vs
    FactorNorm_D95 CSV=0.998).
    Prioridad 2 (fallback si el nombre del CSV no aparece en el RS, ej. fila
    de CSV faltante): lista de prioridad global PTV_ALIAS_PRIORITY.
    """
    if nombre_ptv_csv:
        if nombre_ptv_csv in estructuras:
            return nombre_ptv_csv
        for nombre in estructuras:
            if nombre.lower() == nombre_ptv_csv.lower():
                return nombre
    return buscar_ptv_por_prioridad(estructuras)


def buscar(estructuras: dict, opciones):
    for op in opciones:
        if op in estructuras:
            return op
    for nombre in estructuras:
        for op in opciones:
            if nombre.lower() == op.lower():
                return nombre
    return None


# ─── Procesado de un paciente ─────────────────────────────────────────────────

def procesar_paciente_hipo(anonid: str, carpeta_dicom: Path, output_dir: Path,
                            status: str, nombre_ptv_csv: str = "") -> dict:
    out_path = output_dir / f"{anonid}.npz"
    if out_path.exists():
        return {'anonid': anonid, 'status': 'skipped (ya existe)'}

    # 1. Cargar DICOM
    ct_sitk = cargar_ct(carpeta_dicom)
    rd_sitk = cargar_rd(carpeta_dicom)
    estructuras = cargar_estructuras(carpeta_dicom)

    # 2. Identificar PTV: NombrePTV del CSV (fuente de verdad por-paciente)
    #    con fallback a la lista de prioridad global (loguear cual se uso)
    nombre_ptv = resolver_nombre_ptv(estructuras, nombre_ptv_csv)
    if nombre_ptv is None:
        raise ValueError(
            f"PTV no encontrado (csv='{nombre_ptv_csv}', prioridad={PTV_ALIAS_PRIORITY}). "
            f"Estructuras disponibles: {list(estructuras.keys())}"
        )
    nombre_body    = buscar(estructuras, OPCIONES_BODY)
    nombre_rectum  = buscar(estructuras, ['Rectum'])
    nombre_bladder = buscar(estructuras, ['Bladder'])

    # 3. Mascaras en espacio CT
    def hacer_mascara(nombre):
        if nombre is None:
            sz = ct_sitk.GetSize()
            return np.zeros((sz[2], sz[1], sz[0]), dtype=np.uint8)
        return contornos_a_mascara(estructuras[nombre], ct_sitk)

    ptv_mask     = hacer_mascara(nombre_ptv)
    body_mask    = hacer_mascara(nombre_body)
    rectum_mask  = hacer_mascara(nombre_rectum)
    bladder_mask = hacer_mascara(nombre_bladder)

    # 4. Dosis remuestreada al espacio CT (dosis "nativa" a efectos de este pipeline,
    #    antes de escalar por s y antes de downsamplear/recortar)
    dosis_cgy = remuestrear_a_ct(rd_sitk, ct_sitk, sitk.sitkLinear)

    # 5. CT normalizada [-1, 1] (sobre grilla nativa completa, antes de recortar)
    ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    ct_norm  = np.clip(ct_array, CT_HU_MIN, CT_HU_MAX)
    ct_norm  = (ct_norm - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)
    ct_norm  = ct_norm * 2.0 - 1.0

    # 6. D95 nativo (Gy) y s = 70.0 / D95(PTV) — mismo criterio que el C# (sobre mascara nativa
    #    completa, sin recortar: el PTV siempre entra en la caja de recorte)
    dosis_ptv_cgy = dosis_cgy[ptv_mask > 0]
    if len(dosis_ptv_cgy) == 0:
        raise ValueError("Mascara PTV vacia — no se puede calcular D95")
    d95_nativo_cgy = float(np.percentile(dosis_ptv_cgy, 5))
    d95_delivered_gy = d95_nativo_cgy / 100.0

    # 7. Normalizar dosis a D95(PTV) = 100% de 70 Gy (reusa normalizar_dosis del normo,
    #    solo cambia prescripcion_cgy)
    dosis_norm_pct, s_d95 = normalizar_dosis(dosis_cgy, ptv_mask, prescripcion_cgy=PRESCRIPCION_CGY)

    spacing_mm = ct_sitk.GetSpacing()

    # 7b. Recorte en el plano (misma caja de INPLANE_CROP_MM que normo, centrada en el
    #     centroide del PTV) — ver preprocess.py::procesar_paciente para el razonamiento
    #     completo. OAR/PTV clipeado -> error (revisar a mano); BODY clipeado -> warning.
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

    # 8. PSDM sobre la mascara nativa YA RECORTADA (mismo spacing nativo)
    spacing_zyx_mm = (spacing_mm[2], spacing_mm[1], spacing_mm[0])

    psdm_ptv     = calcular_psdm(ptv_mask,     spacing_zyx_mm)
    psdm_rectum  = calcular_psdm(rectum_mask,  spacing_zyx_mm)
    psdm_bladder = calcular_psdm(bladder_mask, spacing_zyx_mm)

    # 9. Agrupar, recortar axialmente, downsample in-plane
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

    for key in arrays:
        if isinstance(arrays[key], np.ndarray) and arrays[key].ndim == 3:
            arrays[key] = downsample_inplane(arrays[key], INPLANE_SIZE)

    # 10. QC minimo
    n_slices = arrays['ct'].shape[0]
    vol_ptv_cc = float(ptv_mask.sum()) * np.prod(spacing_mm) / 1000.0
    dose_max = float(arrays['dose'].max())
    dose_ptv_d95_downsampled = float(np.percentile(arrays['dose'][arrays['ptv_mask'] > 0], 5))

    # 11. Guardar NPZ
    meta = {
        'anonid':          anonid,
        'spacing_mm':      list(spacing_mm),  # spacing NATIVO — no confundir con el efectivo del array
        'effective_inplane_spacing_mm': INPLANE_CROP_MM / INPLANE_SIZE,  # fijo: 1.953mm/px
        'crop_lado_mm':        INPLANE_CROP_MM,
        'centroide_ptv_xy_mm': [round(cx, 2), round(cy, 2)],
        'body_clip_mm':        clip_body,
        'z_range':         list(z_range),
        'factor_norm':     s_d95,
        's_D95':           s_d95,
        'D95_delivered_Gy': round(d95_delivered_gy, 4),
        'PTV_name_usado':  nombre_ptv,
        'nombre_ptv':      nombre_ptv,
        'nombre_rectum':   nombre_rectum or '',
        'nombre_bladder':  nombre_bladder or '',
        'vol_ptv_cc':      round(vol_ptv_cc, 2),
        'n_slices':        n_slices,
        'Status':          status,
        'prescripcion_Gy': PRESCRIPCION_GY,
        'n_fracciones':    28,
    }

    np.savez_compressed(
        str(out_path),
        ct           = arrays['ct'],
        dose         = arrays['dose'],
        ptv_mask     = arrays['ptv_mask'],
        body_mask    = arrays['body_mask'],
        rectum_mask  = arrays['rectum_mask'],
        bladder_mask = arrays['bladder_mask'],
        psdm_ptv     = arrays['psdm_ptv'],
        psdm_rectum  = arrays['psdm_rectum'],
        psdm_bladder = arrays['psdm_bladder'],
        meta         = np.array([json.dumps(meta)]),
    )

    return {
        'anonid':            anonid,
        'status':            'ok',
        'n_slices':           n_slices,
        'vol_ptv_cc':         round(vol_ptv_cc, 2),
        'dose_max':           round(dose_max, 2),
        'dose_d95_ptv_downsampled': round(dose_ptv_d95_downsampled, 2),
        's_D95':              round(s_d95, 4),
        'D95_delivered_Gy':   round(d95_delivered_gy, 4),
        'PTV_name_usado':     nombre_ptv,
        'Status':             status,
        'body_clip_mm':       clip_body,
    }


# ─── Worker para multiprocessing ─────────────────────────────────────────────

def _worker(args):
    anonid, root_primario, root_fallback, output_dir, status, nombre_ptv_csv = args
    try:
        carpeta = resolver_carpeta(anonid, Path(root_primario), Path(root_fallback))
        return procesar_paciente_hipo(anonid, carpeta, Path(output_dir), status, nombre_ptv_csv)
    except Exception as e:
        return {'anonid': anonid, 'status': f'ERROR: {e}'}


# ─── CLI principal ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocesado DICOM -> NPZ (hipofraccionado)")
    parser.add_argument("--dicom-root-primario", required=True, help="Carpeta principal de DICOMs hipofx")
    parser.add_argument("--dicom-root-fallback", required=True, help="Carpeta fallback (pacientes pasados a normo)")
    parser.add_argument("--csv", required=True, help="CSV metricas_planes_hipofx_D95norm.csv (Status)")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida para NPZs")
    parser.add_argument("--splits", required=True, help="JSON con splits train/val/test hipo")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", nargs="+", help="Procesar solo estos AnonIDs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.splits) as f:
        splits = json.load(f)
    excluidos = set(splits.get('excluidos', []))
    todos = splits['train'] + splits['val'] + splits['test']
    anonids = [a for a in todos if a not in excluidos]

    filas_csv = cargar_csv(Path(args.csv))

    if args.only:
        anonids = [a for a in anonids if a in set(args.only)]
        log.info(f"Modo --only: procesando {len(anonids)} pacientes")

    log.info(f"Total a procesar: {len(anonids)} pacientes")
    log.info(f"Output: {output_dir}")
    log.info(f"Workers: {args.workers}")

    worker_args = [
        (a, args.dicom_root_primario, args.dicom_root_fallback, args.output_dir,
         filas_csv.get(a, {}).get('Status', ''),
         filas_csv.get(a, {}).get('NombrePTV', ''))
        for a in anonids
    ]

    resultados = []
    if args.workers == 1:
        for wa in tqdm(worker_args, desc="Preprocesando"):
            resultados.append(_worker(wa))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, wa): wa[0] for wa in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocesando"):
                resultados.append(future.result())

    ok   = [r for r in resultados if r['status'] == 'ok']
    skip = [r for r in resultados if 'skipped' in r['status']]
    err  = [r for r in resultados if 'ERROR' in r['status']]

    log.info("\n=== Resumen ===")
    log.info(f"OK:      {len(ok)}")
    log.info(f"Skipped: {len(skip)}")
    log.info(f"Errores (warning de PTV u otro fallo): {len(err)}")

    if err:
        log.warning("Errores:")
        for r in err:
            log.warning(f"  {r['anonid']}: {r['status']}")

    log_path = output_dir / "preprocess_hipo_log.json"
    with open(log_path, 'w') as f:
        json.dump(resultados, f, indent=2)
    log.info(f"Log guardado en {log_path}")

    if ok:
        n_slices = [r['n_slices'] for r in ok]
        vol_ptv  = [r['vol_ptv_cc'] for r in ok]
        d95_down = [r['dose_d95_ptv_downsampled'] for r in ok]
        s_d95    = [r['s_D95'] for r in ok]
        from collections import Counter
        ptv_usado = Counter(r['PTV_name_usado'] for r in ok)

        log.info("\n=== QC ===")
        log.info(f"N cortes/paciente: {np.mean(n_slices):.1f} +/- {np.std(n_slices):.1f} [{min(n_slices)}-{max(n_slices)}]")
        log.info(f"Vol PTV (cc):      {np.mean(vol_ptv):.1f} +/- {np.std(vol_ptv):.1f}")
        log.info(f"s_D95 usado:       min={min(s_d95):.4f} median={float(np.median(s_d95)):.4f} max={max(s_d95):.4f}")
        log.info(f"D95 PTV downsampleado (% pres): {np.mean(d95_down):.2f} +/- {np.std(d95_down):.2f} (deberia ser ~100)")
        log.info(f"PTV usado por paciente: {dict(ptv_usado)}")

        clipeados = [r for r in ok if r.get('body_clip_mm')]
        log.info(f"BODY clipeado (lateral, esperado): {len(clipeados)}/{len(ok)} pacientes")
        for r in clipeados:
            log.info(f"  {r['anonid']}: {r['body_clip_mm']}")


if __name__ == "__main__":
    main()
