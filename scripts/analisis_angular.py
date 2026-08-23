"""
Analisis diagnostico — estructura angular de la dosis VMAT (validacion pre-arc-prior).

Testea 4 hipotesis (ver RESUMEN_prior_terma_arcos.md / CLAUDE_CODE_CONTEXT):
  H1: la dosis real tiene estructura angular (estrias) en dosis media-baja.
  H2: esa estructura sigue la POSICION de los OAR (evitacion anatomica), no es solo
      un artefacto mecanico de muestreo de control points.
  H3: la amplitud crece en planes mas forzados (overlap PTV-OAR, no-cumplidores).
  H4: el RESIDUO del modelo (pred-real) tiene la misma estructura angular → el modelo
      no la captura y un canal de geometria de arcos podria ayudar.

Puramente diagnostico: NO entrena nada, NO toca configs/splits. Solo inferencia
(reusa el checkpoint ya entrenado exp_hipo_002b_finetune_clean) + analisis espectral.

Dataset: val+test de splits_hipo_v2_clean_balanced.json (22+31=53 pacientes, dataset
limpio ya sin contaminacion nodal). No se usa train (evita cualquier señal de overfitting
del modelo en la comparacion pred vs real, aunque el objetivo aca es diagnostico de la
dosis REAL principalmente, no de generalizacion).

⚠️ LIMITACION DE DATOS (ver hallazgo durante este analisis): el CSV con features
geometricas completas (metricas_planes_hipofx_D95norm_clean.csv, con NArcos y MUs) vivia
en C:\\Pablo\\ProstateDoseProject\\dicoms hipofx\\, carpeta que ya no existe en disco
(limpieza de espacio, ver memoria de proyecto). Por eso:
  - overlap_rel_recto/vejiga y volumenes SE RECALCULAN directo desde las mascaras del NPZ
    (mismo patron de calibracion de voxel que compute_overlap_real.py — de hecho MAS
    correcto que el CSV viejo, que tenia el bug Math.Min conocido).
  - Flags de cumplimiento vienen de data/gt_dvh_hipo_256.csv (fuente de verdad ya
    establecida en el proyecto, D5).
  - MUs y NArcos NO estan disponibles para este cohorte → el control "MUs" de H3 y la
    ESTRATIFICACION TRANSVERSAL por NArcos NO se pueden ejecutar. Se documenta como
    hallazgo/limitacion en el summary, no se inventa un proxy.

Uso:
    .venv/Scripts/python.exe scripts/analisis_angular.py
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import stats
from scipy.ndimage import map_coordinates
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate import cargar_modelo  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Configuracion
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT = _REPO_ROOT / "checkpoints/exp_hipo_002b_finetune_clean/epoch=028.ckpt"
CONFIG_YAML = _REPO_ROOT / "configs/exp_hipo_002b_finetune_clean.yaml"
PROCESSED_DIR = Path(r"C:\Pablo\ProstateDoseProject\processed_hipo")
SPLITS_PATH = _REPO_ROOT / "data/splits/splits_hipo_v2_clean_balanced.json"
GT_DVH_CSV = _REPO_ROOT / "data/gt_dvh_hipo_256.csv"

OUT_DIR = _REPO_ROOT / "results/analisis_angular"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RADII_MULT = [1.5, 2.0, 2.5, 3.0]          # multiplos de R_eq(PTV)
N_THETA = 180                               # muestras angulares (paso 2 grados)
BAND_LOW = (1, 3)                           # confounder: forma del cuerpo
BAND_MID = (4, 12)                          # banda de interes ("estrias")
BAND_HIGH_START = 13                        # ruido: n >= 13 (hasta Nyquist=89)
MIN_BODY_COVERAGE = 0.5                     # min fraccion del circulo dentro de BODY
MIN_PTV_PIXELS = 15                         # min area PTV en la slice para usarla
MIN_OAR_PIXELS = 5                          # min area OAR para computar su angulo

PRESCRIPCION_GY = 70.0


# ──────────────────────────────────────────────────────────────────────────────
# Geometria / muestreo angular
# ──────────────────────────────────────────────────────────────────────────────

def centroide(mask_2d: np.ndarray):
    """(row, col) del centroide de una mascara binaria 2D. None si vacia."""
    ys, xs = np.nonzero(mask_2d)
    if len(ys) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def angulo_desde_centro(row_c, col_c, row_p, col_p) -> float:
    """Angulo (rad, [0,2pi)) de un punto respecto al centro, convencion:
    theta=0 hacia ANTERIOR (row decreciente, verificado empiricamente — ver
    encabezado del script y log de verificacion impreso al arrancar), creciendo
    hacia un lado (arbitrario, no verificado L/R — no afecta ninguna metrica
    usada aca, ver docstring del modulo)."""
    u = col_p - col_c           # eje "lateral" (sin verificar L/R)
    v = row_c - row_p           # eje anterior-posterior: v>0 = anterior
    ang = np.arctan2(u, v)
    return ang % (2 * np.pi)


def muestrear_circulo(arr_2d: np.ndarray, row_c: float, col_c: float, r_px: float,
                       n_theta: int = N_THETA) -> tuple:
    """Muestrea arr_2d sobre un circulo de radio r_px centrado en (row_c,col_c).
    Devuelve (thetas [rad, 0..2pi), valores interpolados bilineal)."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    # misma convencion que angulo_desde_centro: u=r*sin(theta) (lateral), v=r*cos(theta) (ant-post)
    rows = row_c - r_px * np.cos(thetas)
    cols = col_c + r_px * np.sin(thetas)
    coords = np.stack([rows, cols], axis=0)
    valores = map_coordinates(arr_2d, coords, order=1, mode="constant", cval=0.0)
    return thetas, valores


def llenar_gaps_circular(valores: np.ndarray, validos: np.ndarray) -> np.ndarray:
    """Interpola linealmente (circular) los angulos invalidos (fuera de BODY)."""
    n = len(valores)
    if validos.all():
        return valores.copy()
    idx_validos = np.where(validos)[0]
    out = valores.copy()
    # Extender circularmente triplicando para interp lineal simple con wraparound
    idx_ext = np.concatenate([idx_validos - n, idx_validos, idx_validos + n])
    val_ext = np.concatenate([valores[idx_validos]] * 3)
    idx_todos = np.arange(n)
    out = np.interp(idx_todos, idx_ext, val_ext)
    return out


def espectro_amplitud(signal: np.ndarray) -> np.ndarray:
    """Amplitud (RMS de cada armonico, a_n = 2|X[n]|/N) de un rfft real, n=0..N/2."""
    n = len(signal)
    X = np.fft.rfft(signal)
    a = np.abs(X) * (2.0 / n)
    a[0] = a[0] / 2.0  # DC: no se dobla
    return a  # longitud n//2+1


def metricas_banda(signal: np.ndarray, denom: float = None) -> dict:
    """amp_estrias (banda media, adimensional vs. la dosis media) + bandas de
    control (baja=confounder forma, alta=ruido) + reconstruccion banda-media
    (para localizar el angulo de la estructura, filtrando forma de cuerpo y ruido).

    `denom`: normalizador explicito. Por defecto (None) usa la media del propio
    signal — correcto para la dosis REAL/PREDICHA (misma escala, dosis en %).
    Para el RESIDUO (pred-real) hay que pasar denom=mean(dosis real) explicito:
    el residuo tiene media ~0 (error no sesgado), asi que normalizar por su
    propia media dividiria por ~0 y no es comparable en escala con real/pred."""
    mean_val = float(np.mean(signal))
    a = espectro_amplitud(signal)
    if denom is None:
        denom = abs(mean_val) if abs(mean_val) > 1e-6 else 1e-6

    def rms_banda(lo, hi):
        return float(np.sqrt(np.sum(a[lo:hi + 1] ** 2)))

    amp_low = rms_banda(*BAND_LOW) / denom
    amp_mid = rms_banda(*BAND_MID) / denom
    amp_high = rms_banda(BAND_HIGH_START, len(a) - 1) / denom

    # Reconstruccion banda-media (n=4..12) para localizar el angulo de la estructura
    n = len(signal)
    X = np.fft.rfft(signal)
    X_band = np.zeros_like(X)
    X_band[BAND_MID[0]:BAND_MID[1] + 1] = X[BAND_MID[0]:BAND_MID[1] + 1]
    recon = np.fft.irfft(X_band, n=n)

    return {
        "amp_low": amp_low, "amp_mid": amp_mid, "amp_high": amp_high,
        "mean_dose_circulo": mean_val,
        "theta_min_idx": int(np.argmin(recon)), "theta_max_idx": int(np.argmax(recon)),
        "recon_band": recon,
        "spectrum": a,
    }


def amp_mid_null(signal: np.ndarray, rng: np.random.Generator, n_shuffles: int = 8) -> float:
    """Control de H1: baraja el orden angular (destruye la estructura, preserva
    el histograma de valores) y mide amp_mid resultante — piso de ruido esperado
    por azar para esta banda, con esta cantidad de muestras."""
    vals = []
    n = len(signal)
    for _ in range(n_shuffles):
        perm = rng.permutation(n)
        vals.append(metricas_banda(signal[perm])["amp_mid"])
    return float(np.mean(vals))


def circmean_deg(angulos_deg: np.ndarray) -> float:
    rad = np.deg2rad(angulos_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360)


def circstd_deg(angulos_deg: np.ndarray) -> float:
    rad = np.deg2rad(angulos_deg)
    R = np.sqrt(np.mean(np.cos(rad)) ** 2 + np.mean(np.sin(rad)) ** 2)
    R = min(R, 1.0 - 1e-12)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


def diff_angular_deg(a_deg: float, b_deg: float) -> float:
    """a-b envuelto a (-180, 180]."""
    d = (a_deg - b_deg + 180) % 360 - 180
    return float(d)


def rayleigh_test(angulos_deg: np.ndarray) -> dict:
    """Test de uniformidad de Rayleigh (Zar 1999, Biostatistical Analysis, aprox. p)."""
    rad = np.deg2rad(angulos_deg)
    n = len(rad)
    C, S = np.sum(np.cos(rad)), np.sum(np.sin(rad))
    R = np.sqrt(C ** 2 + S ** 2) / n
    z = n * R ** 2
    p = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n)
                       - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2))
    p = float(np.clip(p, 0.0, 1.0))
    return {"n": int(n), "R": float(R), "z": float(z), "p": p,
            "media_circular_deg": circmean_deg(angulos_deg)}


# ──────────────────────────────────────────────────────────────────────────────
# Calibracion de spacing in-plane (mascaras downsampleadas a 256, spacing meta
# es nativo — ver CLAUDE_CODE_CONTEXT.md, bug conocido). Mismo patron que
# compute_overlap_real.py: calibrar via vol_ptv_cc nativo / voxeles PTV downsampleados.
# Asume in-plane isotropico tras downsample (spacing nativo sx==sy verificado en
# muestra de 15 pacientes, ver log de arranque).
# ──────────────────────────────────────────────────────────────────────────────

def spacing_efectivo_mm(meta: dict, ptv_mask: np.ndarray) -> float:
    n_vox_ptv = float(ptv_mask.sum())
    sp_z = float(meta["spacing_mm"][2])
    vol_ptv_cc = float(meta["vol_ptv_cc"])
    if n_vox_ptv < 1 or sp_z <= 0 or vol_ptv_cc <= 0:
        return float(meta["spacing_mm"][0])  # fallback: spacing nativo (aprox)
    voxel_area_mm2 = vol_ptv_cc * 1000.0 / (n_vox_ptv * sp_z)
    return float(np.sqrt(voxel_area_mm2))


# ──────────────────────────────────────────────────────────────────────────────
# Overlap real (recalculado desde mascaras, evita el bug Math.Min del CSV viejo)
# ──────────────────────────────────────────────────────────────────────────────

def overlap_y_volumenes(meta: dict, ptv_mask: np.ndarray, rectum_mask: np.ndarray,
                         bladder_mask: np.ndarray) -> dict:
    sp = meta["spacing_mm"]
    sp_eff = spacing_efectivo_mm(meta, ptv_mask)
    voxel_cc = (sp_eff ** 2) * sp[2] / 1000.0

    vol_ptv_cc = float(meta["vol_ptv_cc"])  # nativo, ya en meta (sin bug)
    vol_rectum_cc = float(rectum_mask.sum()) * voxel_cc
    vol_bladder_cc = float(bladder_mask.sum()) * voxel_cc
    solap_rectum_cc = float(((ptv_mask > 0) & (rectum_mask > 0)).sum()) * voxel_cc
    solap_bladder_cc = float(((ptv_mask > 0) & (bladder_mask > 0)).sum()) * voxel_cc

    return {
        "VolPTV_cc": vol_ptv_cc,
        "VolRectum_cc": vol_rectum_cc,
        "VolBladder_cc": vol_bladder_cc,
        "Solap_PTV_Rectum_cc": solap_rectum_cc,
        "Solap_PTV_Bladder_cc": solap_bladder_cc,
        "overlap_rel_recto": solap_rectum_cc / vol_rectum_cc if vol_rectum_cc > 0 else float("nan"),
        "overlap_rel_vejiga": solap_bladder_cc / vol_bladder_cc if vol_bladder_cc > 0 else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Inferencia + analisis angular de 1 paciente
# ──────────────────────────────────────────────────────────────────────────────

def cargar_npz(npz_path: Path) -> tuple:
    data = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    arrays = {
        "ct": np.array(data["ct"], dtype=np.float32),
        "dose": np.array(data["dose"], dtype=np.float32),
        "ptv_mask": np.array(data["ptv_mask"], dtype=np.uint8),
        "body_mask": np.array(data["body_mask"], dtype=np.uint8),
        "rectum_mask": np.array(data["rectum_mask"], dtype=np.uint8),
        "bladder_mask": np.array(data["bladder_mask"], dtype=np.uint8),
        "psdm_ptv": np.array(data["psdm_ptv"], dtype=np.float32),
        "psdm_rectum": np.array(data["psdm_rectum"], dtype=np.float32),
        "psdm_bladder": np.array(data["psdm_bladder"], dtype=np.float32),
    }
    data.close()
    return arrays, meta


def calcular_n_slices_target(processed_dir: Path, splits_path: Path) -> int:
    """Replica DoseDataModule._calcular_n_slices_target (max Z del TRAIN,
    redondeado al multiplo de 8 por arriba). Necesario para pad_z_to de
    inferir_dosis al evaluar un modelo arch='unet3d' fuera del DataModule."""
    with open(splits_path) as f:
        splits = json.load(f)
    n_slices_list = []
    for aid in splits["train"]:
        p = processed_dir / f"{aid}.npz"
        if not p.exists():
            continue
        data = np.load(str(p), allow_pickle=True)
        n_slices_list.append(int(data["ct"].shape[0]))
        data.close()
    n_max = max(n_slices_list)
    return ((n_max + 7) // 8) * 8


def inferir_dosis(model, device, arrays: dict, pad_z_to: int = None) -> np.ndarray:
    """
    pad_z_to: OBLIGATORIO para modelos arch='unet3d' (usar calcular_n_slices_target
    con el mismo splits/processed_dir del checkpoint). Sin esto, el MAE se
    dispara (verificado: 7.44 vs 1.60 de BODY-MAE en un paciente de prueba de
    exp_normo_3dunet, este ultimo identico al numero oficial de evaluate.py)
    porque el 3D fue entrenado SIEMPRE con Z padeado a un tamano fijo
    (DoseDataModule._pad_or_crop_z: CT=-1, PSDM=1, masks=0 fuera de la
    anatomia real, padding simetrico) y un forward con Z nativo (sin padding)
    lo saca de esa distribucion — los kernels 3D SI cruzan Z, a diferencia
    del 2D (que procesa cada corte independiente y por eso nunca necesito este
    parametro). Para 2D, dejar pad_z_to=None (default, sin cambios de comportamiento).
    """
    z_native = arrays["ct"].shape[0]
    pad_before = 0
    if pad_z_to is not None and pad_z_to > z_native:
        pad_total = pad_z_to - z_native
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

    def _prep(key: str, pad_value: float) -> torch.Tensor:
        t = torch.from_numpy(arrays[key].astype(np.float32)).unsqueeze(0)
        if pad_before or (pad_z_to is not None and pad_z_to > z_native):
            t = F.pad(t, (0, 0, 0, 0, pad_before, pad_z_to - z_native - pad_before), value=pad_value)
        return t.to(device)

    batch = {
        "ct":            _prep("ct", -1.0),
        "body_mask":     _prep("body_mask", 0.0),
        "psdm_ptv":      _prep("psdm_ptv", 1.0),
        "psdm_rectum":   _prep("psdm_rectum", 1.0),
        "psdm_bladder":  _prep("psdm_bladder", 1.0),
    }
    with torch.no_grad():
        x = model._build_input(batch)
        pred = model(x)
    pred = pred[0].cpu().numpy().astype(np.float32)
    if pad_before or (pad_z_to is not None and pad_z_to > z_native):
        pred = pred[pad_before:pad_before + z_native]
    return pred


def analizar_paciente(anonid: str, arrays: dict, meta: dict, dose_pred: np.ndarray) -> dict:
    """Corre el analisis angular completo (todas las slices/radios) para 1 paciente
    y devuelve un dict agregado (fila del CSV final) + datos crudos para plots."""
    dose_real = arrays["dose"]
    ptv = arrays["ptv_mask"]
    body = arrays["body_mask"].astype(np.float32)
    rectum = arrays["rectum_mask"]
    bladder = arrays["bladder_mask"]
    residuo = dose_pred - dose_real

    sp_eff = spacing_efectivo_mm(meta, ptv)
    z_con_ptv = [z for z in range(ptv.shape[0]) if ptv[z].sum() >= MIN_PTV_PIXELS]

    # Por radio: listas de valores por-slice (para agregar por paciente)
    por_radio = {m: {"amp_low": {"real": [], "pred": [], "res": []},
                      "amp_mid": {"real": [], "pred": [], "res": []},
                      "amp_high": {"real": [], "pred": [], "res": []},
                      "amp_mid_null_real": [],
                      "mean_dose": {"real": [], "pred": []},
                      "theta_min_real_deg": [], "theta_max_real_deg": [],
                      "theta_min_pred_deg": [], "theta_min_res_deg": [],
                      "peso_amp_mid_real": [],
                      "recon_real": [], "recon_pred": [], "recon_res": [],
                      "n_slices_validas": 0, "n_slices_totales": 0}
                 for m in RADII_MULT}
    rng = np.random.default_rng(abs(hash(anonid)) % (2 ** 32))

    theta_rectum_list, theta_bladder_list = [], []
    ejemplo_slices = {}  # para plots: guarda D(theta) crudo de algunas slices

    for z in z_con_ptv:
        cen = centroide(ptv[z])
        if cen is None:
            continue
        row_c, col_c = cen
        area_px = float(ptv[z].sum())
        r_eq_px = np.sqrt(area_px / np.pi)

        cen_r = centroide(rectum[z]) if rectum[z].sum() >= MIN_OAR_PIXELS else None
        cen_b = centroide(bladder[z]) if bladder[z].sum() >= MIN_OAR_PIXELS else None
        theta_rectum = angulo_desde_centro(row_c, col_c, *cen_r) if cen_r else None
        theta_bladder = angulo_desde_centro(row_c, col_c, *cen_b) if cen_b else None
        if theta_rectum is not None:
            theta_rectum_list.append(np.degrees(theta_rectum))
        if theta_bladder is not None:
            theta_bladder_list.append(np.degrees(theta_bladder))

        for m in RADII_MULT:
            por_radio[m]["n_slices_totales"] += 1
            r_px = m * r_eq_px

            thetas, body_s = muestrear_circulo(body[z], row_c, col_c, r_px)
            validos = body_s > 0.5
            cobertura = validos.mean()
            if cobertura < MIN_BODY_COVERAGE:
                continue

            _, real_s = muestrear_circulo(dose_real[z], row_c, col_c, r_px)
            _, pred_s = muestrear_circulo(dose_pred[z], row_c, col_c, r_px)
            _, res_s = muestrear_circulo(residuo[z], row_c, col_c, r_px)

            real_s = llenar_gaps_circular(real_s, validos)
            pred_s = llenar_gaps_circular(pred_s, validos)
            res_s = llenar_gaps_circular(res_s, validos)

            met_real = metricas_banda(real_s)
            denom_real = abs(met_real["mean_dose_circulo"]) if abs(met_real["mean_dose_circulo"]) > 1e-6 else 1e-6
            met_pred = metricas_banda(pred_s, denom=denom_real)
            met_res = metricas_banda(res_s, denom=denom_real)

            theta_grid_deg = np.degrees(thetas)
            por_radio[m]["amp_low"]["real"].append(met_real["amp_low"])
            por_radio[m]["amp_low"]["pred"].append(met_pred["amp_low"])
            por_radio[m]["amp_low"]["res"].append(met_res["amp_low"])
            por_radio[m]["amp_mid"]["real"].append(met_real["amp_mid"])
            por_radio[m]["amp_mid"]["pred"].append(met_pred["amp_mid"])
            por_radio[m]["amp_mid"]["res"].append(met_res["amp_mid"])
            por_radio[m]["amp_high"]["real"].append(met_real["amp_high"])
            por_radio[m]["amp_high"]["pred"].append(met_pred["amp_high"])
            por_radio[m]["amp_high"]["res"].append(met_res["amp_high"])
            por_radio[m]["mean_dose"]["real"].append(met_real["mean_dose_circulo"])
            por_radio[m]["mean_dose"]["pred"].append(met_pred["mean_dose_circulo"])
            por_radio[m]["theta_min_real_deg"].append(theta_grid_deg[met_real["theta_min_idx"]])
            por_radio[m]["theta_max_real_deg"].append(theta_grid_deg[met_real["theta_max_idx"]])
            por_radio[m]["theta_min_pred_deg"].append(theta_grid_deg[met_pred["theta_min_idx"]])
            por_radio[m]["theta_min_res_deg"].append(theta_grid_deg[met_res["theta_min_idx"]])
            por_radio[m]["peso_amp_mid_real"].append(met_real["amp_mid"])
            por_radio[m]["amp_mid_null_real"].append(amp_mid_null(real_s, rng))
            por_radio[m]["recon_real"].append(met_real["recon_band"])
            por_radio[m]["recon_pred"].append(met_pred["recon_band"])
            por_radio[m]["recon_res"].append(met_res["recon_band"])
            por_radio[m]["n_slices_validas"] += 1

            if m == 2.0:
                ejemplo_slices[z] = {
                    "theta_deg": theta_grid_deg, "real": real_s, "pred": pred_s, "res": res_s,
                    "theta_rectum_deg": np.degrees(theta_rectum) if theta_rectum is not None else None,
                    "theta_bladder_deg": np.degrees(theta_bladder) if theta_bladder is not None else None,
                }

    fila = {"AnonID": anonid, "sp_eff_mm": sp_eff,
            "n_slices_con_ptv": len(z_con_ptv)}

    for m in RADII_MULT:
        d = por_radio[m]
        suf = f"r{m}"
        fila[f"n_slices_validas_{suf}"] = d["n_slices_validas"]
        for banda in ["amp_low", "amp_mid", "amp_high"]:
            for fuente in ["real", "pred", "res"]:
                vals = d[banda][fuente]
                fila[f"{banda}_{fuente}_{suf}"] = float(np.mean(vals)) if vals else float("nan")
        for fuente in ["real", "pred"]:
            vals = d["mean_dose"][fuente]
            fila[f"mean_dose_{fuente}_{suf}"] = float(np.mean(vals)) if vals else float("nan")
        vals_null = d["amp_mid_null_real"]
        fila[f"amp_mid_null_real_{suf}"] = float(np.mean(vals_null)) if vals_null else float("nan")

        pesos = np.array(d["peso_amp_mid_real"])
        for nombre_theta in ["theta_min_real_deg", "theta_max_real_deg",
                              "theta_min_pred_deg", "theta_min_res_deg"]:
            vals = np.array(d[nombre_theta])
            if len(vals) == 0:
                fila[f"{nombre_theta}_{suf}"] = float("nan")
                continue
            # circular mean ponderado por amp_mid real (mas peso a slices con estructura clara)
            w = pesos if pesos.sum() > 0 else np.ones_like(vals)
            rad = np.deg2rad(vals)
            ang = np.degrees(np.arctan2(np.sum(w * np.sin(rad)), np.sum(w * np.cos(rad)))) % 360
            fila[f"{nombre_theta}_{suf}"] = float(ang)

    fila["theta_rectum_deg"] = circmean_deg(np.array(theta_rectum_list)) if theta_rectum_list else float("nan")
    fila["theta_bladder_deg"] = circmean_deg(np.array(theta_bladder_list)) if theta_bladder_list else float("nan")
    fila["n_slices_con_rectum"] = len(theta_rectum_list)
    fila["n_slices_con_bladder"] = len(theta_bladder_list)

    # overlap/volumenes (recalculados desde mascaras, ver docstring del modulo)
    fila.update(overlap_y_volumenes(meta, ptv, rectum, bladder))

    crudo = {"por_radio": por_radio, "ejemplo_slices": ejemplo_slices,
             "theta_rectum_list": theta_rectum_list, "theta_bladder_list": theta_bladder_list}
    return fila, crudo


# ──────────────────────────────────────────────────────────────────────────────
# H1 — ¿hay estructura angular en la dosis real?
# ──────────────────────────────────────────────────────────────────────────────

def test_h1(df: pd.DataFrame) -> dict:
    out = {}
    for m in RADII_MULT:
        suf = f"r{m}"
        real = df[f"amp_mid_real_{suf}"].dropna()
        null = df.loc[real.index, f"amp_mid_null_real_{suf}"]
        w = stats.wilcoxon(real, null, alternative="greater")
        out[suf] = {
            "n": int(len(real)),
            "amp_mid_real_mean": float(real.mean()), "amp_mid_real_std": float(real.std()),
            "amp_mid_null_mean": float(null.mean()), "amp_mid_null_std": float(null.std()),
            "amp_low_real_mean": float(df[f"amp_low_real_{suf}"].mean()),
            "amp_high_real_mean": float(df[f"amp_high_real_{suf}"].mean()),
            "ratio_real_vs_null": float(real.mean() / null.mean()) if null.mean() > 0 else float("nan"),
            "wilcoxon_p": float(w.pvalue),
        }
    # Veredicto sobre el radio "headline" r=2.0
    h = out["r2.0"]
    if h["wilcoxon_p"] < 0.01 and h["ratio_real_vs_null"] > 1.3:
        veredicto = "CONFIRMADA"
    elif h["wilcoxon_p"] < 0.05:
        veredicto = "CONFIRMADA (debil)"
    else:
        veredicto = "RECHAZADA"
    out["veredicto"] = veredicto
    return out


# ──────────────────────────────────────────────────────────────────────────────
# H2 — ¿las estrias siguen a los OAR? (con chequeo de confounder mecanico vs anatomico)
# ──────────────────────────────────────────────────────────────────────────────

def test_h2(df: pd.DataFrame) -> dict:
    out = {}
    theta_rectum = df["theta_rectum_deg"].dropna()
    theta_bladder = df["theta_bladder_deg"].dropna()

    out["rayleigh_theta_rectum_poblacional"] = rayleigh_test(theta_rectum.to_numpy())
    out["rayleigh_theta_bladder_poblacional"] = rayleigh_test(theta_bladder.to_numpy())
    out["circstd_theta_rectum_deg"] = circstd_deg(theta_rectum.to_numpy())
    out["circstd_theta_bladder_deg"] = circstd_deg(theta_bladder.to_numpy())

    por_radio = {}
    for m in RADII_MULT:
        suf = f"r{m}"
        theta_min = df[f"theta_min_real_deg_{suf}"].dropna()
        rayleigh_min = rayleigh_test(theta_min.to_numpy())
        circstd_min = circstd_deg(theta_min.to_numpy())

        # Diferencia angular theta_min - theta_OAR, por paciente (solo donde ambos existen)
        idx_r = theta_min.index.intersection(theta_rectum.index)
        idx_b = theta_min.index.intersection(theta_bladder.index)
        diffs_rectum = np.array([diff_angular_deg(theta_min[i], theta_rectum[i]) for i in idx_r])
        diffs_bladder = np.array([diff_angular_deg(theta_min[i], theta_bladder[i]) for i in idx_b])
        rayleigh_diff_rectum = rayleigh_test(diffs_rectum % 360) if len(diffs_rectum) > 1 else None
        rayleigh_diff_bladder = rayleigh_test(diffs_bladder % 360) if len(diffs_bladder) > 1 else None

        # Confounder: desviacion del theta_min respecto a SU media poblacional vs
        # desviacion del theta_rectum respecto a SU media poblacional (mismos pacientes)
        mean_min = circmean_deg(theta_min.to_numpy())
        mean_rectum = circmean_deg(theta_rectum.to_numpy())
        dev_min = np.array([diff_angular_deg(theta_min[i], mean_min) for i in idx_r])
        dev_rectum = np.array([diff_angular_deg(theta_rectum[i], mean_rectum) for i in idx_r])
        if len(dev_min) > 2:
            rho, p_rho = stats.spearmanr(dev_min, dev_rectum)
        else:
            rho, p_rho = float("nan"), float("nan")

        por_radio[suf] = {
            "n": int(len(theta_min)),
            "rayleigh_theta_min_poblacional": rayleigh_min,
            "circstd_theta_min_deg": circstd_min,
            "rayleigh_diff_theta_min_menos_theta_rectum": rayleigh_diff_rectum,
            "rayleigh_diff_theta_min_menos_theta_bladder": rayleigh_diff_bladder,
            "confounder_spearman_dev_min_vs_dev_rectum": {"rho": float(rho), "p": float(p_rho), "n": int(len(dev_min))},
        }
    out["por_radio"] = por_radio

    # Veredicto (r=2.0 headline): firma mecanica si circstd(theta_min) << circstd(theta_rectum)
    # Y la correlacion de desviaciones es no significativa; anatomica si correlacion positiva y significativa.
    h = por_radio["r2.0"]
    rho = h["confounder_spearman_dev_min_vs_dev_rectum"]["rho"]
    p_rho = h["confounder_spearman_dev_min_vs_dev_rectum"]["p"]
    circstd_min = h["circstd_theta_min_deg"]
    circstd_rectum = out["circstd_theta_rectum_deg"]
    if not np.isnan(rho) and p_rho < 0.05 and rho > 0.3:
        veredicto = "ANATOMICA (los minimos siguen la variacion inter-paciente del OAR)"
    elif circstd_min < 0.6 * circstd_rectum:
        veredicto = "MECANICA (minimos mucho mas fijos entre pacientes que la posicion real del OAR)"
    else:
        veredicto = "NO CONCLUYENTE"
    out["veredicto"] = veredicto
    out["circstd_min_vs_rectum_ratio"] = float(circstd_min / circstd_rectum) if circstd_rectum > 0 else float("nan")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# H3 — ¿la amplitud crece en planes forzados?
# ──────────────────────────────────────────────────────────────────────────────

def test_h3(df: pd.DataFrame) -> dict:
    out = {"nota_mus": "MUs NO disponible para este cohorte — el CSV con esa columna "
                       "(metricas_planes_hipofx_D95norm_clean.csv) ya no existe en disco "
                       "(ver docstring del script). Control de modulacion general OMITIDO, "
                       "no se reporta un numero donde no hay dato."}
    features = ["overlap_rel_recto", "overlap_rel_vejiga", "VolRectum_cc", "VolBladder_cc", "VolPTV_cc"]
    por_radio = {}
    for m in RADII_MULT:
        suf = f"r{m}"
        y = df[f"amp_mid_real_{suf}"]
        corr = {}
        for feat in features:
            x = df[feat]
            valid = x.notna() & y.notna()
            if valid.sum() > 3:
                rho, p = stats.spearmanr(x[valid], y[valid])
            else:
                rho, p = float("nan"), float("nan")
            corr[feat] = {"spearman_rho": float(rho), "p": float(p), "n": int(valid.sum())}

        grupos = {}
        for tag in ["RV65", "RV55", "BV65"]:
            flag = df[f"Flag_{tag}_cumple"]
            cumple = y[flag == 1].dropna()
            no_cumple = y[flag == 0].dropna()
            if len(cumple) > 1 and len(no_cumple) > 1:
                u = stats.mannwhitneyu(no_cumple, cumple, alternative="greater")
                p_u = float(u.pvalue)
            else:
                p_u = float("nan")
            grupos[tag] = {
                "n_cumple": int(len(cumple)), "n_no_cumple": int(len(no_cumple)),
                "amp_mid_cumple_mean": float(cumple.mean()) if len(cumple) else float("nan"),
                "amp_mid_no_cumple_mean": float(no_cumple.mean()) if len(no_cumple) else float("nan"),
                "mannwhitney_p_no_cumple_mayor": p_u,
            }
        por_radio[suf] = {"correlaciones_features": corr, "cumple_vs_no_cumple": grupos}
    out["por_radio"] = por_radio

    h = por_radio["r2.0"]
    rhos_sig = [v["spearman_rho"] for v in h["correlaciones_features"].values()
                if not np.isnan(v["p"]) and v["p"] < 0.05 and v["spearman_rho"] > 0]
    grupos_sig = [v for v in h["cumple_vs_no_cumple"].values()
                  if not np.isnan(v["mannwhitney_p_no_cumple_mayor"]) and v["mannwhitney_p_no_cumple_mayor"] < 0.05]
    if rhos_sig or grupos_sig:
        veredicto = "CONFIRMADA"
    else:
        veredicto = "RECHAZADA"
    out["veredicto"] = veredicto
    return out


# ──────────────────────────────────────────────────────────────────────────────
# H4 — ¿el modelo captura la estructura?
# ──────────────────────────────────────────────────────────────────────────────

def test_h4(df: pd.DataFrame) -> dict:
    por_radio = {}
    for m in RADII_MULT:
        suf = f"r{m}"
        real = df[f"amp_mid_real_{suf}"].dropna()
        pred = df.loc[real.index, f"amp_mid_pred_{suf}"]
        res = df.loc[real.index, f"amp_mid_res_{suf}"]

        w_pred = stats.wilcoxon(real, pred, alternative="greater")  # real > pred => modelo suaviza
        theta_min_real = df.loc[real.index, f"theta_min_real_deg_{suf}"]
        theta_min_res = df.loc[real.index, f"theta_min_res_deg_{suf}"]
        diffs = np.array([diff_angular_deg(theta_min_res[i], theta_min_real[i]) for i in real.index])
        rayleigh_coincidencia = rayleigh_test(diffs % 360)

        rho, p_rho = stats.spearmanr(res, real)

        por_radio[suf] = {
            "n": int(len(real)),
            "amp_mid_real_mean": float(real.mean()),
            "amp_mid_pred_mean": float(pred.mean()),
            "amp_mid_res_mean": float(res.mean()),
            "wilcoxon_real_mayor_que_pred_p": float(w_pred.pvalue),
            "ratio_pred_vs_real": float(pred.mean() / real.mean()) if real.mean() > 0 else float("nan"),
            "ratio_res_vs_real": float(res.mean() / real.mean()) if real.mean() > 0 else float("nan"),
            "rayleigh_coincidencia_angular_residuo_vs_real": rayleigh_coincidencia,
            "spearman_amp_mid_res_vs_amp_mid_real": {"rho": float(rho), "p": float(p_rho)},
        }
    out = {"por_radio": por_radio}

    h = por_radio["r2.0"]
    modelo_suaviza = h["wilcoxon_real_mayor_que_pred_p"] < 0.05 and h["ratio_pred_vs_real"] < 0.85
    residuo_estructurado = h["rayleigh_coincidencia_angular_residuo_vs_real"]["p"] < 0.05
    residuo_escala_con_real = (h["spearman_amp_mid_res_vs_amp_mid_real"]["p"] < 0.05
                                and h["spearman_amp_mid_res_vs_amp_mid_real"]["rho"] > 0.3)
    if modelo_suaviza and residuo_estructurado:
        veredicto = "CONFIRMADA (el modelo suaviza y el residuo esta estructurado en los mismos angulos)"
    elif residuo_estructurado and not modelo_suaviza:
        veredicto = "CONFIRMADA (residuo estructurado angularmente, aunque el modelo no suaviza sistematicamente)"
    elif not residuo_estructurado:
        veredicto = "RECHAZADA (el residuo no muestra estructura angular localizada — el modelo ya aprendio el prior)"
    else:
        veredicto = "NO CONCLUYENTE"
    out["veredicto"] = veredicto
    out["modelo_suaviza_r2.0"] = bool(modelo_suaviza)
    out["residuo_estructurado_r2.0"] = bool(residuo_estructurado)
    out["residuo_escala_con_amplitud_real_r2.0"] = bool(residuo_escala_con_real)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_polar_overlay(df: pd.DataFrame, crudos: dict):
    suf = "r2.0"
    sub = df[["AnonID", f"amp_mid_real_{suf}"]].dropna().sort_values(f"amp_mid_real_{suf}")
    if len(sub) < 3:
        return
    elegidos = {
        "baja": sub.iloc[0]["AnonID"],
        "media": sub.iloc[len(sub) // 2]["AnonID"],
        "alta": sub.iloc[-1]["AnonID"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})
    for ax, (etiqueta, anonid) in zip(axes, elegidos.items()):
        ejemplo = crudos[anonid]["ejemplo_slices"]
        if not ejemplo:
            continue
        thetas_deg = next(iter(ejemplo.values()))["theta_deg"]
        real_avg = np.mean([v["real"] for v in ejemplo.values()], axis=0)
        pred_avg = np.mean([v["pred"] for v in ejemplo.values()], axis=0)
        res_avg = np.mean([v["res"] for v in ejemplo.values()], axis=0)
        thetas_rad = np.deg2rad(thetas_deg)
        thetas_closed = np.append(thetas_rad, thetas_rad[0])

        ax.plot(thetas_closed, np.append(real_avg, real_avg[0]), color="black", linewidth=1.6, label="Real")
        ax.plot(thetas_closed, np.append(pred_avg, pred_avg[0]), color="firebrick", linewidth=1.4,
                linestyle="--", label="Predicha")
        ax2 = ax
        theta_r = [v["theta_rectum_deg"] for v in ejemplo.values() if v["theta_rectum_deg"] is not None]
        theta_b = [v["theta_bladder_deg"] for v in ejemplo.values() if v["theta_bladder_deg"] is not None]
        if theta_r:
            ax.axvline(np.deg2rad(circmean_deg(np.array(theta_r))), color="red", linestyle=":", label="θ recto")
        if theta_b:
            ax.axvline(np.deg2rad(circmean_deg(np.array(theta_b))), color="cyan", linestyle=":", label="θ vejiga")
        ax.set_theta_zero_location("N")
        amp = df.loc[df["AnonID"] == anonid, f"amp_mid_real_{suf}"].values[0]
        ax.set_title(f"{etiqueta} amp_estrias — {anonid}\n(amp_mid_real={amp:.3f})", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("D(θ) promedio (todas las slices con PTV, r=2.0×R_eq) — real vs. predicha, θ=0 anterior",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "polar_overlay_representativos.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Segunda figura: residuo solo, mismos 3 pacientes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})
    for ax, (etiqueta, anonid) in zip(axes, elegidos.items()):
        ejemplo = crudos[anonid]["ejemplo_slices"]
        if not ejemplo:
            continue
        thetas_deg = next(iter(ejemplo.values()))["theta_deg"]
        res_avg = np.mean([v["res"] for v in ejemplo.values()], axis=0)
        thetas_rad = np.deg2rad(thetas_deg)
        thetas_closed = np.append(thetas_rad, thetas_rad[0])
        ax.plot(thetas_closed, np.append(res_avg, res_avg[0]), color="darkorange", linewidth=1.6)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_theta_zero_location("N")
        ax.set_title(f"{etiqueta} — Residuo (pred-real)\n{anonid}", fontsize=9)
    fig.suptitle("Residuo (pred − real) promedio, r=2.0×R_eq", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "polar_overlay_residuo.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_fft_spectrum(crudos: dict):
    todas_real, todas_pred, todas_res = [], [], []
    for anonid, crudo in crudos.items():
        for v in crudo["ejemplo_slices"].values():
            todas_real.append(espectro_amplitud(v["real"]))
            todas_pred.append(espectro_amplitud(v["pred"]))
            todas_res.append(espectro_amplitud(v["res"]))
    if not todas_real:
        return
    espectro_real = np.mean(todas_real, axis=0)
    espectro_pred = np.mean(todas_pred, axis=0)
    espectro_res = np.mean(todas_res, axis=0)
    n_arm = min(30, len(espectro_real))
    arm = np.arange(n_arm)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, escala in zip(axes, ["lineal", "log"]):
        ax.plot(arm, espectro_real[:n_arm], marker="o", ms=3, color="black", label="Real")
        ax.plot(arm, espectro_pred[:n_arm], marker="o", ms=3, color="firebrick", label="Predicha")
        ax.plot(arm, espectro_res[:n_arm], marker="o", ms=3, color="darkorange", label="Residuo")
        ax.axvspan(BAND_LOW[0], BAND_LOW[1], color="gray", alpha=0.15, label="banda baja (confounder forma)")
        ax.axvspan(BAND_MID[0], BAND_MID[1], color="steelblue", alpha=0.15, label="banda media (estrias)")
        ax.set_xlabel("Armonico angular n")
        ax.set_ylabel("Amplitud promedio (% prescripcion)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        if escala == "log":
            ax.set_yscale("log")
            ax.set_xlim(1, n_arm - 1)
            ax.set_title("Zoom log (n≥1, sin DC)", fontsize=10)
        else:
            ax.set_title("Escala lineal (incl. DC)", fontsize=10)
    fig.suptitle("Espectro de amplitud angular promedio (r=2.0×R_eq, todas las slices/pacientes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "espectro_fft_promedio.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_scatters(df: pd.DataFrame):
    suf = "r2.0"
    y = df[f"amp_mid_real_{suf}"]
    features = ["overlap_rel_recto", "overlap_rel_vejiga", "VolRectum_cc", "VolBladder_cc", "VolPTV_cc"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, feat in zip(axes.flat, features):
        x = df[feat]
        ax.scatter(x, y, s=22, alpha=0.7, color="steelblue")
        valid = x.notna() & y.notna()
        if valid.sum() > 3:
            rho, p = stats.spearmanr(x[valid], y[valid])
            ax.set_title(f"{feat}\nSpearman rho={rho:.2f} p={p:.3f}", fontsize=9)
        ax.set_xlabel(feat)
        ax.set_ylabel("amp_estrias (real, r=2.0×R_eq)")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.5, "MUs: NO DISPONIBLE\n(CSV de features perdido —\nver docstring del script)",
                        fontsize=10, color="firebrick", transform=axes.flat[-1].transAxes)
    fig.suptitle("H3 — amp_estrias (real) vs. features geometricas (recalculadas desde mascaras)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "scatter_amp_vs_features.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_circular_hist(df: pd.DataFrame):
    suf = "r2.0"
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), subplot_kw={"projection": "polar"})
    datos = [
        ("θ_min (estrias, real)", df[f"theta_min_real_deg_{suf}"].dropna()),
        ("θ_recto", df["theta_rectum_deg"].dropna()),
        ("θ_vejiga", df["theta_bladder_deg"].dropna()),
    ]
    bins = np.linspace(0, 2 * np.pi, 25)
    for ax, (titulo, serie) in zip(axes, datos):
        rad = np.deg2rad(serie.to_numpy())
        counts, edges = np.histogram(rad, bins=bins)
        centros = (edges[:-1] + edges[1:]) / 2
        ancho = edges[1] - edges[0]
        ax.bar(centros, counts, width=ancho, color="steelblue", alpha=0.8, edgecolor="white")
        media = circmean_deg(serie.to_numpy())
        ax.axvline(np.deg2rad(media), color="firebrick", linewidth=2, label=f"media circ.={media:.0f}°")
        ax.set_theta_zero_location("N")
        ax.set_title(f"{titulo} (n={len(serie)})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Distribucion angular poblacional — θ=0 anterior, r=2.0×R_eq", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "histograma_circular_minimos.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _default_json(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    pacientes = [(a, "val") for a in splits["val"]] + [(a, "test") for a in splits["test"]]
    print(f"Pacientes a analizar: {len(pacientes)} (val={len(splits['val'])}, test={len(splits['test'])})")

    cfg = OmegaConf.load(str(CONFIG_YAML))
    cfg.data.processed_dir = str(PROCESSED_DIR)
    torch.set_float32_matmul_precision("high")
    model, device = cargar_modelo(str(CHECKPOINT), cfg)

    conv1 = model.model.inc.block[0]
    print(f"Conv1 in_channels real del checkpoint: {conv1.in_channels} (cfg.model.in_channels={cfg.model.in_channels})")
    if conv1.in_channels != cfg.model.in_channels:
        raise RuntimeError(f"Mismatch de canales: checkpoint espera {conv1.in_channels}, cfg dice {cfg.model.in_channels}")

    gt_dvh = pd.read_csv(GT_DVH_CSV).set_index("AnonID")

    filas = []
    crudos = {}
    for anonid, split_name in tqdm(pacientes, desc="Inferencia + analisis angular"):
        npz_path = PROCESSED_DIR / f"{anonid}.npz"
        arrays, meta = cargar_npz(npz_path)
        dose_pred_pct = inferir_dosis(model, device, arrays)
        fila, crudo = analizar_paciente(anonid, arrays, meta, dose_pred_pct)
        fila["split"] = split_name
        for tag, col in [("RV65", "Flag_Rectum_V65Gy_lt15_gt256"),
                          ("RV55", "Flag_Rectum_V55Gy_lt25_gt256"),
                          ("BV65", "Flag_Bladder_V65Gy_lt15_gt256")]:
            fila[f"Flag_{tag}_cumple"] = int(gt_dvh.loc[anonid, col]) if anonid in gt_dvh.index else None
        filas.append(fila)
        crudos[anonid] = crudo

    df = pd.DataFrame(filas)
    df.to_csv(OUT_DIR / "metricas_angulares_por_paciente.csv", index=False)
    print(f"\nGuardado: {OUT_DIR / 'metricas_angulares_por_paciente.csv'} ({len(df)} pacientes)")

    print("\nCorriendo tests de hipotesis...")
    resultados_h1 = test_h1(df)
    resultados_h2 = test_h2(df)
    resultados_h3 = test_h3(df)
    resultados_h4 = test_h4(df)

    print("\nGenerando plots...")
    plot_polar_overlay(df, crudos)
    plot_fft_spectrum(crudos)
    plot_scatters(df)
    plot_circular_hist(df)

    summary = {
        "n_pacientes": len(df),
        "n_val": int((df["split"] == "val").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "checkpoint": str(CHECKPOINT),
        "config": str(CONFIG_YAML),
        "radios_multiplos_Req": RADII_MULT,
        "n_theta": N_THETA,
        "banda_baja_confounder": BAND_LOW,
        "banda_media_estrias": BAND_MID,
        "banda_alta_ruido_desde": BAND_HIGH_START,
        "min_cobertura_body": MIN_BODY_COVERAGE,
        "limitaciones_de_datos": {
            "MUs": "no disponible — CSV metricas_planes_hipofx_D95norm_clean.csv ya no existe en disco",
            "NArcos": "no disponible — mismo CSV faltante; ESTRATIFICACION TRANSVERSAL (2 vs 3 arcos) NO se pudo ejecutar",
            "features_geometricas_recalculadas": "VolPTV/Rectum/Bladder_cc y overlap_rel_recto/vejiga se "
                "recalcularon directo desde las mascaras del NPZ (mismo patron de calibracion de voxel que "
                "scripts/compute_overlap_real.py), NO desde el CSV — evita ademas el bug Math.Min conocido.",
        },
        "H1_estructura_angular_en_dosis_real": resultados_h1,
        "H2_estrias_siguen_a_OAR": resultados_h2,
        "H3_amplitud_crece_en_planes_forzados": resultados_h3,
        "H4_residuo_del_modelo_estructurado": resultados_h4,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_default_json)
    print(f"Guardado: {OUT_DIR / 'summary.json'}")

    print("\n=== VEREDICTOS ===")
    print(f"H1 (estructura angular existe):      {resultados_h1['veredicto']}")
    print(f"H2 (estrias siguen al OAR):           {resultados_h2['veredicto']}")
    print(f"H3 (amplitud crece en planes forzados): {resultados_h3['veredicto']}")
    print(f"H4 (residuo del modelo estructurado): {resultados_h4['veredicto']}")

    return df, crudos, summary


if __name__ == "__main__":
    main()
