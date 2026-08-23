"""
Extraccion de 3 metricas de complejidad VMAT desde el RTPLAN (RP) DICOM real, para
usarlas (Parte B, scripts/arbitro_complejidad.py) como ARBITRO de si el piso del
hombro medio-bajo de OAR (ver results/diagnostico_piso/summary.json) es de
GENERACION (agresividad del plan) o ANATOMICO.

⚠️ ALCANCE: SOLO normofx (decision de Pablo, 2026-08-19). Los RP de hipofx no estan
disponibles localmente (solo 7-8 casos de piloto en dicom_pilot/) y una sonda de
vecinos/neighbors necesita cientos de casos para tener sentido estadistico -- esa
parte queda fuera de esta tarea. El archivo real de normofx (386 pacientes, RP/RS/RD/CT
reales de Eclipse) esta en el share de red:
    \\\\10.100.0.252\\centro_de_datos2018\\101_Cosas de\\PABLO\\CNN Prostata\\dicoms normofx
cada subcarpeta es "PT_<hash>" == mismo AnonID que se usa en splits_v1.json / processed/
(384 de esos 386 estan en splits_v1.json; se procesan los 386 igual, la interseccion con
D3 se hace despues en arbitro_complejidad.py).

MLC = Millennium 120 SIEMPRE. Los leaf-boundaries NO se hardcodean: se LEEN del propio
RP (BeamLimitingDeviceSequence[MLCX].LeafPositionBoundaries) y se VALIDAN contra el
patron esperado (40 pares centrales de 5mm + 20 externos de 10mm, -200..+200mm) como
chequeo de integridad -- confirmado en un caso real (dicom_pilot/PT_003fe2bb84986507).

Las 3 metricas, con su definicion exacta tomada de los papers en papers/Metricas/
(Pablo los proveyo el 2026-08-19 porque "VMAT1.md" no existia localmente hasta ese
momento):

- MU Factor (MU/cGy) -- Nguyen & Chan 2020 ("Metricas VMAT2.md" Sec 2.B.1, citando a
  Crowe et al. 2014): "ratio of the total monitor units to the prescribed dose in cGy".
  Se normaliza POR FRACCION (MU_por_fraccion_del_plan / dosis_por_fraccion_cGy), que es
  algebraicamente equivalente a MU_de_todo_el_curso/dosis_total_del_curso (ambas
  cancelan NumeroDeFracciones) pero es exactamente la cantidad que Masi et al. 2013
  llaman "MU normalizado a 2Gy" (mismo objetivo: comparar planes con distinta
  fraccionacion). Verificado en un caso real: da ~3.4 MU/cGy, dentro del rango
  reportado por Nguyen&Chan (mediana 2.76-3.63 MU/cGy) -- si se usara la dosis TOTAL
  del curso (7800 cGy) en vez de por-fraccion, el numero sale ~39x mas chico y fuera de
  cualquier rango publicado.

- MCSv -- Masi et al. 2013 (Med Phys 40:071718, Appendix "MODULATION COMPLEXITY SCORE
  FOR VMAT PLANS", Eq. A1-A4). Adaptacion de McNiven et al. 2010 (MCS, IMRT
  step-and-shoot) a VMAT, sustituyendo "segmento" por "control point":
    - LSV_cp (leaf sequence variability, Eq. A2): por banco de leaves (izq/der),
      penaliza diferencias de posicion entre leaves ADYACENTES relativas al rango
      posmax(CP) = max(pos_n) - min(pos_n) SOLO sobre las N leaves activas (dentro
      del jaw Y) en ESE control point -- Eq. A1. LSV_cp = LSV_izq * LSV_der.
    - AAV_cp (aperture area variability, Eq. A3): suma de gaps (leaf B - leaf A) de
      los pares activos en el CP, normalizada a la suma de los gaps MAXIMOS que cada
      par de leaves alcanza en TODO el arco (no solo en los CP donde ese par esta
      activo).
    - MCSv (Eq. A4): "unlike the original step-and-shoot case, during a VMAT arc MUs
      are delivered continuously between adjacent control points" -> se promedia
      LSV_cp y AAV_cp con el CP siguiente, se pesa por el MU entregado entre esos dos
      CP consecutivos, y se suma sobre TODOS los CP del arco. Esta tarea extiende esa
      suma sobre TODOS los arcos del plan (MU-weighted), no solo un arco.

- SAS10 (small aperture score, umbral 10mm) -- Crowe et al. 2014 / Younge et al. 2016,
  formula reproducida en la revision de Chiavassa et al. 2019 (PMC6774599) y usada por
  Nguyen & Chan 2020 (misma referencia 4 que MU Factor):
    SAS(x) = sum_cp [ N(0<gap<x)_cp / N(gap>0)_cp * MU_cp / MU_plan ]
  fraccion MU-weighted del plan entregada con gaps de par de leaves en (0, 10)mm,
  sobre los pares de leaves NO bloqueados por el jaw Y (gap>0). A diferencia de MCSv,
  la literatura no describe un promediado con el CP adyacente para SAS -- se usa el
  gap del propio CP.

Uso:
    .venv/Scripts/python.exe scripts/extraer_complejidad_rp.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent

RTPLAN_ROOT = Path(r"\\10.100.0.252\centro_de_datos2018\101_Cosas de\PABLO\CNN Prostata\dicoms normofx")
SPLITS_NORMO = _REPO_ROOT / "data/splits/splits_v1.json"

OUT_DIR = _REPO_ROOT / "results/complejidad_arbitro"
DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RX_CGY_NORMO = 7800.0  # 78Gy/39fx, convencion fija del proyecto (d4_subset_pareado.py)
GAP_ABIERTO_MM = 0.5  # por debajo de esto se considera "cerrado" (ruido de redondeo DICOM)
SAS_UMBRAL_MM = 10.0

# Patron esperado Millennium 120 (60 pares, -200..+200mm): 10 pares@10mm + 40 pares@5mm + 10 pares@10mm
_ANCHOS_ESPERADOS = np.array([10.0] * 10 + [5.0] * 40 + [10.0] * 10)


# ──────────────────────────────────────────────────────────────────────────────
# Lectura y validacion del MLC
# ──────────────────────────────────────────────────────────────────────────────

def validar_millennium120(boundaries: np.ndarray) -> None:
    if len(boundaries) != 61:
        raise ValueError(f"Se esperaban 61 leaf-boundaries (60 pares), vinieron {len(boundaries)}")
    anchos = np.diff(boundaries)
    if not np.allclose(anchos, _ANCHOS_ESPERADOS, atol=0.01):
        raise ValueError(f"Patron de leaves no es Millennium 120 (40x5mm+20x10mm): anchos={anchos.tolist()}")
    if not (np.isclose(boundaries[0], -200.0) and np.isclose(boundaries[-1], 200.0)):
        raise ValueError(f"Leaf boundaries no van de -200 a +200: [{boundaries[0]}, {boundaries[-1]}]")


def leaf_boundaries_y_anchos(beam) -> tuple:
    mlc_dev = None
    for d in beam.BeamLimitingDeviceSequence:
        if d.RTBeamLimitingDeviceType == "MLCX":
            mlc_dev = d
            break
    if mlc_dev is None or "LeafPositionBoundaries" not in mlc_dev:
        raise ValueError("Beam sin BeamLimitingDeviceSequence[MLCX].LeafPositionBoundaries")
    boundaries = np.array([float(x) for x in mlc_dev.LeafPositionBoundaries])
    validar_millennium120(boundaries)
    anchos = np.diff(boundaries)
    centros = (boundaries[:-1] + boundaries[1:]) / 2.0
    return boundaries, anchos, centros


# ──────────────────────────────────────────────────────────────────────────────
# Beams / control points (con carry-forward de jaws, que Eclipse solo exporta en CP0)
# ──────────────────────────────────────────────────────────────────────────────

def arcos_de_tratamiento(ds) -> list:
    return [b for b in ds.BeamSequence
            if b.get("TreatmentDeliveryType") == "TREATMENT"
            and b.get("BeamType") == "DYNAMIC"
            and b.get("RadiationType") == "PHOTON"
            and len(b.ControlPointSequence) > 2]


def beam_meterset(ds, beam) -> float:
    fg = ds.FractionGroupSequence[0]
    for rb in fg.ReferencedBeamSequence:
        if int(rb.ReferencedBeamNumber) == int(beam.BeamNumber):
            return float(rb.BeamMeterset)
    raise ValueError(f"Beam {beam.BeamNumber} sin ReferencedBeamSequence/BeamMeterset")


def extraer_control_points(beam) -> list:
    """Devuelve lista de dicts por CP: gantry, cmw, leafA(60), leafB(60), jaw_y=(y1,y2).
    Aplica carry-forward: Eclipse solo redefine un BeamLimitingDevicePositionSequence
    cuando ese dispositivo cambio de posicion respecto del CP anterior (los jaws Y/X
    tipicamente solo aparecen en CP0 porque no se mueven durante el arco)."""
    cps_out = []
    leaf_a = leaf_b = jaw_y = None
    for cp in beam.ControlPointSequence:
        devs = cp.get("BeamLimitingDevicePositionSequence", None)
        if devs:
            for d in devs:
                t = d.RTBeamLimitingDeviceType
                if t == "MLCX":
                    pos = [float(x) for x in d.LeafJawPositions]
                    leaf_a = np.array(pos[:60])
                    leaf_b = np.array(pos[60:])
                elif t in ("ASYMY", "Y"):
                    pos = [float(x) for x in d.LeafJawPositions]
                    jaw_y = (pos[0], pos[1]) if len(pos) == 2 else (-abs(pos[0]), abs(pos[0]))
        if leaf_a is None or leaf_b is None:
            continue  # CP incompleto (no deberia pasar en CP0, pero por robustez)
        cps_out.append({
            "gantry": float(cp.GantryAngle) if "GantryAngle" in cp else None,
            "cmw": float(cp.CumulativeMetersetWeight),
            "leafA": leaf_a.copy(),
            "leafB": leaf_b.copy(),
            "jaw_y": jaw_y if jaw_y is not None else (-200.0, 200.0),
        })
    return cps_out


def pares_activos(jaw_y: tuple, centros: np.ndarray) -> np.ndarray:
    y1, y2 = min(jaw_y), max(jaw_y)
    return (centros >= y1) & (centros <= y2)


# ──────────────────────────────────────────────────────────────────────────────
# MCSv — LSV (Eq. A1-A2) y AAV (Eq. A3), Masi et al. 2013 Appendix
# ──────────────────────────────────────────────────────────────────────────────

def _lsv_banco(pos: np.ndarray, activos: np.ndarray) -> float:
    idx_activos = np.where(activos)[0]
    if len(idx_activos) < 2:
        return 1.0  # sin modulacion posible con <2 leaves activas -> factor neutro
    pos_act = pos[idx_activos]
    posmax = float(pos_act.max() - pos_act.min())
    if posmax <= 0:
        return 1.0
    n = len(idx_activos)
    diffs = np.abs(np.diff(pos_act))
    return float(np.sum(posmax - diffs) / ((n - 1) * posmax))


def calc_lsv_cp(leafA: np.ndarray, leafB: np.ndarray, activos: np.ndarray) -> float:
    return _lsv_banco(leafA, activos) * _lsv_banco(leafB, activos)


def calc_gap_cp(leafA: np.ndarray, leafB: np.ndarray) -> np.ndarray:
    return np.clip(leafB - leafA, 0.0, None)


def calc_aav_cp(gap: np.ndarray, activos: np.ndarray, max_gap_pares: np.ndarray) -> float:
    denom = float(np.sum(max_gap_pares[activos]))
    if denom <= 0:
        return 1.0
    return float(np.sum(gap[activos]) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# Extraccion por paciente
# ──────────────────────────────────────────────────────────────────────────────

def extraer_paciente(rp_path: Path) -> dict:
    ds = pydicom.dcmread(str(rp_path), stop_before_pixels=True)
    arcos = arcos_de_tratamiento(ds)
    if len(arcos) == 0:
        raise ValueError("Sin arcos de tratamiento (DYNAMIC+TREATMENT+PHOTON)")

    n_fx = int(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
    dosis_por_fx_cgy = RX_CGY_NORMO / n_fx

    mu_total = 0.0
    seg_mcsv = []  # (peso_MU, LSV_seg, AAV_seg)
    seg_sas = []   # (peso_MU, SAS10_cp)

    for beam in arcos:
        mu_beam = beam_meterset(ds, beam)
        mu_total += mu_beam
        _, _, centros = leaf_boundaries_y_anchos(beam)
        cps = extraer_control_points(beam)
        if len(cps) < 2:
            continue

        activos_por_cp = [pares_activos(cp["jaw_y"], centros) for cp in cps]
        gaps_por_cp = [calc_gap_cp(cp["leafA"], cp["leafB"]) for cp in cps]
        lsv_por_cp = [calc_lsv_cp(cp["leafA"], cp["leafB"], act)
                      for cp, act in zip(cps, activos_por_cp)]

        # maximo por-par de gap en TODO el arco (Eq. A3, denominador de AAV)
        gaps_stack = np.stack(gaps_por_cp)  # (n_cp, 60)
        max_gap_pares = gaps_stack.max(axis=0)
        aav_por_cp = [calc_aav_cp(g, act, max_gap_pares) for g, act in zip(gaps_por_cp, activos_por_cp)]

        for i in range(len(cps) - 1):
            w = (cps[i + 1]["cmw"] - cps[i]["cmw"]) * mu_beam
            if w <= 0:
                continue
            lsv_seg = (lsv_por_cp[i] + lsv_por_cp[i + 1]) / 2.0
            aav_seg = (aav_por_cp[i] + aav_por_cp[i + 1]) / 2.0
            seg_mcsv.append((w, lsv_seg, aav_seg))

            gap_i, act_i = gaps_por_cp[i], activos_por_cp[i]
            n_abiertos = int(np.sum(act_i & (gap_i > GAP_ABIERTO_MM)))
            n_chicos = int(np.sum(act_i & (gap_i > GAP_ABIERTO_MM) & (gap_i < SAS_UMBRAL_MM)))
            sas_cp = (n_chicos / n_abiertos) if n_abiertos > 0 else 0.0
            seg_sas.append((w, sas_cp))

    pesos_mcsv = np.array([s[0] for s in seg_mcsv])
    prod_mcsv = np.array([s[1] * s[2] for s in seg_mcsv])
    mcsv = float(np.sum(pesos_mcsv * prod_mcsv) / np.sum(pesos_mcsv)) if pesos_mcsv.sum() > 0 else float("nan")

    pesos_sas = np.array([s[0] for s in seg_sas])
    vals_sas = np.array([s[1] for s in seg_sas])
    sas10 = float(np.sum(pesos_sas * vals_sas) / np.sum(pesos_sas)) if pesos_sas.sum() > 0 else float("nan")

    mu_factor = mu_total / dosis_por_fx_cgy

    return {
        "MU_factor": mu_factor,
        "MCSv": mcsv,
        "SAS10": sas10,
        "NArcos": len(arcos),
        "MU_total": mu_total,
        "dosis_por_fraccion_cgy": dosis_por_fx_cgy,
        "n_fracciones_dicom": n_fx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    with open(SPLITS_NORMO) as f:
        s = json.load(f)
    ids_d3 = set(s["train"]) | set(s["val"]) | set(s["test"])

    carpetas = sorted([p for p in RTPLAN_ROOT.iterdir() if p.is_dir()])
    print(f"Carpetas encontradas en el share: {len(carpetas)}")
    print(f"De esas, en splits_v1.json (cohorte D3 normo): "
          f"{len(set(p.name for p in carpetas) & ids_d3)}")

    filas = []
    fallidos = []
    for carpeta in tqdm(carpetas, desc="Extrayendo complejidad RP"):
        anonid = carpeta.name
        rps = sorted(carpeta.glob("RP*.dcm"))
        if len(rps) == 0:
            fallidos.append({"AnonID": anonid, "motivo": "sin archivo RP*.dcm"})
            continue
        if len(rps) > 1:
            fallidos_extra_note = f" (hay {len(rps)}, se usa el primero)"
        else:
            fallidos_extra_note = ""
        try:
            metricas = extraer_paciente(rps[0])
        except Exception as e:
            fallidos.append({"AnonID": anonid, "motivo": f"{type(e).__name__}: {e}"})
            continue

        # chequeos de rango (Sec II.B.2 de Masi: MCSv in [0,1]; SAS10 en [0,1] por definicion)
        if not (0.0 <= metricas["MCSv"] <= 1.0 + 1e-6):
            fallidos.append({"AnonID": anonid, "motivo": f"MCSv fuera de [0,1]: {metricas['MCSv']:.4f}"})
            continue
        if not (0.0 <= metricas["SAS10"] <= 1.0 + 1e-6):
            fallidos.append({"AnonID": anonid, "motivo": f"SAS10 fuera de [0,1]: {metricas['SAS10']:.4f}"})
            continue

        fila = {"AnonID": anonid, "dataset": "normo", "en_splits_v1": anonid in ids_d3}
        fila.update(metricas)
        filas.append(fila)

    df = pd.DataFrame(filas)
    df.to_csv(DATA_DIR / "complejidad_rp.csv", index=False)

    resumen = {
        "fuente_rp": str(RTPLAN_ROOT),
        "n_carpetas_totales": len(carpetas),
        "n_parseados_ok": len(filas),
        "n_fallidos": len(fallidos),
        "n_en_splits_v1_ok": int(df["en_splits_v1"].sum()) if len(df) else 0,
        "fallidos_detalle": fallidos,
        "descriptivos": {
            "MU_factor": {"mean": float(df["MU_factor"].mean()), "std": float(df["MU_factor"].std()),
                          "min": float(df["MU_factor"].min()), "max": float(df["MU_factor"].max())},
            "MCSv": {"mean": float(df["MCSv"].mean()), "std": float(df["MCSv"].std()),
                     "min": float(df["MCSv"].min()), "max": float(df["MCSv"].max())},
            "SAS10": {"mean": float(df["SAS10"].mean()), "std": float(df["SAS10"].std()),
                      "min": float(df["SAS10"].min()), "max": float(df["SAS10"].max())},
            "NArcos": df["NArcos"].value_counts().to_dict(),
        } if len(df) else {},
    }
    with open(DATA_DIR / "extraccion_rp_summary.json", "w") as f:
        json.dump(resumen, f, indent=2, default=str)

    print(f"\n=== EXTRACCION RP — RESUMEN ===")
    print(f"OK: {len(filas)}/{len(carpetas)}  Fallidos: {len(fallidos)}")
    if len(df):
        print(df[["MU_factor", "MCSv", "SAS10", "NArcos"]].describe())
    if fallidos:
        print("\nFallidos (primeros 10):")
        for x in fallidos[:10]:
            print(" ", x)
    print(f"\nGuardado: {DATA_DIR / 'complejidad_rp.csv'}")
    return resumen


if __name__ == "__main__":
    main()
