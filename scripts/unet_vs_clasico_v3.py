"""PROMPT_comparacion_unet_vs_clasico.md — Pasos 1-3: comparacion homologada
U-Net vs ML clasico (Proyecto 1) sobre el MISMO split (splits_hipo_v3.json, 33 test).

Requiere el checkpoint de exp_hipo_003_finetune_v3_alignedsplit (entrenado sobre
splits_hipo_v3.json, no splits_hipo_v3_unet.json -- ver Paso 0 / config para el porque).

GT de referencia: data/dataset_p1.csv (fail_RV65/RV55/BV65, value_RV65/RV55/BV65,
split, Status) -- el MISMO que uso el clasico para entrenar/calibrar/evaluar. NO se
usa gt_dvh_hipo_256_v3.csv aca: se detectaron diferencias de hasta ~30pp en algunos
pacientes entre ambas fuentes (recomputo Python 256-grid vs CSV D95norm original) y
usar la fuente del clasico es la unica forma de que las 2 matrices de confusion
salgan del MISMO ground truth.

Score de la U-Net (Paso 1): V65Gy(recto)%, V55Gy(recto)%, V65Gy(vejiga)% calculados
directo por conteo de voxels (no interpolando la curva de 200 puntos de
compute_pred_dvh.py, mas preciso) sobre la dosis predicha renormalizada a
D95(PTV_pred)=100% -- misma logica de renormalizacion que compute_pred_dvh.py.

Protocolo de calibracion (Paso 2) -- replica eval_p1_cv.py/calibrate_p1_cv.py:
  - Pool de calibracion = poblacion natural (Status != UnApproved) de train+val.
  - Recto = combinacion pesimista max(V65_recto_pred, V55_recto_pred) [en escala
    RAW de V-value predicho, no margen] para AUC/umbral; margen = V_tag_pred -
    umbral_clinico_tag maximizado entre tags, SOLO para la zona naranja/rojo
    (misma logica que la variante B del clasico, que siempre usa el margen ahi).
  - Umbral: mediana de 5-fold StratifiedKFold (seed=42) sobre el score YA FIJO de
    la U-Net (NO se re-entrenan 5 U-Nets por fold -- inviable en tiempo/computo;
    el clasico SI refitea LogReg/Ridge por fold). Esto es una asimetria real
    frente al clasico: los folds del U-Net comparten el mismo score "en muestra"
    de un unico modelo ya entrenado sobre esos mismos pacientes de train -- el CV
    aca solo aporta la estructura de particion+agregacion, no una validacion
    verdaderamente out-of-fold. Reportado explicitamente en el JSON de salida
    (campo "caveat_cv_unet") para que Claude.ai lo pese en el argumento.
  - Sensibilidad objetivo: recto=0.95, vejiga=0.90 (los valores REALES de
    thresholds_cv.json / calibrate_p1_cv.py -- el prompt decia recto=0.85, que es
    el valor viejo de calibrate_p1.py/v3 no-CV; se usan los vigentes para que la
    comparacion sea contra el clasico REAL, no una version superada).
  - Bootstrap 1000, seed=42, para IC de AUC y de sens/esp en test.

Uso:
    python scripts/unet_vs_clasico_v3.py \
        --checkpoint checkpoints/exp_hipo_003_finetune_v3_alignedsplit/epoch=XXX.ckpt \
        --config configs/exp_hipo_003_finetune_v3_alignedsplit.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.datamodules.dose_datamodule import DoseDataModule, DosePatientDataset  # noqa: E402
from evaluate import cargar_modelo  # noqa: E402
from evaluate_hipo import punto_operacion_sens_objetivo, confusion_binaria  # noqa: E402

RX_GY = 70.0
UMBRAL_CLINICO = {"RV65": 15.0, "RV55": 25.0, "BV65": 15.0}
OAR_TAGS = {"recto": ["RV65", "RV55"], "vejiga": ["BV65"]}
SENS_OBJETIVO = {"recto": 0.95, "vejiga": 0.90}
N_FOLDS = 5
N_BOOTSTRAP = 1000
SEED = 42
TAG_TO_VCOL = {"RV65": "V65_recto_pred", "RV55": "V55_recto_pred", "BV65": "V65_vejiga_pred"}


# ─── Paso 1 — inferencia + DVH renormalizado ──────────────────────────────────

def renorm_y_v_values(dose_pred_raw, ptv_mask, rectum_mask, bladder_mask):
    dose_pred_raw = np.clip(np.asarray(dose_pred_raw, dtype=np.float32), 0.0, None)
    d95_pred_raw = float(np.percentile(dose_pred_raw[ptv_mask > 0], 5))
    factor = 100.0 / d95_pred_raw
    dose_norm = dose_pred_raw * factor

    def v_at(mask, umbral_gy):
        vals = dose_norm[mask > 0]
        if len(vals) == 0:
            return float("nan")
        umbral_pct = umbral_gy / RX_GY * 100.0
        return 100.0 * float(np.mean(vals >= umbral_pct))

    return {
        "V65_recto_pred":   v_at(rectum_mask, 65.0),
        "V55_recto_pred":   v_at(rectum_mask, 55.0),
        "V65_vejiga_pred":  v_at(bladder_mask, 65.0),
        "factor_renorm": factor,
        "d95_pred_pre_renorm_pct": d95_pred_raw,
    }


def inferir_todos(model, device, npz_paths, fixed_n_slices, desc):
    ds = DosePatientDataset(npz_paths, augment=False, fixed_n_slices=fixed_n_slices, cache_in_ram=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    filas = []
    for batch in tqdm(loader, desc=desc):
        anonid = batch["anonid"][0] if isinstance(batch["anonid"], list) else batch["anonid"]
        if isinstance(anonid, (list, tuple)):
            anonid = anonid[0]
        batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            x = model._build_input(batch_gpu)
            pred = model(x)
        v = renorm_y_v_values(
            pred[0].cpu().numpy(),
            batch["ptv_mask"][0].numpy(),
            batch["rectum_mask"][0].numpy(),
            batch["bladder_mask"][0].numpy(),
        )
        v["AnonID"] = anonid
        filas.append(v)
    return pd.DataFrame(filas).set_index("AnonID")


# ─── Paso 2 — calibracion CV + evaluacion homologada ──────────────────────────

def score_oar(df, tags, col_suffix=""):
    """Score RAW (max V-value predicho entre tags) y margen (max V_pred - umbral_clinico)."""
    vcols = [TAG_TO_VCOL[t] + col_suffix for t in tags]
    score_raw = df[vcols].to_numpy().max(axis=1)
    margen = np.max([df[TAG_TO_VCOL[t] + col_suffix].to_numpy() - UMBRAL_CLINICO[t] for t in tags], axis=0)
    return score_raw, margen


def label_oar(df, tags):
    return np.max([df[f"fail_{t}"].to_numpy() for t in tags], axis=0)


def bootstrap_auc_ci(y_true, y_score, rng):
    vals = []
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            vals.append(float("nan"))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_sens_esp_ci(y_true, y_pred, rng):
    sens_list, esp_list = [], []
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        c = confusion_binaria(y_true[idx], y_pred[idx])
        sens_list.append(c["sensibilidad"])
        esp_list.append(c["especificidad"])

    def ci(vals):
        lo, hi = np.nanpercentile(np.array(vals), [2.5, 97.5])
        return [float(lo), float(hi)]
    return {"sensibilidad_ci95": ci(sens_list), "especificidad_ci95": ci(esp_list)}


def bootstrap_umbral_ci(label, score, sens_objetivo, rng):
    n = len(label)
    vals = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b, s_b = label[idx], score[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            continue
        vals.append(punto_operacion_sens_objetivo(y_b, s_b, sens_objetivo))
    if not vals:
        return [float("nan"), float("nan")], 0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)], len(vals)


def cv_umbral_unet(label_pool, score_pool, sens_objetivo):
    """Mismos folds/estructura que calibrate_p1_cv.py, pero SIN refit -- el score
    de la U-Net ya esta fijo (ver caveat en el docstring del modulo)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    umbrales, folds_omitidos = [], 0
    for _, val_idx in skf.split(score_pool.reshape(-1, 1), label_pool):
        y_val, s_val = label_pool[val_idx], score_pool[val_idx]
        if y_val.sum() == 0 or y_val.sum() == len(y_val):
            folds_omitidos += 1
            continue
        umbrales.append(punto_operacion_sens_objetivo(y_val, s_val, sens_objetivo))
    arr = np.array(umbrales)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "umbrales_por_fold": arr.tolist(), "mediana": float(np.median(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)), "iqr": [float(q1), float(q3)],
        "folds_omitidos": folds_omitidos,
    }


def calibrar_delta(margen_pool, no_verde_mask):
    margen_no_verde = margen_pool[no_verde_mask]
    if len(margen_no_verde) == 0:
        return 0.0, {"n": 0}
    delta = float(np.median(margen_no_verde))
    stats = {"n": int(len(margen_no_verde)), "min": float(np.min(margen_no_verde)),
              "median": delta, "mean": float(np.mean(margen_no_verde)),
              "std": float(np.std(margen_no_verde)), "max": float(np.max(margen_no_verde))}
    return delta, stats


def evaluar_oar(df_pool, df_test, oar_label, tags, rng):
    label_pool = label_oar(df_pool, tags)
    score_pool, margen_pool = score_oar(df_pool, tags)
    label_test = label_oar(df_test, tags)
    score_test, margen_test = score_oar(df_test, tags)

    sens_obj = SENS_OBJETIVO[oar_label]
    cv = cv_umbral_unet(label_pool, score_pool, sens_obj)
    umbral = cv["mediana"]
    ci_umbral, n_efectivo = bootstrap_umbral_ci(label_pool, score_pool, sens_obj, rng)

    no_verde_pool = score_pool >= umbral
    delta, delta_stats = calibrar_delta(margen_pool, no_verde_pool)

    auc = float(roc_auc_score(label_test, score_test))
    auc_ci = bootstrap_auc_ci(label_test, score_test, rng)
    y_pred_no_verde = (score_test >= umbral).astype(int)
    conf = confusion_binaria(label_test, y_pred_no_verde)
    ci_sens_esp = bootstrap_sens_esp_ci(label_test, y_pred_no_verde, rng)

    zona = np.where(score_test < umbral, "verde", np.where(margen_test >= delta, "rojo", "naranja"))
    zona_counts = {z: int((zona == z).sum()) for z in ("verde", "naranja", "rojo")}
    zona_vs_real = pd.crosstab(
        pd.Series(zona, name="zona_predicha"),
        pd.Series(np.where(label_test == 1, "falla", "cumple"), name="real"),
    ).to_dict()

    return {
        "sens_objetivo": sens_obj,
        "cv_umbral": cv,
        "umbral_mediana_cv": umbral,
        "umbral_bootstrap_ci95": ci_umbral,
        "umbral_bootstrap_n_efectivo": n_efectivo,
        "delta_severidad_pp": delta,
        "distribucion_margen_no_verde_pool": delta_stats,
        "n_no_verde_pool": int(no_verde_pool.sum()),
        "auc": auc, "auc_ci95": auc_ci,
        "sens_esp_test": {**conf, **ci_sens_esp},
        "zona_counts": zona_counts,
        "zona_vs_real": zona_vs_real,
    }


def evaluar_constraint_individual(df_test, tag, rng):
    score = df_test[TAG_TO_VCOL[tag]].to_numpy()
    y_test = df_test[f"fail_{tag}"].to_numpy()
    auc = float(roc_auc_score(y_test, score))
    ci = bootstrap_auc_ci(y_test, score, rng)
    return {"auc": auc, "auc_ci95": ci, "n_test": len(y_test), "n_positivos_test": int(y_test.sum())}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-p1-csv", default="data/dataset_p1.csv")
    p.add_argument("--proyecto1-summary", default="results/proyecto1_v3_cv/metrics_summary.json")
    p.add_argument("--output-dir", default="results/comparacion_clasico_vs_unet_v3")
    p.add_argument("--unet-eval-dir", default="results/unet_v3_eval")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_dir = Path(args.unet_eval_dir)
    unet_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.data.cache_train = False
    cfg.data.cache_val = False
    model, device = cargar_modelo(args.checkpoint, cfg)

    dm = DoseDataModule(cfg)
    dm.setup()
    n_target = dm.train_ds.fixed_n_slices

    print("\n=== Paso 1: inferencia sobre TODO el split (train+val+test) ===")
    df_train = inferir_todos(model, device, dm.train_ds.npz_paths, n_target, "Train (para pool calibracion)")
    df_val   = inferir_todos(model, device, dm.val_ds.npz_paths,   n_target, "Val (para pool calibracion)")
    df_test  = inferir_todos(model, device, dm.test_ds.npz_paths,  n_target, "Test (evaluacion final)")

    df_p1 = pd.read_csv(args.dataset_p1_csv).set_index("AnonID")

    df_all_unet = pd.concat([df_train, df_val, df_test])
    df_all = df_p1.join(df_all_unet, how="inner")
    faltantes = set(df_p1.index) - set(df_all_unet.index)
    if faltantes:
        print(f"[WARN] {len(faltantes)} AnonID de dataset_p1.csv sin prediccion U-Net: {faltantes}")

    df_test_full = df_all[df_all["split"] == "test"]
    cols_out = ["Status", "split", "V65_recto_pred", "V55_recto_pred", "V65_vejiga_pred",
                "value_RV65", "value_RV55", "value_BV65", "fail_RV65", "fail_RV55", "fail_BV65"]
    df_test_full[cols_out].to_csv(unet_dir / "unet_pred_dvh_test.csv")
    print(f"Guardado: {unet_dir / 'unet_pred_dvh_test.csv'} (n={len(df_test_full)})")
    df_all[cols_out].to_csv(unet_dir / "unet_pred_dvh_all.csv")

    print("\n=== Paso 2: calibracion CV + evaluacion homologada ===")
    df_pool = df_all[(df_all["Status"] != "UnApproved") & (df_all["split"].isin(["train", "val"]))]
    print(f"Pool de calibracion (natural train+val, UnApproved excluido): n={len(df_pool)}")

    rng = np.random.default_rng(SEED)

    resultados_constraints = {tag: evaluar_constraint_individual(df_test_full, tag, rng)
                               for tag in ("RV65", "RV55", "BV65")}
    print("AUC por constraint (U-Net):", {t: round(r["auc"], 3) for t, r in resultados_constraints.items()})

    resultado_oar = {}
    for oar_label, tags in OAR_TAGS.items():
        print(f"\n[{oar_label.upper()}]")
        r = evaluar_oar(df_pool, df_test_full, oar_label, tags, rng)
        print(f"  umbral(mediana CV)={r['umbral_mediana_cv']:.3f}  AUC={r['auc']:.3f} {r['auc_ci95']}")
        print(f"  sens={r['sens_esp_test']['sensibilidad']:.3f}  esp={r['sens_esp_test']['especificidad']:.3f}")
        resultado_oar[oar_label] = r

    # ── Paso 3: traer el clasico para la tabla comparativa ──
    clasico = None
    if Path(args.proyecto1_summary).exists():
        clasico = json.load(open(args.proyecto1_summary, encoding="utf-8"))
    else:
        print(f"[WARN] {args.proyecto1_summary} no encontrado — se guarda solo el lado U-Net")

    summary = {
        "n_test": len(df_test_full),
        "test_anonids": sorted(df_test_full.index.tolist()),
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED,
        "sens_objetivo": SENS_OBJETIVO,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "caveat_cv_unet": (
            "La calibracion CV de la U-Net NO refitea un modelo por fold (a diferencia del "
            "clasico, que refitea LogReg/Ridge por fold en calibrate_p1_cv.py) -- reusa el "
            "score fijo de un UNICO U-Net ya entrenado sobre esos mismos pacientes de "
            "train+val. La estructura de folds/mediana/IQR es la misma, pero no es una "
            "validacion out-of-fold real para la U-Net. Pesar con cautela al comparar la "
            "estabilidad (IQR) del umbral entre modelos."
        ),
        "unet": {
            "por_constraint": resultados_constraints,
            "por_oar": resultado_oar,
        },
        "clasico_proyecto1_v3_cv": clasico,
    }
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nGuardado: {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
