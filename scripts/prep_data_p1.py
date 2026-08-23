"""Tarea 1 (Proyecto 1 — herramienta ML clasica de tomografo): carga, dedup, features
y split para el dataset hipofraccionado completo (Approved + Rejected + UnApproved).

Ver PROMPT_claude_code_proyecto ML Tomo.md para el diseno completo. Resumen:

- Fuente: data/metricas_planes_hipofx_D95norm.csv (sep ';', 199 filas incl. duplicados).
- Dedup por AnonID (keep='first') -> 195 pacientes unicos.
- Se descarta 1 paciente mas (PT_f4bd8360f920284c, Rejected) por no tener NINGUN dato de
  Rectum/Bladder (extraccion fallida, no solo el flag) -> 194 pacientes utilizables.
- 7 features geometricas pre-plan (conocibles en tomografo, sin plan):
    VolRectum_cc, VolBladder_cc, VolPTV_cc, Solap_PTV_Rectum_cc, Solap_PTV_Bladder_cc,
    overlap_rel_recto  = Solap_PTV_Rectum_cc  / VolRectum_cc,
    overlap_rel_vejiga = Solap_PTV_Bladder_cc / VolBladder_cc.
- tiene_VVSS: TODO, no hay columna en el CSV que lo indique (no bloquea, ver prompt).
- Labels (positivo = FALLA el constraint, mismo convenio que evaluate_hipo.py /
  baseline_ml_clasico.py): fail_RV65, fail_RV55, fail_BV65, derivados de
  Flag_*_D95norm==0.
- Split (splits_hipo_v3.json): val y test SOLO de poblacion natural (TreatmentApproved +
  Rejected); UnApproved entero a train. Estratificacion 2D de compliance operativo
  (recto_op_fail = RV65 fail OR RV55 fail; vejiga_op_fail = BV65 fail), reparto
  proporcional por mayor resto (Hamilton) dentro de cada celda — mismo metodo que
  scripts/make_splits_hipo.py, generaliza bien porque el vector COMPLETO de 3 flags tiene
  una celda con solo 2 miembros (demasiado chica para garantizar representacion en las 3
  particiones a la vez).

Uso:
    python scripts/prep_data_p1.py
"""

import json
import random
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = _REPO_ROOT / "data" / "metricas_planes_hipofx_D95norm.csv"
OUT_DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"
OUT_SPLITS_JSON = _REPO_ROOT / "data" / "splits" / "splits_hipo_v3.json"

SEED = 42
FRACTIONS_NATURAL = {"train": 0.65, "val": 0.15, "test": 0.20}
CELL_ORDER = ["solo_vejiga", "ambos_fallan", "solo_recto", "ambos_cumplen"]  # chicas primero

FEATURE_COLS_RAW = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
]
FEATURE_COLS_FINAL = FEATURE_COLS_RAW + ["overlap_rel_recto", "overlap_rel_vejiga"]

FLAG_COLS = {
    "RV65": "Flag_Rectum_V65Gy_lt15_D95norm",
    "RV55": "Flag_Rectum_V55Gy_lt25_D95norm",
    "BV65": "Flag_Bladder_V65Gy_lt15_D95norm",
}

# Valor continuo del DVH (%) asociado a cada flag, sin el prefijo Flag_ — usado en la
# Tarea 3 (regresion de severidad).
VALUE_COLS = {
    "RV65": "Rectum_V65Gy_lt15_D95norm",
    "RV55": "Rectum_V55Gy_lt25_D95norm",
    "BV65": "Bladder_V65Gy_lt15_D95norm",
}

REQUIRED_COLS = FEATURE_COLS_RAW + list(FLAG_COLS.values()) + list(VALUE_COLS.values())


def apportion(n, fractions):
    """Reparto proporcional por mayor resto (Hamilton), exacto en n."""
    splits = list(fractions.keys())
    exact = {s: n * fractions[s] for s in splits}
    base = {s: int(exact[s]) for s in splits}
    remainder = n - sum(base.values())
    order = sorted(splits, key=lambda s: exact[s] - base[s], reverse=True)
    for s in order[:remainder]:
        base[s] += 1
    return base


def assign_cell(ids, rng, fractions):
    ids_shuffled = list(ids)
    rng.shuffle(ids_shuffled)
    counts = apportion(len(ids_shuffled), fractions)
    out = {}
    i = 0
    for split in ("train", "val", "test"):
        n = counts[split]
        out[split] = ids_shuffled[i : i + n]
        i += n
    return out


def classify_cell_2d(row):
    recto_fail = row[FLAG_COLS["RV65"]] == 0 or row[FLAG_COLS["RV55"]] == 0
    vejiga_fail = row[FLAG_COLS["BV65"]] == 0
    if recto_fail and vejiga_fail:
        return "ambos_fallan"
    if recto_fail:
        return "solo_recto"
    if vejiga_fail:
        return "solo_vejiga"
    return "ambos_cumplen"


def cargar_y_preparar():
    df_raw = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")
    n_raw = len(df_raw)

    df = df_raw.drop_duplicates(subset="AnonID", keep="first").copy()
    n_dedup = len(df)
    n_dup_removed = n_raw - n_dedup

    faltantes = df[df[REQUIRED_COLS].isna().any(axis=1)]
    if len(faltantes) > 0:
        print(f"[WARN] Descartando {len(faltantes)} paciente(s) con features/flags faltantes "
              f"(extraccion incompleta): {faltantes['AnonID'].tolist()}")
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_final = len(df)

    print(f"Filas crudas: {n_raw}  ->  dedup por AnonID: {n_dedup} (-{n_dup_removed})  "
          f"->  sin faltantes: {n_final} (-{len(faltantes)})")

    df["overlap_rel_recto"] = df["Solap_PTV_Rectum_cc"] / df["VolRectum_cc"]
    df["overlap_rel_vejiga"] = df["Solap_PTV_Bladder_cc"] / df["VolBladder_cc"]

    for tag, col in FLAG_COLS.items():
        df[f"fail_{tag}"] = (df[col] == 0).astype(int)
    for tag, col in VALUE_COLS.items():
        df[f"value_{tag}"] = df[col]

    return df


def hacer_split(df):
    natural = df[df["Status"].isin(["TreatmentApproved", "Rejected"])].copy()
    unapproved = df[df["Status"] == "UnApproved"].copy()
    assert len(natural) + len(unapproved) == len(df), "Status inesperado fuera de las 3 categorias conocidas"

    natural["cell"] = natural.apply(classify_cell_2d, axis=1)
    cells = {c: natural.loc[natural["cell"] == c, "AnonID"].tolist() for c in CELL_ORDER}

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for cell_name in CELL_ORDER:
        assigned = assign_cell(cells[cell_name], rng, FRACTIONS_NATURAL)
        for split, ids in assigned.items():
            splits[split].extend(ids)

    # UnApproved entero a train (nunca val/test).
    splits["train"].extend(unapproved["AnonID"].tolist())

    for s in splits:
        splits[s] = sorted(splits[s])

    # --- Verificaciones de integridad ---
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert set(splits["train"]) & set(splits["val"]) == set()
    assert set(splits["train"]) & set(splits["test"]) == set()
    assert set(splits["val"]) & set(splits["test"]) == set()
    assert all_ids == set(df["AnonID"])
    assert len(splits["val"]) > 0 and all(i in natural["AnonID"].values for i in splits["val"])
    assert len(splits["test"]) > 0 and all(i in natural["AnonID"].values for i in splits["test"])

    cell_counts = {c: len(cells[c]) for c in CELL_ORDER}
    return splits, cell_counts


def reportar_prevalencias(df, splits):
    by_id = df.set_index("AnonID")
    print("\n" + "=" * 78)
    print("PREVALENCIA POR PARTICION (n = pacientes, eventos de falla por constraint)")
    print("=" * 78)
    sin_eventos = []
    for split_name in ("train", "val", "test"):
        ids = splits[split_name]
        sub = by_id.loc[ids]
        status_counts = sub["Status"].value_counts().to_dict()
        print(f"\n[{split_name}] n={len(ids)}  Status={status_counts}")
        for tag in FLAG_COLS:
            n_fail = int(sub[f"fail_{tag}"].sum())
            pct = 100.0 * n_fail / len(sub) if len(sub) else float("nan")
            print(f"    {tag}: {n_fail}/{len(sub)} fallas ({pct:.1f}%)")
            if split_name in ("val", "test") and n_fail == 0:
                sin_eventos.append((split_name, tag))
    if sin_eventos:
        print(f"\n[WARN] Particiones sin ningun evento de falla: {sin_eventos}")
    else:
        print("\nOK: val y test tienen al menos 1 evento de falla en los 3 constraints.")
    return sin_eventos


def main():
    df = cargar_y_preparar()
    splits, cell_counts = hacer_split(df)
    sin_eventos = reportar_prevalencias(df, splits)

    split_of_id = {}
    for split_name, ids in splits.items():
        for i in ids:
            split_of_id[i] = split_name
    df["split"] = df["AnonID"].map(split_of_id)

    out_cols = (
        ["AnonID", "Status", "split"] + FEATURE_COLS_FINAL
        + [f"fail_{t}" for t in FLAG_COLS] + [f"value_{t}" for t in VALUE_COLS]
    )
    df[out_cols].to_csv(OUT_DATASET_CSV, index=False)
    print(f"\nGuardado dataset: {OUT_DATASET_CSV} ({len(df)} filas, columnas: {out_cols})")

    out_json = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "metadata": {
            "fuente": str(CSV_PATH),
            "estratificacion": "2D_compliance_operativo_D95norm (recto_op_fail=RV65_fail OR RV55_fail; vejiga_op_fail=BV65_fail), solo sobre poblacion natural",
            "celdas_2d_natural": cell_counts,
            "fracciones_natural": FRACTIONS_NATURAL,
            "unapproved_a_train": True,
            "n_unapproved": int((df["Status"] == "UnApproved").sum()),
            "random_state": SEED,
            "n_total": len(df),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "particiones_sin_eventos": sin_eventos,
        },
    }
    OUT_SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"Guardado split: {OUT_SPLITS_JSON}")


if __name__ == "__main__":
    main()
