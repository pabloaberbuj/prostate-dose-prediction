"""Proyecto 1 (classical ML) rerun sobre el split nuevo del pipeline CT corregido.

Ver HIPOFX_KICKOFF.md paso "Hipo sobre pipeline corregido": la parte U-Net del proyecto
descarto el dataset viejo (bug de carga de CT) y reconstruyo el split con FOV=34cm
(data/splits/splits_hipo_ctfix_v4.json). El kickoff pide explicitamente re-entrenar la
rama de ML clasico (Proyecto 1: regresion logistica + gradient boosting) SOBRE ESE MISMO
SPLIT, para que la comparacion clasico-ML vs U-Net sea manzanas-con-manzanas en el mismo
test set (un error pasado fue comparar entre splits distintos).

Este script es un PARALELO de scripts/prep_data_p1.py (que NO se toca — sigue
perteneciendo a splits_hipo_v3.json y a los resultados ya congelados en
results/proyecto1_v3*). Reusa exactamente las mismas 7 formulas de features y la misma
logica de exclusion por datos faltantes que prep_data_p1.py. La UNICA diferencia real es
de donde sale la asignacion train/val/test:

  - prep_data_p1.py:      calcula su PROPIO split (val/test solo de poblacion natural
                           Approved+Rejected, UnApproved entero a train).
  - este script:          la asignacion sale de data/splits/splits_hipo_ctfix_v4.json
                           (split ya generado por la Tarea U-Net: 2D por cumple/no-cumple
                           x tercil de overlap PTV-Recto; UnApproved NO se enruta aparte,
                           queda mezclado en las 3 particiones igual que el resto).

AnonIDs incluidos: la union de train+val+test de splits_hipo_ctfix_v4.json, menos su
propia lista "excluidos" (ya no se solapan por construccion, se verifica igual).

Salida: data/dataset_p1_ctfix.csv (mismas columnas que data/dataset_p1.csv).

Uso:
    python scripts/prep_data_p1_ctfix.py
"""

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = _REPO_ROOT / "data" / "metricas_planes_hipofx_D95norm.csv"
SPLITS_JSON = _REPO_ROOT / "data" / "splits" / "splits_hipo_ctfix_v4.json"
OUT_DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1_ctfix.csv"

# --- Identico a prep_data_p1.py (Tarea 1) ---
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
VALUE_COLS = {
    "RV65": "Rectum_V65Gy_lt15_D95norm",
    "RV55": "Rectum_V55Gy_lt25_D95norm",
    "BV65": "Bladder_V65Gy_lt15_D95norm",
}
REQUIRED_COLS = FEATURE_COLS_RAW + list(FLAG_COLS.values()) + list(VALUE_COLS.values())


def cargar_y_preparar():
    """Identico a prep_data_p1.cargar_y_preparar(): dedup por AnonID (keep='first'),
    descarta filas con features/flags faltantes, computa overlap_rel_* y fail_*/value_*.
    """
    df_raw = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")
    n_raw = len(df_raw)

    df = df_raw.drop_duplicates(subset="AnonID", keep="first").copy()
    n_dedup = len(df)

    faltantes = df[df[REQUIRED_COLS].isna().any(axis=1)]
    if len(faltantes) > 0:
        print(f"[WARN] Descartando {len(faltantes)} paciente(s) con features/flags faltantes "
              f"(extraccion incompleta): {faltantes['AnonID'].tolist()}")
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_final = len(df)

    print(f"Filas crudas: {n_raw}  ->  dedup por AnonID: {n_dedup} (-{n_raw - n_dedup})  "
          f"->  sin faltantes: {n_final} (-{len(faltantes)})")

    df["overlap_rel_recto"] = df["Solap_PTV_Rectum_cc"] / df["VolRectum_cc"]
    df["overlap_rel_vejiga"] = df["Solap_PTV_Bladder_cc"] / df["VolBladder_cc"]

    for tag, col in FLAG_COLS.items():
        df[f"fail_{tag}"] = (df[col] == 0).astype(int)
    for tag, col in VALUE_COLS.items():
        df[f"value_{tag}"] = df[col]

    return df


def cargar_split_ctfix():
    with open(SPLITS_JSON, encoding="utf-8") as f:
        split_json = json.load(f)
    splits = {s: list(split_json[s]) for s in ("train", "val", "test")}
    excluidos = set(split_json.get("excluidos", []))

    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert set(splits["train"]) & set(splits["val"]) == set()
    assert set(splits["train"]) & set(splits["test"]) == set()
    assert set(splits["val"]) & set(splits["test"]) == set()
    overlap_excl = all_ids & excluidos
    if overlap_excl:
        print(f"[WARN] {len(overlap_excl)} AnonID(s) estan en train/val/test Y en "
              f"'excluidos' de {SPLITS_JSON.name}: {sorted(overlap_excl)} -- se excluyen.")
        all_ids -= excluidos
        for s in splits:
            splits[s] = [i for i in splits[s] if i not in excluidos]

    return splits, all_ids, split_json.get("metadata", {})


def reportar_prevalencias(df, splits):
    by_id = df.set_index("AnonID")
    print("\n" + "=" * 78)
    print("PREVALENCIA POR PARTICION (split_hipo_ctfix_v4) -- n = pacientes, eventos de falla")
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
    df_all = cargar_y_preparar()
    splits, ids_ctfix, split_meta = cargar_split_ctfix()

    faltan_en_csv = ids_ctfix - set(df_all["AnonID"])
    if faltan_en_csv:
        raise RuntimeError(
            f"{len(faltan_en_csv)} AnonID(s) de {SPLITS_JSON.name} no aparecen en "
            f"{CSV_PATH.name} tras dedup/dropna (extraccion incompleta o AnonID "
            f"inexistente): {sorted(faltan_en_csv)}"
        )

    df = df_all[df_all["AnonID"].isin(ids_ctfix)].reset_index(drop=True)
    print(f"\nAnonIDs de {SPLITS_JSON.name} presentes y utilizables: {len(df)}/{len(ids_ctfix)}")

    split_of_id = {}
    for split_name, ids in splits.items():
        for i in ids:
            split_of_id[i] = split_name
    df["split"] = df["AnonID"].map(split_of_id)
    assert df["split"].notna().all(), "Hay AnonIDs sin particion asignada"

    sin_eventos = reportar_prevalencias(df, splits)

    out_cols = (
        ["AnonID", "Status", "split"] + FEATURE_COLS_FINAL
        + [f"fail_{t}" for t in FLAG_COLS] + [f"value_{t}" for t in VALUE_COLS]
    )
    df[out_cols].to_csv(OUT_DATASET_CSV, index=False)
    print(f"\nGuardado dataset: {OUT_DATASET_CSV} ({len(df)} filas, columnas: {out_cols})")
    print(f"Split fuente: {SPLITS_JSON} (metadata: estratificacion="
          f"{split_meta.get('estratificacion')!r}, unapproved mezclado en las 3 particiones)")
    if sin_eventos:
        print(f"[WARN] particiones_sin_eventos: {sin_eventos}")


if __name__ == "__main__":
    main()
