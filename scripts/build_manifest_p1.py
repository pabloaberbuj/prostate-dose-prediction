"""Tarea 6 (Proyecto 1): serializacion final para inferencia.

Verifica que todos los artefactos de models/proyecto1/ (Tareas 2-4) cargan sin
re-fitear nada, y escribe manifest.json con la metadata necesaria para reproducir /
auditar el despliegue (version de features, orden de columnas, split usado, fecha).

Uso:
    python scripts/build_manifest_p1.py
"""

import json
from datetime import datetime
from pathlib import Path

import joblib

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO_ROOT / "models" / "proyecto1"
SPLITS_JSON = _REPO_ROOT / "data" / "splits" / "splits_hipo_v3.json"
DATASET_CSV = _REPO_ROOT / "data" / "dataset_p1.csv"

FEATURE_COLS_FINAL = [
    "VolRectum_cc", "VolBladder_cc", "VolPTV_cc",
    "Solap_PTV_Rectum_cc", "Solap_PTV_Bladder_cc",
    "overlap_rel_recto", "overlap_rel_vejiga",
]
CONSTRAINTS = ["RV65", "RV55", "BV65"]


def verificar_carga(path: Path, claves_esperadas: list) -> dict:
    obj = joblib.load(path)
    faltantes = [k for k in claves_esperadas if k not in obj]
    if faltantes:
        raise ValueError(f"{path}: faltan claves {faltantes}")
    return obj


def main():
    print("Verificando que todos los artefactos cargan sin re-fitear...")
    archivos = {}
    for tag in CONSTRAINTS:
        clf_path = MODELS_DIR / f"clf_{tag}.joblib"
        reg_path = MODELS_DIR / f"reg_{tag}.joblib"
        clf = verificar_carga(clf_path, ["scaler", "logreg", "gb", "feature_cols"])
        reg = verificar_carga(reg_path, ["scaler", "modelo", "modelo_tipo", "feature_cols", "threshold"])
        assert clf["feature_cols"] == FEATURE_COLS_FINAL, f"{clf_path}: orden de features inesperado"
        assert reg["feature_cols"] == FEATURE_COLS_FINAL, f"{reg_path}: orden de features inesperado"
        print(f"  OK  clf_{tag}.joblib (logreg + gb)")
        print(f"  OK  reg_{tag}.joblib (modelo={reg['modelo_tipo']})")
        archivos[tag] = {"clasificacion": clf_path.name, "regresion": reg_path.name, "modelo_regresion": reg["modelo_tipo"]}

    thresholds_path = MODELS_DIR / "thresholds.json"
    with open(thresholds_path, encoding="utf-8") as f:
        thresholds = json.load(f)
    print(f"  OK  {thresholds_path.name}")

    with open(SPLITS_JSON, encoding="utf-8") as f:
        splits_meta = json.load(f)["metadata"]

    manifest = {
        "version_features": "p1_v1_7geometricas",
        "feature_cols_orden": FEATURE_COLS_FINAL,
        "constraints": CONSTRAINTS,
        "modelo_desplegado_clasificacion": "logreg (StandardScaler + LogisticRegression, class_weight=balanced)",
        "modelo_control_clasificacion": "HistGradientBoostingClassifier (no se despliega, solo referencia)",
        "modelo_regresion_severidad": {tag: archivos[tag]["modelo_regresion"] for tag in CONSTRAINTS},
        "archivos": archivos,
        "thresholds_file": thresholds_path.name,
        "dataset_csv": DATASET_CSV.name,
        "splits_file": SPLITS_JSON.name,
        "splits_metadata": splits_meta,
        "fecha_generacion": datetime.now().strftime("%Y-%m-%d"),
        "notas": [
            "tiene_VVSS: feature pendiente (TODO), no incluida en esta version — no hay "
            "columna en el CSV fuente que la indique todavia.",
            "Tarea 7 (extractor en vivo) requiere que Pablo complete configs/config_p1.yaml "
            "con los nombres exactos de estructura del autocontour; el test de "
            "consistencia esta bloqueado ademas por falta de DICOMs crudos en disco "
            "(ver notas en config_p1.yaml).",
        ],
    }
    manifest_path = MODELS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {manifest_path}")


if __name__ == "__main__":
    main()
