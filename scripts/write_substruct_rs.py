"""
Bloque 2 (HANDOFF_substructuras_dosis.md) -- submascaras -> RS modificado.

Toma el `masks.npz` de `split_oar_by_dose.py` (Bloque 1) y agrega `<ROI>_hot` /
`<ROI>_cold` como ROIs nuevas a una copia del RS original, via rt-utils. El ROI
original (ej. Rectum) NO se borra -- se necesita intacto para el DVH de calidad
global (linea de control).

Las mascaras en `masks.npz` ya estan en convencion rt-utils (Columns, Rows,
n_slices) -- las guardo asi en Bloque 1 justamente para no transponer aca.

Uso:
    .venv/Scripts/python.exe scripts/write_substruct_rs.py \
        --dicom-dir "//10.100.0.252/.../PT_44a81316c45f64f5" \
        --masks-npz results/substruct/PT_44a81316c45f64f5/masks.npz \
        --output results/substruct/PT_44a81316c45f64f5/RS_substruct.dcm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bridge.rtstruct_io import cargar_rtstruct_ct_only, regenerar_uids  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", required=True, help="Carpeta con CT.*.dcm y RS*.dcm reales del paciente")
    parser.add_argument("--masks-npz", required=True, help="Salida de split_oar_by_dose.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)

    print(f"Cargando {args.masks_npz} ...")
    data = np.load(args.masks_npz, allow_pickle=True)
    hot, cold = data["hot"], data["cold"]
    roi_name = str(data["roi_name"])
    nombre_hot, nombre_cold = f"{roi_name}_hot", f"{roi_name}_cold"
    print(f"  {nombre_hot}:  {int(hot.sum())} voxels")
    print(f"  {nombre_cold}: {int(cold.sum())} voxels")

    print(f"Cargando serie CT + RS de {dicom_dir} via rt-utils (solo CT.*.dcm) ...")
    rtstruct = cargar_rtstruct_ct_only(dicom_dir)

    disponibles = rtstruct.get_roi_names()
    for nuevo in (nombre_hot, nombre_cold):
        if nuevo in disponibles:
            raise ValueError(
                f"'{nuevo}' ya existe en el RS -- este script solo agrega ROIs, no "
                "sobrescribe (¿corriste esto antes sobre el mismo RS de salida como entrada?)."
            )
    if roi_name not in disponibles:
        print(f"  [AVISO] '{roi_name}' original no esta en este RS (¿nombre distinto?). "
              f"Disponibles: {disponibles}")

    # Verificar que la mascara nativa que se agrega calza con la serie CT recien cargada
    # (mismo n_slices) -- si masks.npz se genero contra otra carpeta/serie, mejor fallar
    # ruidosamente aca que escribir un RS desalineado.
    n_slices_rs = len(rtstruct.series_data)
    if hot.shape[2] != n_slices_rs:
        raise AssertionError(
            f"n_slices de las mascaras ({hot.shape[2]}) != CT cargada ahora ({n_slices_rs}) "
            "-- masks.npz probablemente viene de otro --dicom-dir."
        )

    print(f"Agregando '{nombre_hot}' y '{nombre_cold}' (el '{roi_name}' original queda intacto) ...")
    rtstruct.add_roi(mask=hot, name=nombre_hot, color=[255, 0, 0])
    rtstruct.add_roi(mask=cold, name=nombre_cold, color=[0, 0, 255])
    regenerar_uids(rtstruct, series_description="Substruct por dosis (Bloque 2) - TEST")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(output_path))

    print(f"\nGuardado: {output_path}")
    print(f"ROIs en el RS de salida: {rtstruct.get_roi_names()}")
    print(f"Pablo: importar en Eclipse y confirmar que '{nombre_hot}'/'{nombre_cold}' "
          f"particionan el '{roi_name}' original sin overlap ni huecos visibles.")


if __name__ == "__main__":
    main()
