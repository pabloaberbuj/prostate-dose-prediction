"""
Bloque 0 (HANDOFF_substructuras_dosis.md) -- de-risk aislado: mascara -> RS -> import Eclipse.

Genera una mascara dummy trivial (un ROI existente del RS real, erosionado N mm, en
geometria NATIVA de la CT) y la agrega como ROI nuevo a una copia del RS original via
rt-utils. Objetivo unico de este script: que Pablo importe el RS resultante en Eclipse
y confirme que el ROI nuevo aparece alineado con la anatomia (sin corrimiento/rotacion,
FrameOfReferenceUID y posiciones OK). Si esto falla, los Bloques 1-3 no sirven --
resolver alineacion aca antes de seguir.

Deliberadamente NO usa dose_pred: la mascara sale de erosionar un ROI existente del
RS real, para aislar el riesgo de rt-utils/geometria del riesgo de la calibracion
in-plane del grid 256x256 (ese es un riesgo aparte, ver
src/bridge/unet_to_target.py::calibrar_grid_256_inplane, que recien entra en juego en
split_oar_by_dose.py -- Bloque 1).

Gotchas encontrados y resueltos en `src/bridge/rtstruct_io.py`:
- `RTStructBuilder.create_from` escanea TODOS los archivos del directorio con
  `pixel_array`, y el RD (RTDOSE) tambien tiene PixelData -- se cuela como corte
  extra de la serie si no se filtra a solo CT.*.dcm (ver docstring de ese modulo).

Segundo gotcha (confirmado por Pablo al intentar importar), resuelto en
`rtstruct_io.py::regenerar_uids`: `RTStructBuilder.create_from` lee el RS original
con `dcmread` sin tocar sus UIDs -- el `.ds` resultante conserva el mismo
SOPInstanceUID/SeriesInstanceUID que el RS YA importado en Eclipse, que lo rechaza
como duplicado.

Convencion de ejes: rt-utils devuelve/espera mascaras 3D con shape (Columns, Rows,
n_slices) -- ver rt_utils.image_helper.create_empty_series_mask. Este script lee la
mascara nativa CON rt-utils (get_roi_mask_by_name) y la reescribe con rt-utils
(add_roi) sin ninguna transposicion manual: al usar la misma convencion rt-utils en
ambos extremos, el riesgo de "eje trocado" queda aislado (no hay round-trip por un
array numpy en convencion (row, col) estandar en el medio).

Uso:
    .venv/Scripts/python.exe scripts/substruct_derisk_mask.py \
        --dicom-dir ../dicom_pilot/PT_003fe2bb84986507 \
        --roi-name Rectum \
        --erode-mm 3 \
        --output results/substruct/PT_003fe2bb84986507/RS_derisk.dcm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bridge.unet_to_target import leer_geometria_ct  # noqa: E402
from src.bridge.mask_morph import erode_mm  # noqa: E402
from src.bridge.rtstruct_io import cargar_rtstruct_ct_only, regenerar_uids  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", required=True, help="Carpeta con CT.*.dcm y RS*.dcm reales del paciente")
    parser.add_argument("--roi-name", default="Rectum")
    parser.add_argument("--erode-mm", type=float, default=3.0)
    parser.add_argument("--new-roi-name", default="TEST_ROI_derisk")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)

    print(f"Leyendo geometria CT de {dicom_dir} ...")
    ct_geom = leer_geometria_ct(dicom_dir)  # valida orientacion axial pura, da spacing/z
    row_spacing, col_spacing = ct_geom["pixel_spacing"]  # DICOM: PixelSpacing=[row_spacing, column_spacing]
    z_steps = np.diff(ct_geom["ipp_z_per_slice"])
    z_spacing = float(np.median(z_steps)) if z_steps.size else 1.0
    print(f"  spacing: col={col_spacing:.4f}mm row={row_spacing:.4f}mm z={z_spacing:.4f}mm "
          f"n_slices={ct_geom['n_slices']}")

    print("Cargando serie CT via rt-utils (solo CT.*.dcm, sin RD/RP contaminando) ...")
    rtstruct = cargar_rtstruct_ct_only(dicom_dir)

    disponibles = rtstruct.get_roi_names()
    if args.roi_name not in disponibles:
        raise ValueError(f"'{args.roi_name}' no esta en el RS. Disponibles: {disponibles}")

    print(f"Leyendo mascara nativa de '{args.roi_name}' via rt-utils ...")
    mask = rtstruct.get_roi_mask_by_name(args.roi_name)  # (Columns, Rows, n_slices) bool
    if mask.shape[2] != ct_geom["n_slices"]:
        raise AssertionError(
            f"n_slices de la mascara ({mask.shape[2]}) != CT real ({ct_geom['n_slices']}) "
            "-- revisar filtrado de la serie."
        )
    vol_before_cc = mask.sum() * col_spacing * row_spacing * z_spacing / 1000.0
    print(f"  shape={mask.shape}  volumen={vol_before_cc:.1f}cc")

    print(f"Erosionando {args.erode_mm}mm (distancia euclidea real, sampling anisotropico) ...")
    # axis0 (rt-utils lo llama "Columns") = eje Y -> row_spacing; axis1 ("Rows") =
    # eje X -> col_spacing; axis2 = slices -> z_spacing. Ver gotcha verificado
    # empiricamente en unet_to_target.py::dose_pred_a_grid_nativo -- para spacing
    # in-plane isotropico (como en los pacientes probados hasta ahora) esto no
    # cambia el resultado, pero es el orden correcto en general.
    eroded = erode_mm(mask, (row_spacing, col_spacing, z_spacing), args.erode_mm)
    vol_after_cc = eroded.sum() * col_spacing * row_spacing * z_spacing / 1000.0
    print(f"  volumen tras erosion={vol_after_cc:.1f}cc  ({int(eroded.sum())} voxels)")
    if eroded.sum() < 50:
        print("  [AVISO] menos de 50 voxels tras erosionar -- bajar --erode-mm.")

    rtstruct.add_roi(mask=eroded, name=args.new_roi_name, color=[255, 0, 0])
    regenerar_uids(rtstruct, series_description="Substruct de-risk (Bloque 0) - TEST")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rtstruct.save(str(output_path))

    print(f"\nGuardado: {output_path}")
    print(f"Pablo: importar este RS en Eclipse y confirmar que '{args.new_roi_name}' "
          f"aparece alineado con la anatomia real (deberia ser un {args.roi_name} "
          f"'encogido' concentricamente {args.erode_mm}mm, sin corrimiento/rotacion, "
          "mismo FrameOfReferenceUID).")


if __name__ == "__main__":
    main()
