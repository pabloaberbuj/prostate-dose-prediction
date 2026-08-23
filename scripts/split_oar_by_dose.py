"""
Bloque 1 (HANDOFF_substructuras_dosis.md) -- split de dosis predicha en submascaras.

Divide un OAR (por defecto Rectum) en `<ROI>_hot` / `<ROI>_cold` segun si la dosis
predicha por la U-Net (`dose_pred`, de `predict_one.py`) supera un umbral, en
resolucion NATIVA de la CT (no la de 256x256 del modelo). Particion exacta por
construccion: `cold = roi_mask & ~hot`, nunca se limpian por separado.

Pipeline:
  1. Clip + renormalizar dose_pred a D95(PTV)=100% -- igual que compute_pred_dvh.py,
     sobre el grid 256x256 (mismo criterio que usa el resto del pipeline KBP).
  2. Resamplear esa dosis normalizada al grid NATIVO de la CT
     (`unet_to_target.dose_pred_a_grid_nativo`, misma calibracion in-plane empirica
     ya validada en el piloto RD -- ver `write_rd_dicom.py`).
  3. Leer la mascara nativa del OAR via rt-utils (`get_roi_mask_by_name`, NO la de
     256x256 del npz).
  4. Umbralar (modo `pct_rx`: dosis >= X% Rx: donde viven V70/V65 con umbral 85%;
     modo `percentil`: percentil P de la dosis dentro del propio OAR).
  5. Limpieza morfologica de `hot` (closing 3D en mm reales + remover componentes
     conexas chicas), intersectando siempre de nuevo con `roi_mask` para no crecer
     fuera del organo. `cold` se recalcula como complemento -- garantiza
     `hot | cold == roi_mask` sin chequeo adicional.

Uso:
    .venv/Scripts/python.exe scripts/split_oar_by_dose.py \
        --dicom-dir "//10.100.0.252/.../PT_44a81316c45f64f5" \
        --pred-npz results/substruct/PT_44a81316c45f64f5/pred_PT_44a81316c45f64f5.npz \
        --output results/substruct/PT_44a81316c45f64f5/masks.npz \
        --umbral-modo pct_rx --umbral-valor 85
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.bridge.unet_to_target import leer_geometria_ct, dose_pred_a_grid_nativo  # noqa: E402
from src.bridge.mask_morph import close_mm  # noqa: E402
from src.bridge.rtstruct_io import cargar_rtstruct_ct_only  # noqa: E402
from compute_pred_dvh import d95_ptv  # noqa: E402


def limpiar_hot(hot_raw: np.ndarray, roi_mask: np.ndarray, spacing_rtutils_axes: tuple,
                 closing_mm: float, min_component_voxels: int) -> np.ndarray:
    """Closing 3D en mm reales + descarte de componentes conexas chicas. Siempre
    intersecta con `roi_mask` (el closing/dilation puede crecer fuera del organo)."""
    hot = close_mm(hot_raw, spacing_rtutils_axes, closing_mm) & roi_mask
    if hot.sum() == 0:
        return hot
    labeled, n = label(hot)
    if n == 0:
        return hot
    tamanos = np.bincount(labeled.ravel())[1:]  # indice 0 = fondo
    componentes_chicos = np.where(tamanos < min_component_voxels)[0] + 1
    if len(componentes_chicos):
        hot = hot & ~np.isin(labeled, componentes_chicos)
        print(f"  Descartadas {len(componentes_chicos)} componentes conexas "
              f"< {min_component_voxels} voxels (tamanos: {sorted(tamanos[tamanos < min_component_voxels])})")
    return hot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-dir", required=True, help="Carpeta con CT.*.dcm y RS*.dcm reales del paciente")
    parser.add_argument("--pred-npz", required=True, help="Salida de predict_one.py")
    parser.add_argument("--output", required=True, help="Ruta de salida, ej. results/substruct/<PT>/masks.npz")
    parser.add_argument("--roi-name", default="Rectum")
    parser.add_argument("--umbral-modo", choices=["pct_rx", "percentil"], default="pct_rx")
    parser.add_argument("--umbral-valor", type=float, default=85.0,
                         help="pct_rx: %% de Rx (ej. 85). percentil: percentil dentro del OAR (ej. 66)")
    parser.add_argument("--closing-mm", type=float, default=3.0)
    parser.add_argument("--min-component-voxels", type=int, default=50)
    parser.add_argument("--sin-guardar-dosis-nativa", action="store_true",
                         help="No guardar el array de dosis nativa en el npz (mas liviano, pero Bloque 3 "
                              "tendria que re-derivarlo)")
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)

    print(f"Leyendo geometria CT de {dicom_dir} ...")
    ct_geom = leer_geometria_ct(dicom_dir)
    row_spacing, col_spacing = ct_geom["pixel_spacing"]
    z_steps = np.diff(ct_geom["ipp_z_per_slice"])
    z_spacing = float(np.median(z_steps)) if z_steps.size else 1.0
    # Orden para arrays en convencion rt-utils (ver gotcha en unet_to_target.py::
    # dose_pred_a_grid_nativo): axis0 (rt-utils lo llama "Columns") = eje Y ->
    # row_spacing; axis1 ("Rows") = eje X -> col_spacing; axis2 = slices.
    spacing_rtutils_axes = (row_spacing, col_spacing, z_spacing)
    print(f"  spacing: col={col_spacing:.4f}mm row={row_spacing:.4f}mm z={z_spacing:.4f}mm "
          f"n_slices={ct_geom['n_slices']}")

    print(f"Cargando prediccion de {args.pred_npz} ...")
    data = np.load(args.pred_npz, allow_pickle=True)
    dose_pred_raw = np.asarray(data["dose_pred"], dtype=np.float32)
    ptv_mask_256 = data["ptv_mask"]
    meta = json.loads(str(data["meta"][0]))

    # Paso 1: clip + renormalizar a D95(PTV)=100%, igual que compute_pred_dvh.py.
    dose_pred_raw = np.clip(dose_pred_raw, 0.0, None)
    d95_pred = d95_ptv(dose_pred_raw, ptv_mask_256)
    factor_renorm = 100.0 / d95_pred
    dose_pred_norm = dose_pred_raw * factor_renorm
    print(f"  D95(PTV_pred) pre-renorm={d95_pred:.2f}%  factor_renorm={factor_renorm:.4f}")

    # Paso 2: resamplear al grid nativo de la CT.
    print("Resampleando dosis predicha al grid nativo (calibracion in-plane via PTV real) ...")
    dose_native = dose_pred_a_grid_nativo(dose_pred_norm, meta, ct_geom, ptv_mask_256, dicom_dir)
    # (1,2,0), NO (2,1,0): ver gotcha documentado en dose_pred_a_grid_nativo -- el
    # eje que rt-utils llama "Columns" es en realidad Y (axis1 de dose_native), y
    # el que llama "Rows" es en realidad X (axis2). Verificado empiricamente
    # contra las Dose Statistics de Eclipse para PTV_High (match casi exacto:
    # Mean 102.3 vs 102.4%, Min 76.6 vs 77.7%, Max 107.8 vs 108.1%).
    dose_native_rt = dose_native.transpose(1, 2, 0)  # -> convencion rt-utils (Columns, Rows, n_slices)

    # Paso 3: mascara nativa del OAR via rt-utils (NO la de 256x256).
    print(f"Leyendo mascara nativa de '{args.roi_name}' via rt-utils ...")
    rtstruct = cargar_rtstruct_ct_only(dicom_dir)
    if args.roi_name not in rtstruct.get_roi_names():
        raise ValueError(f"'{args.roi_name}' no esta en el RS. Disponibles: {rtstruct.get_roi_names()}")
    roi_mask = rtstruct.get_roi_mask_by_name(args.roi_name)
    if roi_mask.shape != dose_native_rt.shape:
        raise AssertionError(
            f"Shape mascara nativa {roi_mask.shape} != shape dosis nativa {dose_native_rt.shape}"
        )
    vol_roi_cc = roi_mask.sum() * col_spacing * row_spacing * z_spacing / 1000.0
    print(f"  {args.roi_name}: shape={roi_mask.shape}  volumen={vol_roi_cc:.1f}cc")

    # Paso 4: umbral.
    dosis_en_roi = dose_native_rt[roi_mask]
    if args.umbral_modo == "pct_rx":
        umbral_pct = args.umbral_valor
    else:
        umbral_pct = float(np.percentile(dosis_en_roi, args.umbral_valor))
    print(f"  Umbral ({args.umbral_modo}={args.umbral_valor}) -> {umbral_pct:.2f}% Rx  "
          f"(dosis en {args.roi_name}: min={dosis_en_roi.min():.1f}% max={dosis_en_roi.max():.1f}%)")

    hot_raw = roi_mask & (dose_native_rt >= umbral_pct)
    if hot_raw.sum() == 0:
        raise ValueError(
            f"Umbral {umbral_pct:.2f}% no deja ningun voxel de {args.roi_name} -- bajar --umbral-valor."
        )

    # Paso 5: limpieza morfologica de hot; cold = complemento exacto dentro del OAR.
    print(f"Limpieza morfologica: closing {args.closing_mm}mm + descarte de componentes "
          f"< {args.min_component_voxels} voxels ...")
    hot = limpiar_hot(hot_raw, roi_mask, spacing_rtutils_axes, args.closing_mm, args.min_component_voxels)
    cold = roi_mask & ~hot

    vol_hot_cc = hot.sum() * col_spacing * row_spacing * z_spacing / 1000.0
    vol_cold_cc = cold.sum() * col_spacing * row_spacing * z_spacing / 1000.0
    print(f"  {args.roi_name}_hot:  {int(hot.sum())} voxels  {vol_hot_cc:.1f}cc")
    print(f"  {args.roi_name}_cold: {int(cold.sum())} voxels  {vol_cold_cc:.1f}cc")
    assert np.array_equal(hot | cold, roi_mask), "hot|cold != roi_mask -- particion rota (no deberia pasar)"
    for nombre, m in [(f"{args.roi_name}_hot", hot), (f"{args.roi_name}_cold", cold)]:
        if m.sum() < 50:
            print(f"  [AVISO] {nombre} tiene {int(m.sum())} voxels (< 50) -- ajustar umbral/closing.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        hot=hot, cold=cold, roi_mask=roi_mask,
        roi_name=args.roi_name,
        umbral_modo=args.umbral_modo, umbral_valor=args.umbral_valor, umbral_pct_efectivo=umbral_pct,
        closing_mm=args.closing_mm, min_component_voxels=args.min_component_voxels,
        spacing_rtutils_axes=np.array(spacing_rtutils_axes),
        anonid=meta["anonid"],
        dicom_dir=str(dicom_dir), pred_npz_origen=str(args.pred_npz),
        factor_renorm_aplicado=factor_renorm,
    )
    if not args.sin_guardar_dosis_nativa:
        save_kwargs["dose_native_pct"] = dose_native_rt.astype(np.float32)
    np.savez_compressed(str(output_path), **save_kwargs)
    print(f"\nGuardado: {output_path}")
    print("Convencion de mascaras guardadas: rt-utils (Columns, Rows, n_slices) -- "
          "listas para Bloque 2 (write_substruct_rs.py) sin transponer.")


if __name__ == "__main__":
    main()
