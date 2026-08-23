"""Métricas numéricas de bondad de ajuste del comisionamiento (en vez de
sólo mirar `report.png`). Corre los mismos 3 pasos que
`run_commissioning_pipeline.py` (mismos parámetros/energía) y para cada
perfil/diagonal medido compara contra el simulado con el modelo ya
ajustado:

  - tamaño de campo (cruce al 50%) medido vs simulado -> delta en mm
  - penumbra 20-80% medida vs simulada -> delta en mm
  - RMSE y diferencia máx. de dosis (%, normalizada al eje central) en la
    zona de campo (>50% de dosis)
  - índice gamma 1D (dosis global %, DTA mm), criterios 3%/3mm y 2%/2mm

Para output factors, imprime el residuo medido/modelo por cada tamaño de
campo de `outputs.txt` (ya lo calcula el propio ajuste, `fit_output_factors`
lo deja en cada `OutputFactorMeasurement.residual`).

Uso: python commissioning/qc_report.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toolkit.commissioning_parser import MeasurementParser
from toolkit.commissioning_toolkit import CommissioningToolkit

BASE_CONFIG = str(Path(__file__).resolve().parent / "machine_config_base_varian.json")
DATA_DIR = Path(__file__).resolve().parent / "data" / "6x"
ENERGY = "6MV"
KERNEL_SIZE_MM = 400.0

PENUMBRA_FIELD_MM = (100.0, 100.0)
PENUMBRA_DEPTH_MM = 100.0
PROFILE_PLATEAU_DOSE_THRESHOLD = 0.75
PROFILE_PLATEAU_POSITION_FRACTION = 0.85
PROFILE_DIAGONAL_CUTOFF_DEG = 13.0
HS_AXES = ["X", "Y"]
HS_DEPTHS_MM = [100.0]
HS_FIELDS_CM = [(100.0, 100.0), (200.0, 200.0)]
HS_BANDS_PCT = [(40.0, 90.0), (110.0, 150.0)]
HS_BAND_WEIGHTS = [100.0, 5.0]


def crossing(pos, dose, level):
    """Primer y último cruce de `level` (%) en la curva, por interpolación lineal."""
    above = dose >= level
    if not above.any():
        return None, None
    idx = np.where(above)[0]
    left_i, right_i = idx[0], idx[-1]

    def interp_edge(i0, i1):
        if i0 == i1:
            return pos[i0]
        x0, x1, y0, y1 = pos[i0], pos[i1], dose[i0], dose[i1]
        if y1 == y0:
            return x1
        return x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    left = interp_edge(left_i - 1, left_i) if left_i > 0 else pos[left_i]
    right = interp_edge(right_i, right_i + 1) if right_i < len(pos) - 1 else pos[right_i]
    return left, right


def penumbra_width(pos, dose):
    left20, right20 = crossing(pos, dose, 20.0)
    left80, right80 = crossing(pos, dose, 80.0)
    if None in (left20, right20, left80, right80):
        return None
    return 0.5 * ((left80 - left20) + (right20 - right80))


def gamma_1d(pos_m, dose_m, pos_s, dose_s, dose_crit_pct, dta_mm, mask=None):
    """Gamma 1D global (normalizado a 100), pos en mm, dosis en %."""
    if mask is not None:
        pos_m, dose_m = pos_m[mask], dose_m[mask]
    fine_pos = np.arange(pos_s.min(), pos_s.max(), 0.25)
    fine_dose = np.interp(fine_pos, pos_s, dose_s)
    dx = (pos_m[:, None].astype(np.float32) - fine_pos[None, :].astype(np.float32)) / dta_mm
    dd = (dose_m[:, None].astype(np.float32) - fine_dose[None, :].astype(np.float32)) / dose_crit_pct
    gamma = np.sqrt(dx * dx + dd * dd).min(axis=1)
    return float(np.mean(gamma <= 1.0)) * 100.0


def main():
    toolkit = CommissioningToolkit(BASE_CONFIG, verbose=False, kernel_size_mm=KERNEL_SIZE_MM)

    profiles = MeasurementParser.parse_json_profiles(str(DATA_DIR / "profiles_6MV.json"))
    diagonals = MeasurementParser.parse_json_profiles(str(DATA_DIR / "diagonals_6MV.json"))
    of_meas = MeasurementParser.parse_output_factors_json(str(DATA_DIR / "output_factors_6MV.json"))

    print("Ajustando (paso 1/3: penumbra)...")
    toolkit.fit_geometric_penumbra(
        profiles, target_field_mm=PENUMBRA_FIELD_MM, target_depth_mm=PENUMBRA_DEPTH_MM
    )
    print(f"  jaw_offset_mm={toolkit.jaw_offset_mm}  jaw_scale (crudo)={toolkit.jaw_scale}")
    # Corrección: jaw_scale = semiancho_medido/semiancho_nominal en el campo y
    # profundidad de referencia ya incluye la magnificación por divergencia a
    # esa profundidad; _field_size_scaled() la vuelve a aplicar -> divergencia
    # contada dos veces. Ver comentario igual en run_commissioning_pipeline.py.
    ssd_for_scale = float(profiles[0].ssd_mm) if profiles else 1000.0
    mag_ref = (ssd_for_scale + PENUMBRA_DEPTH_MM) / 1000.0
    toolkit.jaw_scale = (toolkit.jaw_scale[0] / mag_ref, toolkit.jaw_scale[1] / mag_ref)
    print(f"  jaw_scale (corregido)={toolkit.jaw_scale}")

    print("Ajustando (paso 2/3: off-axis)...")
    toolkit.fit_profile_correction(
        diagonals,
        plateau_dose_threshold=PROFILE_PLATEAU_DOSE_THRESHOLD,
        plateau_position_fraction=PROFILE_PLATEAU_POSITION_FRACTION,
        diagonal_cutoff_deg=PROFILE_DIAGONAL_CUTOFF_DEG,
    )

    print("Ajustando (paso 3/3: head scatter / output factors)...")
    toolkit.fit_output_factors(
        of_meas,
        energy=ENERGY,
        tail_profiles=profiles,
        axes=HS_AXES,
        depths_mm=HS_DEPTHS_MM,
        fields_mm=HS_FIELDS_CM,
        bands_pct=HS_BANDS_PCT,
        band_weights=HS_BAND_WEIGHTS,
    )

    # --- Output factors: residuo medido/modelo por tamaño de campo ---
    print("\n=== OUTPUT FACTORS: medido vs modelo (residuo = medido/modelo, 1.00=perfecto) ===")
    print(f"{'campo YxX (cm)':>16} {'OF medido':>10} {'OF modelo':>10} {'residuo':>8}")
    for m in sorted(of_meas, key=lambda m: (m.field_y_mm, m.field_x_mm)):
        of_model = m.sc_model * m.sp
        campo = f"{m.field_y_mm/10:g}x{m.field_x_mm/10:g}"
        print(f"{campo:>16} {m.value:>10.4f} {of_model:>10.4f} {m.residual:>8.3f}")
    residuals = np.array([m.residual for m in of_meas])
    print(f"residuo: media={residuals.mean():.3f}  max|1-r|={np.max(np.abs(1 - residuals)):.3f}")

    # --- Perfiles y diagonales: medido vs simulado ---
    all_profiles = list(profiles) + list(diagonals)
    sim_map = toolkit.simulate_profiles_for_report(all_profiles, res_mm=0.5)

    print("\n=== PERFILES/DIAGONAL: medido vs simulado (normalizado al eje central) ===")
    header = (
        f"{'tipo':>4} {'campo(mm)':>11} {'prof(mm)':>9} "
        f"{'campo_med':>10} {'campo_sim':>10} {'Dcampo':>7} "
        f"{'pen_med':>8} {'pen_sim':>8} {'Dpen':>6} "
        f"{'RMSE%':>7} {'max%':>7} {'g3/3':>6} {'g2/2':>6}"
    )
    print(header)
    print("-" * len(header))

    for p in all_profiles:
        key = (
            p.energy,
            int(p.id),
            p.axis.upper(),
            round(float(p.depth_mm or 0.0), 3),
            round(float(p.field_size_mm[0]), 3),
            round(float(p.field_size_mm[1]), 3),
            int(p.position_mm.shape[0]),
        )
        sim = sim_map.get(key)
        if sim is None:
            print(f"[sin simulación para id={p.id} axis={p.axis} field={p.field_size_mm} depth={p.depth_mm}]")
            continue

        adj_pos = np.asarray(p.position_mm, dtype=float) - toolkit._axis_offset_mm(p)
        cax_idx = int(np.argmin(np.abs(adj_pos)))
        meas = p.dose_values
        meas_norm = 100.0 * meas / (meas[cax_idx] if meas[cax_idx] != 0 else meas.max())
        sim_norm = 100.0 * sim / (sim[cax_idx] if sim[cax_idx] != 0 else sim.max())

        if p.scan_type == "DIA":
            fs_med = fs_sim = pen_med = pen_sim = float("nan")
        else:
            l_m, r_m = crossing(adj_pos, meas_norm, 50.0)
            l_s, r_s = crossing(adj_pos, sim_norm, 50.0)
            fs_med = (r_m - l_m) if l_m is not None else float("nan")
            fs_sim = (r_s - l_s) if l_s is not None else float("nan")
            pen_med = penumbra_width(adj_pos, meas_norm)
            pen_sim = penumbra_width(adj_pos, sim_norm)
            pen_med = pen_med if pen_med is not None else float("nan")
            pen_sim = pen_sim if pen_sim is not None else float("nan")

        in_field = meas_norm > 50.0
        diff = sim_norm - meas_norm
        rmse = float(np.sqrt(np.mean(diff[in_field] ** 2))) if in_field.any() else float("nan")
        maxdiff = float(np.max(np.abs(diff[in_field]))) if in_field.any() else float("nan")

        g33 = gamma_1d(adj_pos, meas_norm, adj_pos, sim_norm, dose_crit_pct=3.0, dta_mm=3.0)
        g22 = gamma_1d(adj_pos, meas_norm, adj_pos, sim_norm, dose_crit_pct=2.0, dta_mm=2.0)

        tipo = p.scan_type
        campo = f"{p.field_size_mm[0]:.0f}x{p.field_size_mm[1]:.0f}"
        print(
            f"{tipo:>4} {campo:>11} {p.depth_mm:>9.1f} "
            f"{fs_med:>10.1f} {fs_sim:>10.1f} {fs_sim-fs_med:>7.1f} "
            f"{pen_med:>8.2f} {pen_sim:>8.2f} {pen_sim-pen_med:>6.2f} "
            f"{rmse:>7.2f} {maxdiff:>7.2f} {g33:>5.1f}% {g22:>5.1f}%"
        )


if __name__ == "__main__":
    main()
