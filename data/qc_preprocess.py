"""
QC visual del preprocesado — verifica N pacientes random del dataset procesado.

Para cada paciente muestra:
  - Corte axial central con overlay de máscaras y mapa de dosis.
  - DVH calculado desde el NPZ vs. referencia del CSV de métricas.
  - Estadísticas básicas del NPZ.

Uso:
    python data/qc_preprocess.py \
        --processed-dir data/processed \
        --metrics-csv   data/metricas_planes.csv \
        --n-patients    5 \
        --output-dir    data/qc_output
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def cargar_npz(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    resultado = {k: data[k] for k in data.files if k != 'meta'}
    if 'meta' in data.files:
        resultado['meta'] = json.loads(str(data['meta'][0]))
    return resultado


def calcular_dvh(dosis_3d: np.ndarray, mascara: np.ndarray,
                 n_bins: int = 200) -> tuple:
    """Devuelve (dosis_bins, volumen_acumulado_pct)."""
    dosis_roi = dosis_3d[mascara > 0]
    if len(dosis_roi) == 0:
        return np.array([0, 150]), np.array([100, 0])
    bins = np.linspace(0, max(150, dosis_roi.max() * 1.05), n_bins)
    vol_pct = np.array([
        100.0 * (dosis_roi >= b).sum() / len(dosis_roi)
        for b in bins
    ])
    return bins, vol_pct


def qc_paciente(npz_path: Path, metricas_row: pd.Series, output_dir: Path):
    data  = cargar_npz(npz_path)
    meta  = data.get('meta', {})
    anonid = meta.get('anonid', npz_path.stem)

    ct      = data['ct']           # ZYX, [-1, 1]
    dose    = data['dose']         # ZYX, % prescripción
    ptv     = data['ptv_mask']
    rectum  = data['rectum_mask']
    bladder = data['bladder_mask']
    body    = data['body_mask']

    n_slices, ny, nx = ct.shape
    z_central = n_slices // 2

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'QC: {anonid}  |  {n_slices} cortes  |  PTV: {meta.get("vol_ptv_cc","?")} cc',
                 fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # ── Columnas 0-2: cortes axiales (inferior, central, superior del ROI)
    cortes_mostrar = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
    titulos_cortes = ['Corte inferior ROI', 'Corte central', 'Corte superior ROI']

    for i, (z, titulo) in enumerate(zip(cortes_mostrar, titulos_cortes)):
        ax = fig.add_subplot(gs[0, i])
        # CT en escala de grises
        ct_slice = ct[z]
        ax.imshow(ct_slice, cmap='gray', vmin=-1, vmax=1, aspect='equal')
        # Overlay dosis
        dose_slice = dose[z].copy()
        dose_slice[body[z] == 0] = np.nan
        ax.imshow(dose_slice, cmap='jet', alpha=0.4, vmin=0, vmax=120, aspect='equal')
        # Contornos de estructuras
        for mask, color, label in [
            (ptv[z],     'yellow', 'PTV'),
            (rectum[z],  'red',    'Rectum'),
            (bladder[z], 'cyan',   'Bladder'),
        ]:
            if mask.sum() > 0:
                ax.contour(mask, levels=[0.5], colors=[color], linewidths=0.8)
        ax.set_title(f'{titulo} (z={z})', fontsize=8)
        ax.axis('off')

    # ── Columna 3 fila 0: PSDM PTV corte central
    ax_psdm = fig.add_subplot(gs[0, 3])
    if 'psdm_ptv' in data:
        psdm_slice = data['psdm_ptv'][z_central]
        im = ax_psdm.imshow(psdm_slice, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        plt.colorbar(im, ax=ax_psdm, shrink=0.8)
        ax_psdm.set_title('PSDM PTV (corte central)', fontsize=8)
    ax_psdm.axis('off')

    # ── Fila 1: DVH de PTV, Rectum y Bladder
    dvh_structs = [
        ('PTV',     ptv,     'blue'),
        ('Rectum',  rectum,  'red'),
        ('Bladder', bladder, 'cyan'),
    ]

    ax_dvh = fig.add_subplot(gs[1, :2])
    for nombre, mask, color in dvh_structs:
        if mask.sum() > 0:
            bins, vols = calcular_dvh(dose, mask)
            ax_dvh.plot(bins, vols, color=color, linewidth=1.5, label=nombre)

    # Líneas de constraint para referencia visual
    ax_dvh.axvline(95,  color='blue',  linestyle=':', linewidth=0.8, alpha=0.6, label='95% pres')
    ax_dvh.axvline(100, color='blue',  linestyle='--', linewidth=0.8, alpha=0.6, label='100% pres')
    ax_dvh.axhline(25,  color='red',   linestyle=':', linewidth=0.8, alpha=0.6, label='Rect V70=25%')
    ax_dvh.axhline(50,  color='cyan',  linestyle=':', linewidth=0.8, alpha=0.6, label='Blad V65=50%')
    ax_dvh.set_xlabel('Dosis (% prescripción)', fontsize=8)
    ax_dvh.set_ylabel('Volumen (%)', fontsize=8)
    ax_dvh.set_title('DVH desde NPZ', fontsize=9, fontweight='bold')
    ax_dvh.legend(fontsize=7, loc='upper right')
    ax_dvh.set_xlim(0, 130)
    ax_dvh.set_ylim(0, 105)
    ax_dvh.tick_params(labelsize=7)
    ax_dvh.grid(alpha=0.3)

    # ── Comparación métricas NPZ vs CSV ESAPI
    ax_txt = fig.add_subplot(gs[1, 2:])
    ax_txt.axis('off')

    # Calcular métricas desde el NPZ
    def vx_npz(mask, dosis_pct_umbral):
        roi = dose[mask > 0]
        if len(roi) == 0: return float('nan')
        return 100.0 * (roi >= dosis_pct_umbral).sum() / len(roi)

    def dx_npz(mask, vol_pct):
        roi = dose[mask > 0]
        if len(roi) == 0: return float('nan')
        return float(np.percentile(roi, 100 - vol_pct))

    # Comparación (NPZ vs ESAPI del CSV)
    lineas = [
        ('Métrica',          'NPZ',                                     'ESAPI CSV'),
        ('PTV D95 (%pres)',   f"{dx_npz(ptv, 95):.1f}",                 "~100 (norm)"),
        ('PTV V95 (%vol)',    f"{vx_npz(ptv, 95):.1f}",                 f"{metricas_row.get('PTV_V95pct', '?'):.1f}" if pd.notna(metricas_row.get('PTV_V95pct')) else '?'),
        ('Rect V70 (%vol)',   f"{vx_npz(rectum, 70/0.78):.1f}",         f"{metricas_row.get('Rectum_V70pct', '?'):.1f}" if pd.notna(metricas_row.get('Rectum_V70pct')) else '?'),
        ('Rect Dmean (%)',    f"{dose[rectum>0].mean():.1f}" if rectum.sum()>0 else 'n/a', '—'),
        ('Blad V65 (%vol)',   f"{vx_npz(bladder, 65/0.78):.1f}",        f"{metricas_row.get('Bladder_V65pct', '?'):.1f}" if pd.notna(metricas_row.get('Bladder_V65pct')) else '?'),
        ('FactorNorm',        f"{meta.get('factor_norm', '?'):.4f}",     f"{metricas_row.get('FactorNorm_D95', '?'):.4f}" if pd.notna(metricas_row.get('FactorNorm_D95')) else '?'),
        ('N cortes',          str(n_slices),                             '—'),
    ]

    texto = '\n'.join([f"{l[0]:<20} {l[1]:<12} {l[2]}" for l in lineas])
    ax_txt.text(0.02, 0.95, texto, transform=ax_txt.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_txt.set_title('Comparación métricas', fontsize=9, fontweight='bold')

    plt.savefig(str(output_dir / f"qc_{anonid}.png"), dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--metrics-csv",   required=True)
    parser.add_argument("--n-patients",    type=int, default=5)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--anonids",       nargs="+", help="AnonIDs específicos a revisar")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metrics_csv, sep=';')
    df = df.set_index('AnonID')

    npz_files = list(Path(args.processed_dir).glob("*.npz"))
    print(f"NPZs disponibles: {len(npz_files)}")

    if args.anonids:
        seleccion = [Path(args.processed_dir) / f"{a}.npz" for a in args.anonids]
        seleccion = [p for p in seleccion if p.exists()]
    else:
        random.seed(args.seed)
        seleccion = random.sample(npz_files, min(args.n_patients, len(npz_files)))

    print(f"Revisando {len(seleccion)} pacientes...")
    for npz_path in seleccion:
        anonid = npz_path.stem
        metricas = df.loc[anonid] if anonid in df.index else pd.Series()
        try:
            qc_paciente(npz_path, metricas, output_dir)
            print(f"  ✓ {anonid}")
        except Exception as e:
            print(f"  ✗ {anonid}: {e}")

    print(f"\nImágenes de QC guardadas en {output_dir}")


if __name__ == "__main__":
    main()