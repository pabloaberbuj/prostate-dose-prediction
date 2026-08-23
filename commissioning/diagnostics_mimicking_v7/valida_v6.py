"""PDRT fresco (misma ruta de codigo que el mimicking) del RP v6 REAL
(recalculado por Eclipse) vs el RD de AAA que Pablo acaba de mandar.
Enfoque en cobertura de PTV, que es lo que se ve mal en la captura."""
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import torch
from skimage.draw import polygon as sk_polygon

REPO = Path(r"c:\Pablo\ProstateDoseProject\repo")
sys.path.insert(0, str(REPO))
from pydosert.data.loaders import load_dicom
from src.planning.build_beams import STRUCT_NAMES_DEFAULT, find_dicom_files, patch_rp_missing_ssd
from src.planning.engine_setup import build_dose_engine

DIR_PAC = Path(r"c:\Pablo\ProstateDoseProject\dicom_pilot\PT_003fe2bb84986507")
MIM = Path(r"c:\Pablo\ProstateDoseProject\repo\commissioning\sandbox_results\PT_003fe2bb84986507_mimicking")
RP_V6 = MIM / "RP_v6_eclipse_recalc_ssd.dcm"
RD_V6_AAA = MIM / "RD_v6_eclipse_recalc.dcm"
TMP = Path(r"C:\Users\MEVATE~1\AppData\Local\Temp\claude\c--Pablo-ProstateDoseProject\970561e7-7351-4ce3-9f9f-79b199f3b6ab\scratchpad")
N_FRAC, RX_GY = 39, 78.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK = 8


def cargar(plan_path):
    _, rs, rp, rds = find_dicom_files(DIR_PAC)
    plan = patch_rp_missing_ssd(Path(plan_path), TMP / f"_ssd_{Path(plan_path).stem}.dcm")
    return load_dicom(str(DIR_PAC), dose_path=[str(p) for p in rds], plan_path=[str(plan)],
                      struct_path=str(rs), struct_names=STRUCT_NAMES_DEFAULT,
                      new_spacing=(2.0, 2.0, 2.0), crop_volume=True, device=DEV, dtype=torch.float32)


def leer_rd_en_grilla_2mm(rd_path, ref_shape, ref_res):
    """Lee el RD real (grilla 2.5mm) y lo resamplea a la grilla 2mm de PDRT
    usando el mismo mapeo fisico simple (mismo origen que patient, asumido
    igual FrameOfReference / mismo paciente)."""
    import SimpleITK as sitk
    ds = pydicom.dcmread(str(rd_path))
    arr = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
    px = [float(v) for v in ds.PixelSpacing]
    gfov = np.array([float(v) for v in ds.GridFrameOffsetVector])
    ipp = [float(v) for v in ds.ImagePositionPatient]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((px[1], px[0], float(np.median(np.diff(gfov)))))
    img.SetOrigin((ipp[0], ipp[1], ipp[2] + gfov[0]))
    img.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., 1.))
    return img, ipp, gfov, px


if __name__ == "__main__":
    print("=== Cargando paciente + plan v6 (releido del DICOM real) ===", flush=True)
    patient, arcos = cargar(RP_V6)
    print(f"  grid={tuple(patient.density_image.shape)}", flush=True)
    ptv = patient.structures["PTV_High"].cpu().numpy().astype(bool)
    body = patient.structures["BODY"].cpu().numpy().astype(bool)

    engine = build_dose_engine(patient, beam_template=arcos[0], device=DEV, dtype=torch.float32)
    engine.eval()
    density_image = patient.density_image.to(device=DEV, dtype=torch.float32).unsqueeze(0)

    print("\n=== PDRT forward del v6 (fresco) ===", flush=True)
    tot = None
    t0 = time.time()
    with torch.no_grad():
        for i, bs in enumerate(arcos):
            d = engine.compute_dose(bs, density_image=density_image, beam_chunk_size=CHUNK)
            tot = d if tot is None else tot + d
            print(f"  arco {i}: {time.time()-t0:.0f}s", flush=True)
    dose_pdrt_perfx = tot[0].cpu().numpy()  # Gy/fx
    dose_pdrt_total = dose_pdrt_perfx * N_FRAC  # Gy total del curso

    print("\n=== Leyendo RD de AAA real (Eclipse) del mismo plan v6 ===", flush=True)
    ds_aaa = pydicom.dcmread(str(RD_V6_AAA))
    print(f"  DoseSummationType={ds_aaa.DoseSummationType}  scaling={ds_aaa.DoseGridScaling}")
    dose_aaa_total = ds_aaa.pixel_array.astype(np.float64) * float(ds_aaa.DoseGridScaling)  # Gy TOTAL

    # Resamplear AAA (grilla 2.5mm real) a la grilla 2mm de PDRT via SimpleITK,
    # usando la CT resampleada como referencia geometrica (misma logica que
    # unet_dose_to_pdrt_grid).
    import SimpleITK as sitk
    from pydosert.data.loaders import center_crop_axial, resample_image_to_spacing
    from pydosert.data.utils.dicom_utils import load_ct_series
    ct_series, _ = load_ct_series(str(DIR_PAC))
    ct_ref = resample_image_to_spacing(ct_series, new_spacing=(2.0, 2.0, 2.0), interpolator=sitk.sitkLinear)
    ct_ref = center_crop_axial(ct_ref, max_size_cm=40.0)
    print(f"  ct_ref size={ct_ref.GetSize()}  vs patient grid={tuple(patient.density_image.shape)}")

    rd_ds = pydicom.dcmread(str(RD_V6_AAA))
    px = [float(v) for v in rd_ds.PixelSpacing]
    gfov = np.array([float(v) for v in rd_ds.GridFrameOffsetVector])
    ipp = [float(v) for v in rd_ds.ImagePositionPatient]
    img = sitk.GetImageFromArray(dose_aaa_total.astype(np.float32))
    img.SetSpacing((px[1], px[0], float(np.median(np.diff(gfov)))))
    img.SetOrigin((ipp[0], ipp[1], ipp[2] + gfov[0]))
    img.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., 1.))
    aaa_resampled = sitk.Resample(img, ct_ref, sitk.Transform(), sitk.sitkLinear, 0.0, img.GetPixelID())
    dose_aaa_2mm_total = sitk.GetArrayFromImage(aaa_resampled)  # (z,y,x) Gy total
    print(f"  dose_aaa_2mm_total shape={dose_aaa_2mm_total.shape}")

    rx_frac = RX_GY / N_FRAC

    def resumen(nombre, d_total):
        pct = 100 * d_total / RX_GY
        fuera = body & ~ptv
        print(f"\n--- {nombre} ---")
        for p in (0, 2, 5, 10, 20, 50):
            print(f"    D{100-p:>3d} = {np.percentile(pct[ptv], p):6.1f}%")
        print(f"    media PTV = {pct[ptv].mean():.1f}%   max fuera PTV = {pct[fuera].max():.1f}%")

    resumen("PDRT (fresco, v6 real)", dose_pdrt_total)
    resumen("AAA (Eclipse real, v6, resampleado a grilla PDRT)", dose_aaa_2mm_total)

    diff = dose_pdrt_total - dose_aaa_2mm_total
    print(f"\n=== PDRT - AAA en BODY (Gy) ===")
    print(f"  media {diff[body].mean():+.2f}  p95(|dif|) {np.percentile(np.abs(diff[body]),95):.2f}")
    print(f"  media dentro del PTV {diff[ptv].mean():+.2f}  p95(|dif|) {np.percentile(np.abs(diff[ptv]),95):.2f}")
    print(f"  integral PDRT/AAA en BODY: {dose_pdrt_total[body].sum()/dose_aaa_2mm_total[body].sum():.4f}")

    np.savez_compressed(TMP / "valida_v6.npz", pdrt=dose_pdrt_total.astype(np.float32),
                        aaa=dose_aaa_2mm_total.astype(np.float32), ptv=ptv, body=body)
    print("\nguardado valida_v6.npz")
