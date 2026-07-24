"""
Debug: inspecciona la geometría de CT y RD de un paciente.
Correr antes de preprocess.py para entender el error de dimensiones.

Uso:
    python data/debug_geometry.py --patient-dir C:/ruta/a/PT_0a3eac085fccff90
"""

import argparse
from pathlib import Path
import pydicom
import SimpleITK as sitk
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-dir", required=True)
    args = parser.parse_args()

    carpeta = Path(args.patient_dir)

    # ── Listar todos los DICOMs y sus modalidades ─────────────────────────────
    print("=== Archivos DICOM en la carpeta ===")
    for f in sorted(carpeta.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            mod = getattr(ds, 'Modality', '?')
            sop = str(getattr(ds, 'SOPClassUID', '?'))
            desc = getattr(ds, 'SeriesDescription', '')
            print(f"  {f.name:<40} Modality={mod}  SOP={sop[-10:]}  Desc={desc}")
        except Exception as e:
            print(f"  {f.name}: ERROR {e}")

    # ── CT via SimpleITK ──────────────────────────────────────────────────────
    print("\n=== CT (via SimpleITK ImageSeriesReader) ===")
    try:
        reader = sitk.ImageSeriesReader()
        archivos_ct = reader.GetGDCMSeriesFileNames(str(carpeta))
        print(f"Archivos CT encontrados: {len(archivos_ct)}")
        if archivos_ct:
            reader.SetFileNames(archivos_ct)
            ct = reader.Execute()
            print(f"  Size:      {ct.GetSize()}")
            print(f"  Spacing:   {ct.GetSpacing()}")
            print(f"  Origin:    {ct.GetOrigin()}")
            print(f"  Direction: {ct.GetDirection()}")
            print(f"  Direction len: {len(ct.GetDirection())}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── RD via SimpleITK ──────────────────────────────────────────────────────
    print("\n=== RD (via SimpleITK ReadImage) ===")
    for f in sorted(carpeta.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            modality = str(getattr(ds, 'Modality', ''))
            sop = str(getattr(ds, 'SOPClassUID', ''))
            if modality == 'RTDOSE' or '481.2' in sop:
                print(f"  Archivo RD: {f.name}")
                rd = sitk.ReadImage(str(f))
                print(f"  Size:      {rd.GetSize()}")
                print(f"  Spacing:   {rd.GetSpacing()}")
                print(f"  Origin:    {rd.GetOrigin()}")
                print(f"  Direction: {rd.GetDirection()}")
                print(f"  Direction len: {len(rd.GetDirection())}")

                # Tags DICOM clave
                print(f"  DoseGridScaling: {getattr(ds, 'DoseGridScaling', '?')}")
                if hasattr(ds, 'GridFrameOffsetVector'):
                    print(f"  GridFrameOffsetVector (primeros 3): {list(ds.GridFrameOffsetVector[:3])}")
                if hasattr(ds, 'ImageOrientationPatient'):
                    print(f"  ImageOrientationPatient: {list(ds.ImageOrientationPatient)}")
        except Exception as e:
            print(f"  {f.name}: ERROR {e}")

    # ── Intentar Resample manual para ver dónde explota ───────────────────────
    print("\n=== Test Resample CT → RD ===")
    try:
        reader = sitk.ImageSeriesReader()
        archivos_ct = reader.GetGDCMSeriesFileNames(str(carpeta))
        reader.SetFileNames(archivos_ct)
        ct = reader.Execute()

        for f in sorted(carpeta.glob("*.dcm")):
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if str(getattr(ds, 'Modality', '')) == 'RTDOSE':
                rd = sitk.ReadImage(str(f))
                print(f"CT direction len: {len(ct.GetDirection())}")
                print(f"RD direction len: {len(rd.GetDirection())}")

                # Intentar resample
                try:
                    resampled = sitk.Resample(rd, ct, sitk.Transform(),
                                              sitk.sitkLinear, 0.0, rd.GetPixelID())
                    print("  Resample OK")
                except Exception as e2:
                    print(f"  Resample FALLÓ: {e2}")
                break
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
