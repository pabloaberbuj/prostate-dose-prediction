"""Proyeccion BEV del PTV calculada DESDE CERO (geometria IEC pura, sin usar
nada de PyDoseRT ni de ptv_conforming_init), para verificar de forma
independiente si las aperturas del RP v6 cubren el PTV / abren de mas.
"""
import glob
import itertools

import numpy as np
import pydicom

DIR = r"c:\Pablo\ProstateDoseProject\dicom_pilot\PT_003fe2bb84986507"
RP_V6 = r"c:\Pablo\ProstateDoseProject\repo\commissioning\sandbox_results\PT_003fe2bb84986507_mimicking\RP_v6_dvh.dcm"
SAD = 1000.0


def ptv_puntos(rs_path, nombre="PTV_High"):
    rs = pydicom.dcmread(rs_path)
    num = {int(r.ROINumber): str(r.ROIName) for r in rs.StructureSetROISequence}
    for roi in rs.ROIContourSequence:
        if num.get(int(roi.ReferencedROINumber)) != nombre:
            continue
        pts = [np.array(c.ContourData, float).reshape(-1, 3) for c in roi.ContourSequence]
        return np.vstack(pts)
    raise KeyError(nombre)


def leer_rp(rp_path):
    ds = pydicom.dcmread(rp_path)
    arcos, bounds = [], None
    for beam in ds.BeamSequence:
        for d in getattr(beam, "BeamLimitingDeviceSequence", []):
            if d.RTBeamLimitingDeviceType == "MLCX" and bounds is None:
                bounds = np.array(d.LeafPositionBoundaries, float)
        cps = beam.ControlPointSequence
        if not any(any(d.RTBeamLimitingDeviceType == "MLCX"
                       for d in getattr(cp, "BeamLimitingDevicePositionSequence", [])) for cp in cps):
            continue
        L, G, iso, coll = [], [], None, None
        ul = None
        for cp in cps:
            for d in getattr(cp, "BeamLimitingDevicePositionSequence", []):
                if d.RTBeamLimitingDeviceType == "MLCX":
                    ul = np.array(d.LeafJawPositions, float)
            L.append(ul.copy())
            G.append(float(cp.GantryAngle) if hasattr(cp, "GantryAngle") else G[-1])
            if iso is None and hasattr(cp, "IsocenterPosition"):
                iso = np.array(cp.IsocenterPosition, float)
            if coll is None and hasattr(cp, "BeamLimitingDeviceAngle"):
                coll = float(cp.BeamLimitingDeviceAngle)
        arcos.append(dict(leaf=np.array(L), gantry=np.array(G), iso=iso, coll=coll))
    return arcos, bounds


def proyectar(pts, iso, gantry_deg, coll_deg, signo_gantry, signo_coll):
    g = np.deg2rad(gantry_deg) * signo_gantry
    eX = np.array([np.cos(g), np.sin(g), 0.0])
    eY = np.array([0.0, 0.0, 1.0])
    eZ = np.array([-np.sin(g), np.cos(g), 0.0])
    c = np.deg2rad(coll_deg) * signo_coll
    aX = np.cos(c) * eX + np.sin(c) * eY
    aY = -np.sin(c) * eX + np.cos(c) * eY
    src = iso - SAD * eZ
    v = pts - src
    t = v @ eZ
    u = (v @ aX) * SAD / t
    w = (v @ aY) * SAD / t
    return u, w


def cobertura(arco, bounds, pts, sg, sc):
    lo, hi = bounds[:-1], bounds[1:]
    n = len(lo)
    L = arco["leaf"]
    b1, b2 = L[:, :n], L[:, n:]
    cubre, fuera = [], []
    paso = max(1, len(L) // 30)
    for k in range(0, len(L), paso):
        u, w = proyectar(pts, arco["iso"], arco["gantry"][k], arco["coll"], sg, sc)
        a_tot = a_fuera = 0.0
        for j in range(n):
            m = (w >= lo[j]) & (w < hi[j])
            gap = b2[k, j] - b1[k, j]
            if not m.any():
                a_tot += max(gap, 0) * (hi[j] - lo[j])
                a_fuera += max(gap, 0) * (hi[j] - lo[j])
                continue
            umin, umax = u[m].min(), u[m].max()
            ancho_ptv = umax - umin
            inter = max(0.0, min(umax, b2[k, j]) - max(umin, b1[k, j]))
            a_tot += max(gap, 0) * (hi[j] - lo[j])
            a_fuera += (max(gap, 0) - inter) * (hi[j] - lo[j])
            cubre.append(inter / max(ancho_ptv, 1e-6))
        fuera.append(a_fuera / max(a_tot, 1e-6))
    return float(np.mean(cubre)), float(np.mean(fuera))


if __name__ == "__main__":
    rs = glob.glob(DIR + r"\RS*.dcm")[0]
    rp = glob.glob(DIR + r"\RP.*.dcm")[0]
    pts = ptv_puntos(rs)

    arcos_c, bounds = leer_rp(rp)
    sg, sc = 1, 1  # convencion ya validada contra el plan clinico en sesiones anteriores

    arcos_v6, _ = leer_rp(RP_V6)
    print("--- cobertura del PTV y area abierta fuera del PTV ---")
    for nombre, arcos in [("clinico", arcos_c), ("v6", arcos_v6)]:
        for i, a in enumerate(arcos):
            cub, fue = cobertura(a, bounds, pts, sg, sc)
            print(f"  {nombre:10s} arco {i}: cobertura del PTV {cub*100:5.1f}%   area abierta fuera del PTV {fue*100:5.1f}%")
