"""
Resumen final — diagnostico del hombro medio-bajo de OAR (banda media 40-80%Rx).

Junta los 5 diagnosticos (D1-D5, cada uno ya corrido y guardado en
results/diagnostico_piso/*.json) en un unico veredicto por hipotesis:
  - gap de generalizacion (D1)
  - info 3D que el 2D descarta (D2)
  - piso de CONSISTENCIA de generacion del GT, RapidPlan-normo vs. manual-hipo (D3+D4)
D5 es contexto (no decide, confundido por fraccionamiento/constraints).

Uso:
    .venv/Scripts/python.exe scripts/diagnostico_piso/resumen_final.py
"""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = _REPO_ROOT / "results/diagnostico_piso"


def cargar(nombre):
    with open(OUT_DIR / nombre) as f:
        return json.load(f)


def main():
    d1 = cargar("d1_gap_train_test.json")
    d2 = cargar("d2_coherencia_axial.json")
    d3 = cargar("d3_sonda_vecinos.json")
    d4 = cargar("d4_subset_pareado.json")
    d5 = cargar("d5_banda_media_normo.json")

    # ── D1: gap de generalizacion ──────────────────────────────────────────
    d1_rectum = d1["resultado_por_estructura"]["Rectum"]
    d1_bladder = d1["resultado_por_estructura"]["Bladder"]
    gap_generalizacion = {
        "Rectum": {"ratio_test_vs_train": d1_rectum["ratio_test_vs_train"], "veredicto": d1_rectum["veredicto"]},
        "Bladder": {"ratio_test_vs_train": d1_bladder["ratio_test_vs_train"], "veredicto": d1_bladder["veredicto"]},
    }

    # ── D2: informacion 3D descartada por el 2D ────────────────────────────
    info_3d_descartada = {
        "Rectum": d2["resultados_por_estructura"]["Rectum"]["vfrac"]["veredicto"],
        "Bladder": d2["resultados_por_estructura"]["Bladder"]["vfrac"]["veredicto"],
        "motiva_3d_unet_para_este_hombro": d2["veredicto_D2_motiva_3d_unet_para_hombro_banda_media"],
    }

    # ── D3+D4: piso de consistencia de generacion ──────────────────────────
    piso_generacion = {
        "D3_spread_uno_a_muchos": {
            "spread_normo_pp": d3["spread_normo"], "spread_hipo_pp": d3["spread_hipo"],
            "ratio_hipo_vs_normo": d3["ratio_hipo_vs_normo"],
            "mannwhitney_p": d3["mannwhitney_p_hipo_mayor"],
            "veredicto": d3["veredicto"],
        },
        "D4_subset_pareado_mismo_paciente": {
            "n_pareado": d4["n_pareado"],
            "nota_n": d4["nota_n"],
            "ratio_std_hipo_vs_normo_shoulder": d4["metrica_principal_shoulder_pct_rx"]["ratio_std_hipo_vs_normo"],
            "levene_p_shoulder": d4["metrica_principal_shoulder_pct_rx"]["levene_p"],
            "veredicto_shoulder": d4["veredicto_principal"],
            "ratio_std_hipo_vs_normo_eqd2": d4["metrica_secundaria_eqd2_gy"]["ratio_std_hipo_vs_normo"],
            "levene_p_eqd2": d4["metrica_secundaria_eqd2_gy"]["levene_p"],
            "veredicto_eqd2": d4["veredicto_eqd2"],
        },
        "veredicto_combinado": (
            "CONFIRMADA (con reserva de potencia en D4) — D3 (n=384 normo, n=151 hipo) confirma "
            "con altisima significancia (p=2.8e-30) que el spread del DVH real entre vecinos "
            "anatomicamente casi identicos es 1.6x mayor en hipo (manual) que en normo (RapidPlan). "
            "D4 (n=15, mismo paciente en ambos datasets) va en la MISMA DIRECCION (ratio std "
            f"{d4['metrica_principal_shoulder_pct_rx']['ratio_std_hipo_vs_normo']:.2f}x) pero no "
            "alcanza significancia con esa n tan chica — no contradice a D3, lo corrobora "
            "direccionalmente sin poder confirmarlo solo. El piso del hombro medio-bajo de OAR es, "
            "predominantemente, un piso de CONSISTENCIA DE GENERACION del ground truth (RapidPlan "
            "homogeneo vs. planificacion manual idiosincratica), no un piso de anatomia."
        ),
    }

    # ── D5: contexto ────────────────────────────────────────────────────────
    contexto_normo = {
        "Rectum": d5["comparacion_banda_media_normo_vs_hipo"]["Rectum"],
        "Bladder": d5["comparacion_banda_media_normo_vs_hipo"]["Bladder"],
        "nota": "Contextual, confundido por fraccionamiento/constraints — el hombro banda media "
                "existe TAMBIEN en normo (mismo approach 2D+PSDM+MAE) pero a ~la mitad de magnitud "
                "que en hipo, consistente con (no decisivo por si solo) la lectura de D3/D4: parte "
                "del piso es de dataset/generador, no solo de arquitectura.",
    }

    veredicto_final = {
        "pregunta": "¿Causa del error del hombro medio-bajo de OAR (banda media 40-80%Rx, Rectum/Bladder)?",
        "gap_de_generalizacion_D1": gap_generalizacion,
        "info_3d_descartada_por_2d_D2": info_3d_descartada,
        "piso_de_consistencia_de_generacion_D3_D4": piso_generacion,
        "contexto_normo_D5": contexto_normo,
        "conclusion": (
            "El hombro medio-bajo de OAR NO es principalmente un problema de generalizacion "
            "(D1: Rectum train 6.77pp ~ test 8.36pp, ratio 1.23, dentro del rango 'similar'; "
            "Bladder ratio 1.32, apenas fuera) — mas N/augmentation tendria retorno marginal. "
            "TAMPOCO es informacion 3D que el 2D descarte (D2: el modelo ya seguido la modulacion "
            "axial real de la banda media con rho=0.80-0.94 y sin diferencia de rugosidad Z "
            "significativa en magnitud — un U-Net 3D no esta motivado por ESTE hombro). "
            "La causa dominante, con la evidencia mas fuerte de las 4 vias (D3, n=535 pacientes, "
            "p=2.8e-30), es un PISO DE CONSISTENCIA DE GENERACION: RapidPlan (normo) genera GT "
            "reproducible entre anatomias parecidas; la planificacion manual (hipo) no. D4 "
            "corrobora la misma direccion en el subset pareado (mismo paciente, dos generadores) "
            "aunque sin significancia propia por n=15. D5 (contexto) es consistente: el mismo "
            "approach 2D ya tiene un hombro banda media en normo, pero de la mitad de magnitud. "
            "Implicacion practica: la replanificacion consistente del GT hipo (que Pablo ya "
            "empezo en paralelo) es la palanca con mas potencial para este hombro especifico — "
            "mas que arquitectura 3D o mas datos/augmentation."
        ),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(veredicto_final, f, indent=2)

    print(json.dumps(veredicto_final["conclusion"], indent=2, ensure_ascii=False))
    print(f"\nGuardado: {OUT_DIR / 'summary.json'}")
    return veredicto_final


if __name__ == "__main__":
    main()
