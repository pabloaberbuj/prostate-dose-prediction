---
tags: [prostatedoseproject, aprendizajes, hipotesis-descartadas]
estado: cerrado
---

# Hipótesis descartadas

Ideas que se investigaron en serio, con evidencia experimental dedicada, y se descartaron explícitamente. Se documentan para no volver a gastar tiempo re-investigándolas sin nueva evidencia.

## "Techo de geometría de arcos" (arc ceiling) — DEBUNKED

Se sospechó que existía un límite físico/geométrico en la calidad alcanzable por arcos VMAT (relacionado con ambigüedad angular). La investigación (análisis de TERMA por prior angular) mostró que lo que se interpretaba como techo era en realidad **contaminación nodal** en el dataset — no un límite real de la geometría de entrega. Condicionalmente podría revisitarse si `exp007`/la loss DVH no resuelven la banda media de Rectum/Bladder, pero no está activo hoy.

## kernel_size=55 (recomendado por el paper de PDRT) — sin ganancia

El paper de referencia de PyDoseRT recomienda `kernel_size=55` como el de mejor precisión. Probado en esta comisión: empeoró el acuerdo en PTV/high-dose y resultó ~3x más lento. Se mantiene `kernel_size=25`. Ver [[../subproyectos/06_comisionamiento_pdrt]].

## "CT real no aporta señal" — confirmado 3 veces independientes

No es exactamente una hipótesis descartada sino una hipótesis **confirmada repetidamente** pese a expectativa inicial de que aportaría información: con FOV 50cm, con FOV 34cm, y de nuevo en hipo v4, el CT real (por sobre PSDM+máscaras) no mejora la predicción de dosis. Sí acelera convergencia (~1.8x). Ver [[bugs_criticos_resueltos]] (bug de CT) y [[../subproyectos/01_unet_kbp_normofraccionado]].

## Piso de OAR ("hombro" de la curva) — no es un límite de arquitectura

El diagnóstico D1–D5 sobre el comportamiento en "hombro" de Rectum/Bladder (donde el modelo deja de mejorar) se sospechó primero como límite de información/arquitectura. El veredicto final: es un **piso de consistencia de generación del ground truth** (RapidPlan homogéneo en normo vs. planificación manual idiosincrática en hipo), no un límite físico ni arquitectónico. Confirmado también con `exp_normo_3dunet`: pasar a 3D no mejora sobre 2D — el piso persiste igual con arquitectura 3D y GT consistente.

## U-Net vs RapidPlan como predictor — sin cambios tras fix de CT

Comparación U-Net vs. predictor RapidPlan (n=59) re-corrida sobre `ctfix_fov34`: resultado sin cambios respecto al pre-fix (Rectum ~4.32pp, Bladder ~1.63pp de sesgo medio), con dirección del sesgo variable por paciente. No se considera una hipótesis "descartada" sino un resultado estable que no dependía del bug de CT.
