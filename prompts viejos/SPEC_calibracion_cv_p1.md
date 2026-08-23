# Spec Claude Code — Calibración por CV + IC del punto de operación (Proyecto 1)

## Contexto del problema

En la evaluación v3, la frontera verde de Recto cayó de sens=0.89 (val) a 0.50 (test) pese a
AUC=0.921. Diagnóstico: NO es un modelo malo (el AUC alto confirma buen ordenamiento) — es un
**umbral estimado con varianza enorme**. Val tiene solo 9 positivos de recto → la sensibilidad es
una función escalón grosera (cada positivo ≈ 11pp) y el corte lo definen 1-2 pacientes.

Esta spec ataca la varianza del umbral **sin tocar test ni el modelo**. Dos cambios:
1. Calibrar el umbral con CV estratificado sobre train+val agrupados (no sobre val fijo).
2. Reportar IC bootstrap del punto de operación (honestidad, no maquillaje).

**Regla dura que no se rompe:** test sigue siendo el único holdout. NUNCA se usa test para elegir
umbrales. Los UnApproved (sintéticos) NUNCA entran a la calibración del punto de operación — su
prevalencia (60-70% falla) sesgaría el umbral. Ver más abajo.

---

## Cambio 1 — Calibración de umbrales por CV estratificado

Modificar `scripts/calibrate_p1.py`.

### Pool de calibración
- Usar **población natural de train + val** (Approved + Rejected de ambas particiones), agrupada.
- **Excluir los UnApproved** del pool de calibración de umbrales. Sí siguen en train para *fitear el
  modelo*, pero el *umbral* se calibra solo sobre prevalencia natural (si no, el corte se elige contra
  una prevalencia de falla irreal y no sirve clínicamente).
- Test queda intacto y aparte.

### Procedimiento
Para cada constraint (RV65, RV55, BV65 — recordar que la frontera de Recto combina RV65+RV55 pesimista):

1. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre el pool natural.
2. En cada fold:
   - Fitear scaler + modelo (LogReg para clasificación; Ridge para regresión de severidad) SOLO en el
     train del fold. Ojo: esto re-fitea por fold para no filtrar; es correcto porque acá estamos
     estimando el *umbral*, no el modelo final de despliegue.
   - En el val del fold, encontrar el umbral que alcanza el objetivo de sensibilidad de la frontera
     verde (recto sens≈0.85, vejiga sens≈0.90).
3. **Umbral final = mediana (no media) de los umbrales por fold.** La mediana es robusta a folds
   degenerados (algún fold puede quedar sin positivos y dar un umbral extremo).
4. Reportar la dispersión de umbrales entre folds (min, max, IQR) — es el diagnóstico directo de cuán
   inestable es la calibración para ese constraint.

### Nota importante sobre el modelo de despliegue
El modelo que se **despliega** se sigue fiteando una sola vez sobre TODO el train (natural + UnApproved),
como en v3. El CV es SOLO para elegir el umbral. No confundir: CV → umbral; fit único en train completo
→ pesos del modelo desplegado. El umbral mediano del CV se aplica sobre las probabilidades de ese modelo
único.

---

## Cambio 2 — Frontera verde desde la regresión (probar como alternativa)

Dado que la regresión Ridge reportó correlación cerca-del-umbral 0.83-0.98 (señal continua bien
comportada justo donde el clasificador es un escalón grosero), probar una **segunda variante** de la
frontera verde/no-verde:

- Variante A (actual): umbral sobre la probabilidad del clasificador.
- Variante B (nueva): umbral sobre el **valor predicho por la regresión** (margen al constraint).

Calibrar ambas por CV (Cambio 1) y comparar en test cuál da sensibilidad más estable (val→test). NO
elegir la ganadora mirando test — reportar ambas y que Pablo/Claude.ai decidan con el IC en mano.
Para recto, la hipótesis es que B es más estable. Dejar ambas en el manifest, marcada cuál es la activa.

---

## Cambio 3 — IC bootstrap del punto de operación

Modificar `scripts/eval_p1.py`.

- Con el umbral congelado (mediana del CV), evaluar en test.
- **Bootstrap 1000 (seed=42) sobre el test set** para el IC95 de sens y esp EN EL PUNTO DE OPERACIÓN
  (no solo del AUC — eso ya estaba). Esto muestra que "sens=0.50" viene con un IC ancho y contextualiza
  la caída.
- **Además**, bootstrap sobre la *selección del umbral*: remuestrear el pool de calibración, recalcular
  el umbral, y ver el IC del umbral mismo. Esto cuantifica el "el 0.89 nunca tuvo esa precisión".
- Reportar en `results/proyecto1_v3_cv/metrics_summary.json`: por constraint, AUC [IC95], umbral
  [IC95 e IQR entre folds], sens/esp en test [IC95 bootstrap], variante de frontera (A prob / B reg).

---

## Cambio 4 — Objetivo de sensibilidad conservador para recto (band-aid documentado)

Dado el costo asimétrico (FN de recto = error caro), calibrar la frontera verde de RECTO a un objetivo
más alto (sens=0.95 en el CV) sabiendo que degradará algo en test, para que aterrice en un valor
clínicamente aceptable. Vejiga se mantiene en 0.90. Documentar explícitamente que esto es una elección
conservadora deliberada por costo asimétrico, no un ajuste contra test.

---

## Qué NO hacer

- No recalibrar contra test (rompe el holdout).
- No meter UnApproved en la calibración del umbral (sesga por prevalencia).
- No elegir variante A vs B mirando el resultado de test (se reportan ambas, decide Claude.ai).
- No cambiar los pesos del modelo desplegado (se sigue fiteando en train completo una sola vez).

---

## Entregable de vuelta a Claude.ai

`results/proyecto1_v3_cv/metrics_summary.json` con, por constraint:
- AUC [IC95].
- Umbral final (mediana CV) + IQR entre folds + IC95 bootstrap del umbral.
- Sens/esp en test [IC95 bootstrap] en el punto de operación.
- Comparación variante A (prob) vs B (regresión) para la frontera verde.
- % de casos en cada zona (verde/naranja/rojo) en test.

Con eso se decide en Claude.ai el punto de operación final y si la frontera de recto usa prob o regresión.
