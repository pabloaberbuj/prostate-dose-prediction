# Handoff Claude Code — Pipeline dosis predicha → DVH → ObjectiveTemplate XML

> Bloque para pegar en `CLAUDE_CODE_CONTEXT.md`. Objetivo: cerrar el loop KBP —
> importar el DVH predicho por la U-Net como línea de optimización en Eclipse.
> **Arrancar con 1 paciente de test**, validar end-to-end, después extender a los 59.

## Contexto

- Modelo: `exp002` (`checkpoints/exp002_unet2d_psdm/epoch=191.ckpt`), U-Net 2D + PSDM + MAE.
- Dosis predicha en **relativo %, D95(PTV)=100%**. Rx = **78 Gy** para todos los normo → `Gy = pct × 78/100`.
- Estructuras OAR: `Rectum`, `Bladder` (el mapeo de nombres se puede ajustar en el import de Eclipse).
- El generador de XML ya está escrito y **validado contra un template real de RapidPlan (4249)** e
  importado OK en Eclipse (reconstruye las líneas `Type=1` agrupadas). Ver `scripts/gen_objtemplate.py`
  (copiar `gen_objtemplate.py` de outputs al repo). NO reescribirlo; reusar `build_template(...)`.

## Decodificación del schema ObjectiveTemplate v1.7 (referencia, ya resuelta)

- `Type`: 0=punto, 1=línea, 2=media/gEUD (a confirmar). `Operator`: 0=upper, 1=lower, 99=mean.
- Una **línea** = muchos `<Objective> Type=1` con el mismo `Group` y misma `Priority`. Eclipse los
  reconstruye como una sola línea al importar.
- PTV = puntos fijos (independientes de anatomía): lower Rx×1.01 @100%, upper Rx×1.05 @0%.
- Diferencia conocida vs RapidPlan (NO reproducir): RapidPlan mete un escalón/vertical en la cola
  cerca de Dpres (puntos repetidos a dosis máx con vol→0). Nuestro generador muestrea monótono. Anotar
  como diferencia en la comparación, no corregir.

---

## Paso A — Persistir la dosis predicha (`scripts/evaluate.py`)

Hoy `evaluate.py`/`evaluate_hipo.py` NO guardan `dose_pred` (se calcula en memoria y se descarta).
Agregar flag opcional `--save-pred DIR` (no rompe el uso actual):

- Dentro del loop de evaluación, después de `pred = model(x)`, para cada paciente guardar
  `DIR/pred_<PT>.npz` con:
  - `dose_pred`: array 3D en **relativo %** (mismo espacio que el GT normalizado), shape
    `(N_cortes, 256, 256)`, mismos cortes y orden que el NPZ de entrada.
  - `ptv_mask`, `rectum_mask`, `bladder_mask`: máscaras binarias del mismo NPZ fuente (copiarlas,
    para no depender de re-leer y re-alinear después).
  - `meta`: copiar `meta` del NPZ fuente (por si se necesita trazabilidad).
- Es reuso de tensores ya en memoria + un `np.savez_compressed`. No cambia ninguna métrica.

**Piloto:** correr solo sobre 1 paciente de test (ver "Piloto" abajo) → verificar que el `.npz` sale
bien (shapes, rango de `dose_pred` ~[0, ~110]).

---

## Paso B — DVH desde la predicción (`scripts/compute_pred_dvh.py`, nuevo)

Análogo a `compute_gt_dvh`, pero leyendo `dose_pred` del `.npz` del Paso A. **El DVH en % NO necesita
calibración física de voxel** (el bug de escala de voxel NO aplica: es conteo puro de voxels dentro de
la máscara, no volumen físico).

Pasos por paciente:

1. **Renormalizar a D95(PTV)=100% sobre la predicción** (importante — respeta la condición clínica
   "cobertura de PTV fija"): calcular `d95_pred = percentil(dose_pred[ptv_mask], 5)` (D95 = dosis que
   cubre el 95% del PTV), y `dose_pred_norm = dose_pred / d95_pred * 100`. Así el DVH del OAR queda
   expresado a cobertura de PTV fija, comparable entre pacientes. (El modelo ya predice ~D95=100%, pero
   no exacto; este paso lo fija.)
2. Para cada OAR (`Rectum`, `Bladder`), DVH acumulado sobre los voxels de la máscara:
   ```
   dvox = dose_pred_norm[oar_mask]            # dosis % de cada voxel del OAR
   dose_grid_pct = np.linspace(0, dvox.max(), 200)
   vol_pct = [ (dvox >= d).mean()*100 for d in dose_grid_pct ]
   ```
3. Convertir a Gy: `dose_grid_gy = dose_grid_pct * 78/100`.
4. Guardar/retornar por OAR `(dose_grid_gy, vol_pct)` → es exactamente lo que come `build_template`.

Salida: `DIR/pred_dvh_<PT>.json` con `{"Rectum": {"dose_gy":[...], "vol_pct":[...]}, "Bladder": {...}}`.

---

## Paso C — Generar el XML (`scripts/gen_objtemplate.py`, ya escrito)

```python
from gen_objtemplate import build_template
import json, xml.etree.ElementTree as ET

dvh = json.load(open("pred_dvh_<PT>.json"))
oar = {k: (v["dose_gy"], v["vol_pct"]) for k,v in dvh.items()}
tree = build_template(oar, rx_gy=78, ptv_id="PTV_High",
                      line_priority=50, template_id="UNet_<PT>")
ET.indent(tree, space="")
tree.write("ObjectiveTemplate_UNet_<PT>.xml", encoding="UTF-8", xml_declaration=True)
```

Decisiones fijadas para el primer experimento:
- Línea = DVH predicho **exacto** (sin margen). Perilla de margen queda para después.
- `line_priority = 50` (constante, parámetro).
- PTV: puntos fijos copiados (lower 78×1.01, upper 78×1.05). No tocar.

---

## Piloto — 1 paciente

1. Elegir 1 paciente del **test set normo** (`data/splits/splits_v1.json` → lista `test`).
   Candidato preferido: **GARCIA** si está en test (Pablo planificará este caso en Eclipse para
   cerrar el loop). Si no está, cualquier caso de test sirve para el primer end-to-end.
2. Correr Paso A solo para ese PT → `pred_<PT>.npz`.
3. Paso B → `pred_dvh_<PT>.json`.
4. Paso C → `ObjectiveTemplate_UNet_<PT>.xml`.
5. **Pablo importa el XML en Eclipse** sobre ese paciente y verifica que las líneas Recto/Vejiga se
   reconstruyen y son plausibles (deberían parecerse al DVH real del plan aprobado, no al sintético
   de la prueba anterior).

Entregar a Claude.ai: el `.json` del DVH predicho + el XML + (si es fácil) un PNG del DVH predicho vs
DVH real del plan aprobado de ese paciente, para chequear que la predicción tiene sentido antes de
escalar a 59.

---

## Después del piloto (no ahora — decidir con Claude.ai)

- Extender a los 59 de test (batch de Paso A/B/C).
- Definir la evaluación del loop cerrado: **DVH logrado (plan optimizado con la línea U-Net) vs DVH
  predicho vs DVH del plan clínico original vs RapidPlan**. Métrica: acuerdo de DVH curva-completa
  (dose/DVH score OpenKBP). Esto es el head-to-head del Kickoff 2 (niveles A/B).
- Perilla de margen sobre la línea (borde agresivo estilo RapidPlan vs predicción exacta).
