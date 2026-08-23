# Handoff Claude Code — Subestructuras por dosis predicha (dose mimicking en Eclipse)

> Bloque para `CLAUDE_CODE_CONTEXT.md`. Objetivo: evaluar si subdividir el OAR en regiones
> según la dosis que predice la U-Net, y optimizar con sub-líneas DVH por región, produce un
> plan distinto (mejor) que la línea de estructura completa. Es "dose mimicking del pobre":
> captura parte de la información espacial del mapa 3D, pero 100% dentro de Eclipse (solo RS
> modificado + XML, sin ESAPI de escritura, sin PDRT).

## Hipótesis y diseño experimental

La línea DVH de estructura completa es espacialmente ambigua → el optimizador alcanza ese DVH
con muchas configuraciones, algunas más moduladas que otras. Fijar subestructuras por dosis
fuerza la solución espacial de la U-Net. Apuesta: alcanzable con **menor complejidad/tiempo a
igual calidad**. Efecto esperado más notorio en **recto** y en **planes complejos**. Dirección
de la complejidad: NO asumida (puede subir o bajar), es empírico.

**Tres brazos, todos sobre el mismo paciente:**
1. **Control** — línea de recto completo, derivada de `dose_pred`. (Ya lo produce el pipeline KBP.)
2. **Test** — sub-líneas por subestructura, del *mismo* `dose_pred`.
3. **RapidPlan estándar** — referencia externa (no control directo).

Única variable control vs test = granularidad espacial. Misma prioridad, mismo PTV, mismo todo.

**Métricas** (las 3 de complejidad ya están implementadas en otra rama — reusar):
- Calidad: DVH logrado del recto completo (incluir aunque en próstata suele saturar en la línea).
- Complejidad: MCS, MU/Gy, SAS.
- Tiempo a convergencia de la optimización.
- (Más adelante, si hay efecto: gamma QA-PE medido.)

## Alcance del primer test

- Solo **recto**. PTV y vejiga afuera por ahora (subdividir PTV pelea con cobertura uniforme).
- **2 subestructuras** mutuamente excluyentes: `Rectum_hot` (recto ∩ dosis_pred ≥ q) y
  `Rectum_cold` (el resto). Umbral inicial: dosis ≥ **85% Rx** (donde viven V70/V65). Dejar el
  umbral parametrizable; probar también cuantil (p.ej. percentil 66 de la dosis dentro del recto)
  y comparar cuál da máscaras más estables.
- 1 paciente **complejo** primero (Pablo elige uno de test normo con plan modulado).

---

## Bloque 0 — De-risk aislado: máscara → RS → import Eclipse

**Hacer esto ANTES de construir nada más.** Es la única pieza nueva y la de más riesgo.

- Tomar 1 paciente: su CT DICOM (serie) + RS original.
- Generar una máscara dummy trivial (p.ej. recto erosionado 3 mm, o recto ∩ isodosis alta) en
  **geometría nativa de la CT**.
- Con `rt-utils`: `RTStructBuilder.create_from(ct_dir, rs_path)` → `add_roi(mask=..., name="TEST_ROI")`
  → `save()`. Verificar convención de ejes de la máscara (height, width, slices) contra la serie.
- **Pablo importa el RS modificado en Eclipse** y confirma: (a) el ROI nuevo aparece, (b) está
  **espacialmente alineado** con la anatomía (no corrido/rotado), (c) `FrameOfReferenceUID` y
  posiciones coinciden.

Si esto falla, todo lo demás no sirve → resolver alineación acá antes de seguir.

---

## Bloque 1 — Split de dosis predicha en submáscaras (`split_oar_by_dose.py`, nuevo)

Requiere `dose_pred` persistido (pipeline KBP, flag `--save-pred`).

1. **Upsample** `dose_pred` 256→nativo in-plane (bilineal). Mantener spacing de cortes nativo.
2. **Suavizado** gaussiano de `dose_pred` antes de umbralar (sigma ~2 mm equivalente). Evita que
   el artefacto radial/cuadriculado se vuelva borde de estructura.
3. **Renormalizar** a D95(PTV)=100% sobre la predicción (igual que en `compute_pred_dvh.py`).
4. Cargar máscara **nativa** del recto (del RS original, NO la de 256×256).
5. `Rectum_hot = rectum_mask & (dose_pred_norm >= umbral)`; `Rectum_cold = rectum_mask & ~hot`.
6. **Limpieza morfológica**: closing 3D + remover componentes conexas < N voxels (parámetro).
   Chequear que ninguna submáscara quede vacía o con < ~50 voxels (si pasa, bajar umbral / avisar).
7. Guardar submáscaras nativas → `results/substruct/<PT>/masks.npz`.

Spot-check: reportar volumen (cc) de cada submáscara y que `hot ∪ cold == rectum` (partición exacta).

---

## Bloque 2 — Submáscaras → RS modificado (`write_substruct_rs.py`, nuevo)

- `rt-utils` sobre copia del RS original: agregar `Rectum_hot` y `Rectum_cold` como ROIs nuevas.
- NO borrar el `Rectum` original (se necesita para el DVH de calidad global).
- Guardar `results/substruct/<PT>/RS_substruct.dcm`.

---

## Bloque 3 — Sub-líneas DVH → XML (`gen_objtemplate.py`, ya existe — extender wrapper)

- DVH de cada submáscara desde `dose_pred_norm` (reusar lógica de `compute_pred_dvh.py`, pero
  contando voxels dentro de `Rectum_hot` y `Rectum_cold` por separado).
- `build_template({"Rectum_hot":(d,v), "Rectum_cold":(d,v), ...}, rx_gy=78, line_priority=50)`.
- **Brazo control** en paralelo: `build_template({"Rectum":(d,v)})` (línea completa, mismo dose_pred).
- PTV: puntos fijos, idénticos en ambos brazos.
- Salidas: `ObjectiveTemplate_test_<PT>.xml` (sub-líneas) y `ObjectiveTemplate_control_<PT>.xml`.

---

## Piloto — 1 paciente complejo

1. Bloques 1–3 → RS_substruct + XML test + XML control.
2. **Pablo en Eclipse**: sobre el mismo paciente, correr 3 optimizaciones (test / control /
   RapidPlan estándar), mismos beams/grilla/settings. Registrar tiempo a convergencia.
3. Exportar los 3 planes → calcular las 3 métricas de complejidad + DVH logrado del recto completo.
4. Traer a Claude.ai: tabla comparativa (calidad, MCS, MU/Gy, SAS, tiempo) de los 3 brazos.

Si test ≈ control en calidad pero difiere en complejidad/tiempo → hay señal espacial. Recién ahí
escalar a más pacientes y considerar 3 subestructuras / "empujar las curvas".

---

## Riesgos / notas

- **Alineación de grilla** (bloque 0): el mayor riesgo. Nativo, no 256×256, para el RS.
- **Ruido de predicción baked-in**: suavizar + cuantil + limpieza morfológica, o se contornea el artefacto.
- **rt-utils**: verificar orden de ejes de la máscara y que Eclipse 13.6 acepta el RS re-escrito
  (structure set con ROIs agregadas por librería externa — confirmar que no rompe tags que Eclipse exige).
- **Interpretación**: sin la métrica de calidad no se interpreta la complejidad. El resultado
  interesante es un trade-off, no una métrica sola.
- **Prior art** (verificar, no citado con certeza): "dose mimicking structure-based", "tuning
  structures automated planning". Claude.ai no tiene acceso a base de datos — Pablo confirma refs.
