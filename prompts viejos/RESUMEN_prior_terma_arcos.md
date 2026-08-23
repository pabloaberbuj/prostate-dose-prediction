# Prior de geometría de arcos (TERMA) — resumen de la discusión

> Compilación de lo discutido en el proyecto de predicción de dosis 3D (próstata VMAT,
> normo + hipofraccionado). Sirve como documento de referencia autocontenido para retomar
> el tema sin releer todo el historial.

---

## Qué es y por qué se propuso

**TERMA** (*Total Energy Released per unit MAss*) es la magnitud dosimétrica que describe la
energía liberada por el haz antes de transportarse por scatter — es el input físico de los
algoritmos de dosis tipo convolución-superposición (y conceptualmente relacionado a cómo AAA
calcula dosis). La idea de un **"arc-prior TERMA-style"** es agregar un canal de input a la
U-Net que codifique **geometría de entrega real** (qué tanto fluence deposita cada control
point del arco VMAT sobre cada vóxel, aproximando TERMA), en vez de que el modelo infiera la
dosis solo desde la anatomía (CT + máscaras + PSDM).

**Motivación original:** en la serie normofraccionado, los mapas de diferencia (predicho −
real) de exp002 mostraban un **artefacto radial en forma de estrella** en la zona de dosis
media-baja — atribuido a que el modelo no tiene información de la modulación angular real
de los arcos (dose rate por control point), solo anatomía. Es un patrón consistente con
"faltante de información de entrega", no con ruido de entrenamiento.

---

## Cronología de la discusión

### 1. Serie de ablaciones normofraccionado (cerrada)
- **PSDM > masks** (exp002 gana a exp001): decisión robusta, no cuestionada en ningún punto
  posterior.
- **2.5D (exp004/005, 3 y 5 cortes de contexto axial): sin mejora.** El modelo usa esos
  canales (pesos no nulos) pero no aportan señal — la anatomía de cortes vecinos no informa
  la geometría angular del arco.
- **MomentLoss (exp006): sin mejora neta.** MAE empeora monótono con λ; DVH score plano. Se
  interpretó como mismatch entre momentos de potencia (M1/M2/M10) y los percentiles de DVH
  (D2/D95/D99) que gobiernan los constraints clínicos — **no** como evidencia de que "modelar
  el DVH en la loss" no sirva; el problema fue la elección de *qué* de la forma del DVH
  modelar.
- **Conclusión de la serie:** el artefacto radial es un techo estructural del enfoque
  anatomía-only. Ni 2.5D ni MomentLoss lo resuelven porque atacan ejes ortogonales al
  problema (contexto axial anatómico; forma global de la distribución). Se necesitaría un
  input de geometría de entrega (arc-prior TERMA) o un encoder polar **combinado con** ese
  input — ninguno de los dos sirve solo.
- **Decisión:** pausar arquitecturas nuevas (cascada, dual-encoder, polar) y el arc-prior
  hasta la etapa de largo plazo, porque bajo el objetivo de **clasificación de
  cumplimiento**, el artefacto queda diluido por la integración del DVH y casi no afecta el
  AUC/sensibilidad — no había urgencia para justificar el esfuerzo.

### 2. Etapa hipofraccionado — falsa alarma del "techo de arcos" (superada)
- 4 pacientes fallaban catastróficamente (MAE body 16-22%) en las tres corridas
  independientes (zero-shot, baseline desde cero, finetune), sin excepción, y sin firma
  anatómica escalar distintiva (volúmenes/overlap/MU dentro de 1 std del resto). Lo único
  que compartían: **NArcos=3**. Esto se leyó como confirmación fuerte del techo de arcos y
  le dio urgencia renovada al arc-prior.
- **Reversión:** se descubrió que los 4 casos (y otros 24 más) tenían **irradiación
  ganglionar pélvica** — planes con geometría de tratamiento completamente distinta a
  próstata-sola, contaminando el dataset. Tras excluir los 28 casos, los "arc-limited"
  desaparecen casi del todo (4→0/1/0), y el único remanente es de **2 arcos**, no 3 —
  contradice directamente la hipótesis. **El techo de arcos queda descartado**: era
  correlación espuria por confusor no controlado (NArcos correlacionaba con "tiene
  ganglios", no con dificultad de modulación).
- Consecuencia directa: el arc-prior perdió su evidencia empírica más fuerte (los 4 casos
  "sin explicación anatómica" ya tienen explicación, y no es angular).

### 3. Baseline ML clásico — resta motivación adicional
- Un baseline de regresión logística / gradient boosting sobre 7 features geométricas
  escalares (overlap PTV-OAR, volúmenes) **igualó o superó a la U-Net en AUC de
  clasificación de constraints** (RV65 0.98 vs 0.92; RV55 0.92 vs 0.84; BV65 0.94 vs 0.93).
- Si el arc-prior se justificaba en parte por mejorar clasificación, esa vía ya no compite:
  el clásico gana esa pelea con información mucho más simple.
- **Reordenamiento del proyecto:** se separó en Proyecto 1 (herramienta de tomógrafo →
  camino ML clásico, sin U-Net) y Proyecto 2 (KBP / dosis 3D completa → camino U-Net, con
  métrica de éxito distinta a la clasificación).

### 4. Revisión reciente bajo el objetivo KBP (el giro más importante)
- Con el objetivo redefinido a **KBP / predicción de dosis 3D completa** (no clasificación),
  el artefacto radial vuelve a importar — y esta vez de forma más directa que antes: es
  sesgo estructural en el mapa que se usaría como **target de optimización** de un
  planificador automático. Un optimizador alimentado con un target que tiene textura radial
  espuria podría empujar la fluencia hacia un patrón sin sentido físico real.
- Bajo clasificación, el artefacto se diluía porque el DVH integra sobre todo el volumen
  (por eso el AUC salía bien pese al artefacto visible). Bajo KBP, no hay esa dilución.
- **Matiz sobre 2.5D:** la conclusión "no ayuda" se sostuvo con métricas agregadas por
  slice (MAE, DVH score), que no verifican explícitamente si el volumen 3D reconstruido
  (apilando slices 2D) es **coherente entre cortes vecinos** — algo que para clasificación
  no importa (el ruido inter-slice se promedia en el DVH) pero para KBP sí (el target de
  optimización necesita continuidad axial). Esto nunca se chequeó directamente.

---

## Discusiones saldadas

1. **PSDM sobre masks binarias:** decisión cerrada, no cuestionada en ningún punto.
2. **2.5D no resuelve el artefacto radial** (aunque sí puede aportar algo distinto a
   "resolver el artefacto" — ver pendiente sobre coherencia inter-slice).
3. **MomentLoss (M1/M2/M10) no es la forma correcta de meter DVH en la loss** — el mismatch
   es con los percentiles, no con la idea de usar el DVH en sí.
4. **El "techo de geometría de arcos" está descartado como límite físico** — era
   contaminación de dataset (irradiación nodal), no falta de información angular.
5. **El arc-prior no se justifica por clasificación de constraints** — ahí el ML clásico ya
   gana con información mucho más simple; cualquier justificación de TERMA tiene que venir
   de mejorar el DVH/dosis 3D completo, no del AUC de un flag binario.
6. **La motivación original del artefacto radial (visual, en mapas de diferencia) sigue
   vigente como observación** — no fue refutada, solo se refutó la explicación de "techo de
   arcos" ligada a los 4 casos hipo. El artefacto en sí (visible en exp002 normo e hipo) no
   se negó en ningún momento.

---

## Pendiente / abierto

1. **Cuantificar el artefacto radial con una métrica propia**, no solo "se ve en el mapa de
   diferencia". Propuesta ya identificada: métrica espectral angular — FFT sobre θ a radio
   fijo alrededor del isocentro, buscando picos de frecuencia angular consistentes con el
   patrón de control points. Sin esto, no hay forma objetiva de decidir si el artefacto es
   grande o chico, ni de medir si una intervención lo reduce.
2. **Métrica de DVH-curva-completa** (no solo los 3 puntos de constraints operativos) como
   vara de comparación relevante para KBP, antes de comparar contra RapidPlan.
3. **Verificar coherencia axial del volumen reconstruido** de exp002 (perfiles de dosis en
   función de z a XY fijo) — chequeo barato para saber si 2.5D merece revisarse bajo el
   lente KBP, o si el problema es genuinamente angular (2D) y no de contexto axial.
4. **Con (1) y (2) medidos sobre exp002 actual**, decidir si el artefacto justifica retomar
   arquitecturas pausadas (cascada, dual-encoder, polar) o arc-prior TERMA, o si alcanza con
   post-procesamiento/suavizado del mapa predicho.
5. **Loss híbrida dosis + DVH — YA HAY REFERENCIA CONCRETA (ver sección nueva abajo).**
   Compartido por Pablo: Nguyen et al. 2020, *Med Phys*. Resuelve exactamente el mismatch que
   hizo fallar a MomentLoss (exp006): en vez de momentos de potencia, usa una **aproximación
   diferenciable del DVH real**. Ver detalle técnico en "DVH loss diferenciable (Nguyen et
   al. 2020)" más abajo — es el candidato concreto a implementar, no solo una idea.
6. **Tensión de inferencia sin resolver** (independiente de todo lo anterior): en tomógrafo
   no se conoce cuántos arcos va a usar el plan final. Un arc-prior TERMA solo es utilizable
   en evaluación retrospectiva (con el plan ya hecho) o si se **asume/condiciona** una
   técnica estándar de arcos — el centro no tiene una definida (~50/50 entre 2 y 3 arcos).
   Cualquier intervención basada en arc-prior debe resolver esto para ser usable en el punto
   de uso real, no solo en benchmark retrospectivo.

---

## DVH loss diferenciable (Nguyen et al. 2020) — detalle técnico y aplicabilidad directa

**Referencia completa:** Nguyen D, McBeth R, Sadeghnejad Barkousaraie A, Bohara G, Shen C,
Jia X, Jiang S. *Incorporating human and learned domain knowledge into training deep neural
networks: A differentiable dose volume histogram and adversarial inspired framework for
generating Pareto optimal dose distributions in radiation therapy.* Med Phys. 2020
Mar;47(3):837-849. doi:10.1002/mp.13955.

### Qué propone
El DVH real no es diferenciable (involucra contar vóxeles por encima de un umbral de dosis,
una operación escalón). Los autores lo aproximan reemplazando el escalón por una **sigmoide**:

```
v_s,dt(D, M) = Σ_{i,j,k} Sigmoid( m/βt · (D[i,j,k] − dt) ) · M_s[i,j,k]  /  Σ_{i,j,k} M_s[i,j,k]
```

- `D` = dosis 3D (predicha o real), `M_s` = máscara binaria de la estructura `s`.
- `dt` = umbral de dosis del bin `t` del histograma; `βt` = ancho de ese bin.
- `m` = parámetro de "nitidez" de la sigmoide. A mayor `m`, la aproximación converge al DVH
  exacto (escalón real); a menor `m`, la curva es más suave.
- El DVH aproximado completo es el vector `DVH_s = [v_s,d1, v_s,d2, ..., v_s,dnt]` — un punto
  por cada umbral de dosis muestreado, cubriendo todo el rango, no solo 2-3 puntos clínicos.

Con esto, `∂DVH/∂D` existe → se puede definir una loss:
```
L_DVH = (1/n_s)(1/n_t) Σ_s || DVH_s(D_real) − DVH_s(D_pred) ||²
```
y sumarla a la MSE/MAE de dosis voxel-wise: `L_total = L_MSE + λ_DVH·L_DVH (+ λ_ADV·L_ADV)`.

### Hallazgo clave sobre el parámetro `m` (importante para no repetir el error de MomentLoss)
Los autores probaron `m` alto (curva más fiel al DVH exacto) y encontraron que produce
**mínimos locales más marcados y gradientes más filosos** — peor para el entrenamiento.
Con `m` bajo, la aproximación es menos fiel al DVH real pero **el mínimo global se mantiene
en el mismo lugar** y el gradiente es más suave y manejable. **Usaron `m=1` en todo el
estudio**, no el valor que más se parece al DVH exacto. Es un dato no obvio y a favor de la
implementación: no hay que perseguir la aproximación más "exacta" del DVH, sino la que da
mejor gradiente.

### Resultado reportado (70 pacientes próstata IMRT, dosis relativa)
Comparando MSE solo vs. MSE+DVH vs. MSE+DVH+adversarial: agregar la loss de DVH mejoró
consistentemente conformación, homogeneidad, dose spillage (R50) y cobertura de PTV
(D95/D98/D99) — con mejoras estadísticamente significativas (p≤0.007) en casi todas las
comparaciones. **No mejoró la dosis media a OARs** (donde MSE puro ya es competitivo, por
diseño — MSE minimiza error promedio, que es casi lo mismo que dosis media) ni el máximo en
cabezas femorales (zona de dosis baja y alta variabilidad, lejos del PTV). Es decir: la DVH
loss ayuda donde el objetivo clínico depende de la **forma** de la distribución de dosis
(percentiles, cobertura), no donde ya alcanza con el promedio.

### Por qué esto es directamente aplicable a este proyecto (y corrige MomentLoss)
- **Mismo diagnóstico que ya tenías para exp006:** MomentLoss (Jhanwar et al. 2022) atacaba
  momentos de potencia (M1/M2/M10), que no corresponden a los percentiles clínicos
  (D95/V65Gy/etc.). Esta DVH loss ataca **directamente los puntos del DVH real** (vía la
  aproximación sigmoide), evitando ese mismatch de raíz.
- **Mantiene el mapa 3D como salida** — a diferencia de "predecir el DVH directamente"
  (descartado en la discusión por perder la info espacial que necesita el KBP), esto sigue
  siendo `L_MSE(dosis) + λ·L_DVH(dosis)`: la red predice el volumen completo, y el término
  DVH solo *guía* el entrenamiento hacia los percentiles que importan clínicamente.
- **Directamente instanciable con la infraestructura ya existente:** ya se tiene
  `compute_gt_dvh_hipo.py` (cálculo de DVH sobre la grilla 256×256) — la versión
  diferenciable reemplazaría el cálculo exacto (no diferenciable) por la aproximación
  sigmoide `v_s,dt`, usando los mismos umbrales operativos (V65/V55/V45, D95/D98) como los
  `dt` a incluir en el vector de DVH.
- **La componente adversarial (ADV) es opcional y separable.** El paper reporta que
  MSE+DVH+ADV es mejor que MSE+DVH solo, pero MSE+DVH ya captura la mayor parte de la
  mejora. Dado el tamaño de dataset de este proyecto (mucho menor a los 70 pacientes ×
  1200 planes del paper), empezar sin el componente adversarial (que agrega otra red y
  complejidad de entrenamiento) es la opción más razonable — el paper mismo nota que ADV
  aumenta considerablemente el tiempo de entrenamiento (de 2.3 a 3.8 días) para una mejora
  incremental sobre DVH solo.

### Diferencia de escala/contexto a tener en cuenta antes de portar
El paper usa **IMRT de 7 haces estáticos**, no VMAT, y arrays de 96×96×24 a planes generados
sintéticamente (Pareto surface, no planes clínicos reales aprobados). No cambia la validez de
la loss (es agnóstica a la técnica de entrega), pero sí significa que el rango de "cuánta
mejora esperar" no es directamente transferible a este dataset — hay que medirlo empíricamente
acá, no asumir que se replican los mismos números.

---

## Bibliografía mencionada en el proyecto

> Nota: no tengo acceso a búsqueda en este documento — estas referencias vienen de lo ya
> discutido en el proyecto, no verificadas en este momento. Confirmar cita exacta (año,
> venue) antes de usarlas en un manuscrito.

- **DoseDiff** (Zhang et al., 2024) — PSDM + arquitectura multi-encoder. Base conceptual de
  la decisión PSDM-sobre-masks.
- **HD U-Net** (Nguyen et al., 2019) — U-Net con conexiones densas por nivel. Arquitectura
  alternativa evaluada/considerada en el plan de ablaciones.
- **Moment Loss** (Jhanwar et al., 2022) — MAE + momentos de potencia M1/M2/M10 sobre el
  histograma de dosis. λ=0.01 en el paper original; no replicado en este dataset (exp006).
  Es el antecedente directo de la idea "loss híbrida dosis+DVH", con el mismatch de
  percentiles ya identificado como la causa de que no funcionara acá.
- **OpenKBP** (Babier et al., 2021) — dose score y DVH score como métricas estándar de
  benchmark KBP; se usan en este proyecto para comparabilidad con la literatura.
- **Paper base de Pablo para la loss híbrida dosis+DVH:** Nguyen et al. (2020, *Med Phys*
  47(3):837-849, doi:10.1002/mp.13955) — ver desarrollo técnico completo en la sección
  "DVH loss diferenciable" arriba. Ya no está pendiente de compartir.

---

## Cómo retomar el tema (siguiente paso sugerido)

No conviene invertir en arc-prior/arquitecturas nuevas todavía. El orden lógico:

1. Implementar la métrica espectral angular (cuantifica el artefacto radial).
2. Implementar la métrica de DVH-curva-completa (vara KBP real).
3. Medir ambas sobre exp002 actual (normo e hipo).
4. Con esos números: decidir si el artefacto es grande y si alguna intervención (arc-prior,
   loss híbrida, o simplemente más datos) lo reduce — recién ahí justificar el esfuerzo de
   una arquitectura nueva.

**La loss híbrida dosis+DVH (Nguyen et al. 2020) ya no depende de ese diagnóstico para
arrancar** — es independiente del artefacto radial (ataca cobertura/homogeneidad/conformación,
no directamente la textura angular) y tiene receta de implementación concreta (sigmoide,
`m=1`, sin componente adversarial para empezar). Puede implementarse y probarse en paralelo
a los pasos 1-3, como exp007 de la serie normo (o directo sobre hipo, dado que ya se tiene
`compute_gt_dvh_hipo.py` como base para los umbrales `dt`). No es necesario esperar a decidir
sobre el arc-prior para avanzar con esto.
