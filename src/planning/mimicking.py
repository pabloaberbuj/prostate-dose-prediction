"""Mimicking (PIPELINE_KBP_PDRT.md, sección A4/B paso 4): optimiza
leaves/MU/jaws de los arcos reales de un paciente (partiendo de la geometría
clínica ya tratada, no de una apertura aleatoria) para que la dosis del motor
PDRT comisionado se acerque voxel a voxel a la dosis U-Net (target,
`src/bridge/unet_to_target.py`), sujeto a `deliverability_loss` y (desde
2026-08-15) a `dvh_loss` (términos DVH sobre puntos específicos del target).

`dvh_loss` existe porque el MSE por vóxel NO garantiza las métricas de
percentil que importan clínicamente: en la corrida v5 (2026-08-15, sin tope
de recorrido, mse convergiendo a 0.0059) el plan recalculado en Eclipse tenía
D98(PTV)=88.1% y V70(Recto)=47.0%, mientras el target de U-Net (dosimétricamente
bueno, casi igual al plan clínico real) tenía D98=97.7% y V70=35.6% — un mse
bajo no evitó que la cola de la distribución se corriera justo en los puntos
que importan. Ver `compute_dvh_targets`/`dvh_loss` y
project_mimicking_pipeline_status.md.

IMPORTANTE (2026-08-10): `deliv_weight=0.0` (default) NO es deliverable —
la prueba real en Eclipse de un RP escrito con deliv_weight=0.0 disparó
errores de "Leaf speed is too high" en decenas de control points y bajó la
dosis en PTV al recalcular con AAA. Para generar un RP importable hay que
usar `deliv_weight > 0` (ver `deliverability_loss`).

Parametrización de leaves/jaws vía sigmoid (center+width, `make_ordered_pairs`)
y MU vía softplus — aperturas válidas por construcción, mismo patrón que
`examples/optimization.ipynb` de PyDoseRT (ver notas en COMISIONAMIENTO_PDRT.md
y PIPELINE_KBP_PDRT.md: "copiar este patrón, no optimizar posiciones crudas").
A diferencia del ejemplo (arranca de una apertura random uniforme), acá se
invierte esa misma parametrización para arrancar EXACTAMENTE en la apertura
clínica real de cada arco — `invert_ordered_pairs`/`_inverse_softplus`.

Cada arco se pasa por separado a `engine.compute_dose(...)` (no por
`engine.forward()` directo) porque cada arco tiene su propia secuencia de
ángulos de gantry — `compute_dose` reconfigura la geometría de rotación en
cada llamada a partir del `BeamSequence` que recibe; `forward()` asume la
geometría ya fijada en la construcción del motor y no sirve para alternar
entre arcos con ángulos distintos dentro del mismo closure de LBFGS.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pydosert.data import BeamSequence
from pydosert.objectives.losses import dvh_Vx_loss
from src.bridge.unet_to_target import unet_dose_to_pdrt_grid
from src.planning.build_beams import STRUCT_NAMES_DEFAULT, load_patient_and_arcs
from src.planning.engine_setup import build_dose_engine, load_machine_config
from src.planning.ptv_conforming_init import (
    _rate_limit_sequence,
    compute_ptv_conforming_aperture,
    wrapped_gantry_diff_deg,
)

_wrapped_gantry_diff_deg = wrapped_gantry_diff_deg  # alias histórico

LEAF_MIN_OPENING_MM = 1.0
LEAF_MAX_OPENING_MM = 400.0  # coincide con field_size (400,400) del RP / machine_config
JAW_MIN_OPENING_MM = 1.0
JAW_MAX_OPENING_MM = 400.0


def make_ordered_pairs(x: torch.Tensor, min_opening: float, max_opening: float) -> torch.Tensor:
    """x[...,0]=center crudo, x[...,1]=ancho crudo -> (left,right) ordenado,
    ancho >= min_opening por construcción. Igual a examples/optimization.ipynb.
    """
    center = (max_opening * torch.sigmoid(x[..., 0])) - max_opening / 2.0
    width = min_opening + (max_opening - min_opening) * torch.sigmoid(x[..., 1])
    left = center - 0.5 * width
    right = center + 0.5 * width
    return torch.stack([left, right], dim=-1)


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def invert_ordered_pairs(left_right: torch.Tensor, min_opening: float, max_opening: float) -> torch.Tensor:
    """Inversa de `make_ordered_pairs`: da el parámetro crudo x tal que
    make_ordered_pairs(x) reproduce (left, right) reales — para arrancar la
    optimización en la apertura clínica real. Pares con ancho real fuera de
    [min_opening, max_opening] (leaves cerradas/con overtravel, ancho~0 o
    negativo) se clampean al límite más cercano — no se puede representar un
    ancho menor a min_opening con esta parametrización, la desviación
    dosimétrica de esos pares (casi cerrados de por sí) es despreciable.
    """
    left, right = left_right[..., 0], left_right[..., 1]
    center = 0.5 * (left + right)
    width = (right - left).clamp(min=min_opening, max=max_opening)
    x0 = _logit((center + max_opening / 2.0) / max_opening)
    x1 = _logit((width - min_opening) / (max_opening - min_opening))
    return torch.stack([x0, x1], dim=-1)


def _inverse_softplus(y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    y = y.clamp(min=eps)
    return y + torch.log(-torch.expm1(-y))


class OptimizableArc:
    """Copia reparametrizada (sigmoid/softplus) de un `BeamSequence` real.

    Por default (`closed_start=False`) arranca reproduciendo exactamente la
    apertura/MU real del arco clínico (warm start, ver `PIPELINE_KBP_PDRT.md`).

    Con `closed_start=True` (2026-08-10, a pedido de Pablo: el caso real de
    despliegue no tiene un RP del paciente para arrancar de ahí) se ignora la
    apertura/MU real y se arranca "en frío": MLC cerrado (ancho mínimo,
    centrado en 0), jaws totalmente abiertos (el MLC es el único que
    bloquea), MU uniforme por control point (mismo total que el plan real,
    como estimación gruesa de escala — no es información de secuencia por
    CP, sólo una magnitud de referencia). Probado en la práctica (2026-08-10):
    la optimización queda clavada casi de entrada (LBFGS no logra moverse del
    punto de partida, ver project_mimicking_pipeline_status.md) — motivó
    `ptv_conform_start` como alternativa.

    Con `ptv_conform_start=True` se arranca conformando el MLC/jaws a la
    proyección BEV real del PTV en cada control point (con margen —
    `leaf_margin_mm`/`jaw_margin_mm`), igual al criterio que usa Eclipse para
    sembrar la optimización de VMAT — ver `ptv_conforming_init.py`. Requiere
    `ptv_mask`/`resolution`/`leaf_widths` (del paciente y de la config de
    máquina comisionada). MU sigue siendo uniforme, igual que `closed_start`.

    La geometría del arco (ángulos de gantry/colimador, isocentro, SID,
    field_size) SIGUE viniendo del template real en los tres modos — es una
    decisión de diseño ya tomada en `build_beams.py` (A3: la geometría de
    arcos sale de un plan real tratado, usado como plantilla, no la apertura).

    NOTA: `deliverability_loss` depende de `ref_leaf_positions` etc., que
    son siempre los valores reales (incluso con `closed_start`/
    `ptv_conform_start`) — en el caso real sin RP no habría con qué
    compararlos. Correr con `deliv_weight>0` en estos modos no tiene sentido
    todavía; usar `deliv_weight=0` (ver docstring del módulo).

    `optimize_jaws=False` (default, 2026-08-11 a pedido de Pablo: el equipo
    real no tiene jaw tracking, la mordaza Y queda fija durante todo el arco
    — no tiene sentido optimizarla). La mordaza queda clavada en su valor de
    partida (real, o conformada al PTV según el modo) — no entra en
    `parameters()`, LBFGS nunca la toca. Confirmado con una prueba real: con
    `optimize_jaws=True` (comportamiento viejo) y `deliv_weight=0`, la
    mordaza se fue a valores sin sentido en sólo 6 pasos (ver
    project_mimicking_pipeline_status.md) — al MLC casi no le importa dónde
    está la mordaza para el mse, así que LBFGS la arrastra sin control."""

    def __init__(
        self, beam_sequence: BeamSequence, closed_start: bool = False,
        ptv_conform_start: bool = False, ptv_mask=None, resolution=None,
        leaf_widths=None, leaf_margin_mm: float = 2.0, jaw_margin_mm: float = 2.0,
        leaf_max_step_mm: float = None, machine_config=None,
        optimize_jaws: bool = False,
    ):
        self.gantry_angles = beam_sequence.gantry_angles
        self.collimator_angles = beam_sequence.collimator_angles
        self.field_size = beam_sequence.field_size
        self.iso_center = beam_sequence.iso_center
        self.sid = beam_sequence.sid

        if closed_start:
            n_cp, n_leaves, _ = beam_sequence.leaf_positions.shape
            # x0=0 -> center=0 ; x1 muy negativo -> sigmoid~0 -> ancho~min_opening (cerrado)
            self.raw_leaf = torch.stack([
                torch.zeros(n_cp, n_leaves), torch.full((n_cp, n_leaves), -12.0),
            ], dim=-1).to(beam_sequence.leaf_positions).requires_grad_(True)
            # x0=0 -> center=0 ; x1 muy positivo -> sigmoid~1 -> ancho~max_opening (abierto)
            self.raw_jaw = torch.stack([
                torch.zeros(n_cp), torch.full((n_cp,), 12.0),
            ], dim=-1).to(beam_sequence.jaw_positions).requires_grad_(True)
            mu_total = beam_sequence.mus.detach().sum()
            # mus[0]=0 por convencion DICOM (el primer CP no entrega MU, ver
            # nota en deliverability_loss); el total se reparte en los CP-1
            # segmentos reales.
            mu_uniforme = (mu_total / max(n_cp - 1, 1)).expand(n_cp).clone()
            mu_uniforme[0] = 0.0
            self.raw_mu = _inverse_softplus(mu_uniforme).requires_grad_(True)
        elif ptv_conform_start:
            if ptv_mask is None or resolution is None or leaf_widths is None:
                raise ValueError(
                    "ptv_conform_start=True necesita ptv_mask, resolution y leaf_widths."
                )
            leaf_init, jaw_init = compute_ptv_conforming_aperture(
                ptv_mask, beam_sequence.gantry_angles, beam_sequence.collimator_angles,
                iso_center=beam_sequence.iso_center, sid=beam_sequence.sid, resolution=resolution,
                leaf_widths=leaf_widths, field_size=beam_sequence.field_size,
                leaf_margin_mm=leaf_margin_mm, jaw_margin_mm=jaw_margin_mm,
                leaf_max_step_mm=leaf_max_step_mm,
                max_leaf_speed_mm_s=(float(machine_config.maximum_leaf_speed)
                                      if machine_config is not None else 22.5),
                max_gantry_speed_deg_s=(float(machine_config.maximum_gantry_angle_speed)
                                         if machine_config is not None else 4.8),
                speed_safety=INIT_SPEED_SAFETY,
                device=beam_sequence.leaf_positions.device, dtype=beam_sequence.leaf_positions.dtype,
            )
            self.raw_leaf = invert_ordered_pairs(
                leaf_init.to(beam_sequence.leaf_positions), LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM
            ).clone().requires_grad_(True)
            # La apertura conformada al PTV queda guardada como TECHO de
            # apertura para `project_aperture_bounds` (cerrar hacia adentro
            # sigue libre). Ver esa función.
            self.bound_leaf_positions = leaf_init.to(beam_sequence.leaf_positions).clone()
            self.raw_jaw = invert_ordered_pairs(
                jaw_init.to(beam_sequence.jaw_positions), JAW_MIN_OPENING_MM, JAW_MAX_OPENING_MM
            ).clone().requires_grad_(True)
            n_cp = beam_sequence.mus.shape[0]
            mu_total = beam_sequence.mus.detach().sum()
            # mus[0]=0 por convencion DICOM (el primer CP no entrega MU, ver
            # nota en deliverability_loss); el total se reparte en los CP-1
            # segmentos reales.
            mu_uniforme = (mu_total / max(n_cp - 1, 1)).expand(n_cp).clone()
            mu_uniforme[0] = 0.0
            self.raw_mu = _inverse_softplus(mu_uniforme).requires_grad_(True)
        else:
            self.raw_leaf = invert_ordered_pairs(
                beam_sequence.leaf_positions.detach(), LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM
            ).clone().requires_grad_(True)
            self.raw_jaw = invert_ordered_pairs(
                beam_sequence.jaw_positions.detach(), JAW_MIN_OPENING_MM, JAW_MAX_OPENING_MM
            ).clone().requires_grad_(True)
            self.raw_mu = _inverse_softplus(beam_sequence.mus.detach()).clone().requires_grad_(True)

        self.optimize_jaws = optimize_jaws
        self.raw_jaw.requires_grad_(optimize_jaws)

        # Valores reales del plan clínico (deliverable, ya tratado) SIN el
        # round-trip invert/make_ordered_pairs — referencia para
        # `deliverability_loss` (ver ahí por qué se usan estos, y no una
        # fórmula de velocidad máxima de máquina). Siempre los reales, aun
        # con closed_start=True (ver nota de la clase).
        self.ref_leaf_positions = beam_sequence.leaf_positions.detach().clone()
        self.ref_jaw_positions = beam_sequence.jaw_positions.detach().clone()
        self.ref_mus = beam_sequence.mus.detach().clone()

    def parameters(self):
        params = [self.raw_leaf, self.raw_mu]
        if self.optimize_jaws:
            params.append(self.raw_jaw)
        return params

    def to_beam_sequence(self) -> BeamSequence:
        return BeamSequence(
            mus=F.softplus(self.raw_mu),
            leaf_positions=make_ordered_pairs(self.raw_leaf, LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM),
            jaw_positions=make_ordered_pairs(self.raw_jaw, JAW_MIN_OPENING_MM, JAW_MAX_OPENING_MM),
            gantry_angles=self.gantry_angles,
            collimator_angles=self.collimator_angles,
            field_size=self.field_size,
            iso_center=self.iso_center,
            sid=self.sid,
        )


LEAF_JAW_SPEED_EPS_MM = 0.5  # piso absoluto de tolerancia (ver deliverability_loss)
MU_RATE_EPS = 0.05


# Pesos por estructura para el MSE de dosis (2026-08-13). Motivo: con peso
# uniforme sobre BODY, el 89.1% del presupuesto de la loss caía en vóxels por
# debajo del 20% de la prescripción — justamente donde el target del U-Net es
# menos confiable y probablemente no entregable — y sólo el 0.6% por encima
# del 90% (PTV_High es el 0.47% de BODY). Medido en el paciente piloto; era
# la explicación más probable de que los planes salieran sin modulación.
# Pablo eligió pesos por estructura (interpretables, como en un TPS) en vez de
# una rampa continua por nivel de dosis — esa alternativa quedó anotada como
# pendiente en project_mse_dose_band_weighting.md.
# "BODY" es el peso del tejido restante (todo lo que no cae en otra estructura).
# El peso es el SHARE del presupuesto de la loss, no un peso por vóxel (ver
# `build_mse_weights`): normalizado por volumen, así una estructura grande no
# se come la loss sólo por ser grande, y el balance no cambia de paciente a
# paciente cuando cambian los volúmenes. "BODY" = tejido restante.
STRUCT_WEIGHTS_DEFAULT = {
    "PTV_High": 100.0,
    "Rectum": 30.0,
    "Bladder": 30.0,
    "FemoralHead_L": 10.0,
    "FemoralHead_R": 10.0,
    "BODY": 10.0,
}


# Recorrido máximo de UNA lámina a lo largo de todo el arco (max-min sobre los
# control points). Es la restricción de carro del MLC: medido, el plan v3
# llegaba a 156-177mm en 1-3 láminas por banco y Eclipse avisaba movimiento de
# carro; el plan clínico real se queda en 101-112mm. Ojo que NO es lo mismo que
# el span del banco dentro de un mismo CP (max-min entre láminas), que en
# ambos planes estaba dentro de límite.
LEAF_ARC_RANGE_MAX_MM = 150.0

# Cuánto puede ABRIR una lámina más allá de la silueta del PTV conformada
# (`ptv_conforming_init`). Cerrar hacia adentro queda libre — eso ES la
# modulación; lo que se bloquea es abrir donde no hay blanco, que es lo que se
# veía en la captura de Eclipse (láminas abiertas fuera del PTV) y lo que
# genera los puntos calientes externos.
#
# Bajado de 10.0 a 3.0mm (2026-08-16): medido en el paciente piloto que con
# leaf_margin_mm=5 + aperture_extra_mm=10 (30mm de holgura total por par de
# láminas) el plan v6 optimizado terminaba con 33.8%/32.9% del área abierta
# del MLC por fuera de la silueta real del PTV (vs 7.0%/6.7% del plan
# clínico) — la causa de la mala conformación del 100% y el punto caliente
# >110% en la zona inferoanterior que señaló Pablo. De ese 33.8%, el propio
# init conformado (antes de optimizar) ya aportaba 25.7% con leaf_margin=5mm
# (ver chk_init_conformidad2.py) — el optimizador sólo sumaba ~8 puntos más
# empujando aperturas hacia afuera dentro del margen que este límite permite.
APERTURE_EXTRA_MARGIN_MM = 3.0


def project_aperture_bounds(optim_arcs: list, extra_margin_mm: float = APERTURE_EXTRA_MARGIN_MM) -> None:
    """Impide que las láminas se abran más allá de la silueta del PTV
    (+`leaf_margin_mm` del init +`extra_margin_mm`), dejando libre el cierre
    hacia adentro. Requiere que el arco se haya construido con
    `ptv_conform_start=True` (de ahí sale `bound_leaf_positions`); si no, no
    hace nada.
    """
    with torch.no_grad():
        for oa in optim_arcs:
            lim = getattr(oa, "bound_leaf_positions", None)
            if lim is None:
                continue
            pos = make_ordered_pairs(oa.raw_leaf, LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM)
            # izquierda: no más a la izquierda que el límite; derecha: no más a la derecha
            izq = torch.maximum(pos[..., 0], lim[..., 0] - extra_margin_mm)
            der = torch.minimum(pos[..., 1], lim[..., 1] + extra_margin_mm)
            ancho = der - izq
            angosto = ancho < LEAF_MIN_OPENING_MM
            if angosto.any():
                centro = 0.5 * (izq + der)
                izq = torch.where(angosto, centro - LEAF_MIN_OPENING_MM / 2, izq)
                der = torch.where(angosto, centro + LEAF_MIN_OPENING_MM / 2, der)
            oa.raw_leaf.copy_(invert_ordered_pairs(
                torch.stack([izq, der], dim=-1), LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM))


def project_leaf_arc_range(optim_arcs: list, max_range_mm: float = LEAF_ARC_RANGE_MAX_MM) -> None:
    """Acota el recorrido de cada lámina a lo largo del arco (max-min sobre
    control points) a `max_range_mm` — la restricción de carro del MLC. Las
    láminas que se pasan se comprimen hacia el centro de su propio recorrido.
    """
    with torch.no_grad():
        for oa in optim_arcs:
            pos = make_ordered_pairs(oa.raw_leaf, LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM)
            nuevo = []
            for banco in (0, 1):
                x = pos[..., banco]                       # [CP, N]
                lo = x.min(dim=0).values                  # [N]
                hi = x.max(dim=0).values
                centro = 0.5 * (lo + hi)
                media = max_range_mm / 2.0
                nuevo.append(torch.clamp(x, (centro - media).unsqueeze(0),
                                            (centro + media).unsqueeze(0)))
            izq, der = nuevo
            ancho = der - izq
            angosto = ancho < LEAF_MIN_OPENING_MM
            if angosto.any():
                c = 0.5 * (izq + der)
                izq = torch.where(angosto, c - LEAF_MIN_OPENING_MM / 2, izq)
                der = torch.where(angosto, c + LEAF_MIN_OPENING_MM / 2, der)
            oa.raw_leaf.copy_(invert_ordered_pairs(
                torch.stack([izq, der], dim=-1), LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM))


def project_leaf_velocity(optim_arcs: list, config, safety: float = None) -> None:
    """Proyecta las láminas de vuelta al conjunto FACTIBLE (gradiente
    proyectado): tras cada paso del optimizador, recorta la secuencia de cada
    lámina para que ningún salto entre control points consecutivos supere el
    presupuesto físico del segmento. Modifica `raw_leaf` in place.

    Por qué proyección y no sólo la penalización de `deliverability_loss`:
    con Adam la penalización pierde casi toda su fuerza, porque Adam normaliza
    el paso por parámetro dividiendo por el RMS del gradiente — subir
    `deliv_weight` cambia la dirección pero casi no el tamaño del paso.
    Medido: con Adam lr=0.01 y deliv_weight=50, el exceso creció monótono
    0.75 -> 1.15 -> 1.51 mm en 3 pasos sin autocorregirse. La proyección
    garantiza factibilidad exacta y de paso elimina el problema de calibrar
    `deliv_weight` (ver project_mimicking_pipeline_status.md).

    Se hace en el espacio FÍSICO (mm) y se vuelve al espacio crudo con
    `invert_ordered_pairs`, que es la inversa exacta de `make_ordered_pairs`.
    """
    safety = DELIV_SAFETY if safety is None else safety
    with torch.no_grad():
        for oa in optim_arcs:
            dgantry = wrapped_gantry_diff_deg(oa.gantry_angles).to(oa.raw_leaf.device)
            paso_max = (float(config.maximum_leaf_speed)
                        * (dgantry / float(config.maximum_gantry_angle_speed))
                        * safety).unsqueeze(-1)  # [CP-1, 1]
            pos = make_ordered_pairs(oa.raw_leaf, LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM)
            izq = _rate_limit_sequence(pos[..., 0], paso_max)
            der = _rate_limit_sequence(pos[..., 1], paso_max)
            # El recorte independiente de cada banco puede invertir el par;
            # misma red de seguridad que en el init conformado.
            ancho = der - izq
            angosto = ancho < LEAF_MIN_OPENING_MM
            if angosto.any():
                centro = 0.5 * (izq + der)
                izq = torch.where(angosto, centro - LEAF_MIN_OPENING_MM / 2, izq)
                der = torch.where(angosto, centro + LEAF_MIN_OPENING_MM / 2, der)
            nuevo = torch.stack([izq, der], dim=-1)
            oa.raw_leaf.copy_(
                invert_ordered_pairs(nuevo, LEAF_MIN_OPENING_MM, LEAF_MAX_OPENING_MM)
            )


def _proyectar_todo(optim_arcs, config, project_aperture, aperture_extra_mm,
                    project_arc_range, n_pasadas: int = 5):
    """Aplica las proyecciones en cadena, ITERANDO: las restricciones se pisan
    entre sí (el limitador de velocidad puede volver a abrir una lámina que la
    proyección de apertura acababa de cerrar, porque la acerca a su vecina).
    Con una sola pasada quedaba ~20mm de apertura fuera del PTV; iterando
    convergen, porque los tres conjuntos son convexos y el init pertenece a
    los tres (proyecciones alternadas / POCS).

    La de VELOCIDAD va última en cada pasada a propósito: es la que Eclipse
    chequea más duro, así que conviene que sea la que quede exacta.
    """
    for _ in range(n_pasadas):
        if project_aperture:
            project_aperture_bounds(optim_arcs, aperture_extra_mm)
        if project_arc_range:
            project_leaf_arc_range(optim_arcs)
        project_leaf_velocity(optim_arcs, config)


def parse_struct_weights(texto: str) -> dict:
    """'PTV_High=100,Rectum=30,BODY=10' -> dict. Los nombres ausentes quedan
    con su valor de `STRUCT_WEIGHTS_DEFAULT` (no se borran)."""
    pesos = dict(STRUCT_WEIGHTS_DEFAULT)
    if not texto:
        return pesos
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "=" not in parte:
            raise ValueError(f"Peso mal formado: {parte!r} (se espera Nombre=valor)")
        nombre, valor = parte.split("=", 1)
        pesos[nombre.strip()] = float(valor)
    return pesos


def build_mse_weights(patient, struct_weights: dict, device, dtype):
    """Volumen de pesos para el MSE, uno por vóxel, tal que la loss resulta

        mse = Σ_s  w_s · promedio_dentro_de_s((dosis_pred - target)²)  /  Σ_s w_s

    o sea: cada `w_s` es directamente el SHARE del presupuesto de la loss que
    se le da a esa estructura, independiente de su volumen. Se logra con peso
    por vóxel = `w_s / n_s` (n_s = vóxels donde esa estructura gana).

    Por qué normalizado por volumen y no peso por vóxel crudo: con peso crudo
    la Vejiga (572 cm³) se quedaba con más presupuesto que el PTV (149 cm³) al
    mismo peso nominal — el tamaño mandaba sobre la prioridad clínica, y el
    balance habría cambiado de paciente a paciente según los volúmenes.

    Estructuras superpuestas se resuelven por MÁXIMO peso, no por suma: un
    vóxel que está en PTV_High y en Rectum a la vez cuenta como PTV (el mayor).
    Así no se cuenta dos veces ni depende del orden del dict. Ojo que
    `load_structures` de pydosert elige la primera coincidencia por substring,
    así que "Rectum" es el recto completo (no "Rectum-PTV"), y su solapamiento
    con el PTV queda del lado del PTV por esta regla.

    Devuelve (pesos [H,D,W], resumen) — `resumen` es una lista de
    (nombre, n_voxels, peso, % del presupuesto de la loss) para poder ver el
    balance antes de gastar horas de optimización.
    """
    body = patient.structures.get("BODY")
    if body is None:
        raise ValueError("build_mse_weights necesita 'BODY' en patient.structures.")
    body_mask = body.to(device) > 0

    # 1) asignación disjunta por máximo peso: cada vóxel de BODY pertenece a
    #    una sola estructura (la de mayor peso que lo contiene) o al resto.
    w_rest = float(struct_weights.get("BODY", 1.0))
    ganador_w = torch.full(body.shape, w_rest, device=device, dtype=dtype)
    for name, wt in struct_weights.items():
        if name == "BODY":
            continue
        m = patient.structures.get(name)
        if m is None:
            print(f"[mimicking] aviso: '{name}' no está en patient.structures, se ignora su peso")
            continue
        m = m.to(device) > 0
        ganador_w[m] = torch.maximum(ganador_w[m], torch.tensor(float(wt), device=device, dtype=dtype))

    # 2) peso por vóxel = w_s / n_s  -> el share total de cada estructura es w_s
    weights = torch.zeros(body.shape, device=device, dtype=dtype)
    regiones = []
    for name, wt in list(struct_weights.items()):
        if name == "BODY":
            continue
        m = patient.structures.get(name)
        if m is None:
            continue
        regiones.append((name, float(wt), (m.to(device) > 0) & body_mask & (ganador_w == float(wt))))
    asignado = torch.zeros_like(body_mask)
    for _, _, reg in regiones:
        asignado |= reg
    regiones.append(("(tejido restante)", w_rest, body_mask & ~asignado))

    conteos = {name: int(reg.sum()) for name, _, reg in regiones}
    for name, wt, reg in regiones:
        if conteos[name]:
            weights[reg] = wt / conteos[name]

    # share = w_s / Σ w_s sobre las regiones no vacías (por la normalización
    # por volumen, el share NO depende de n_s).
    suma_w = sum(wt for name, wt, _ in regiones if conteos[name])
    resumen = [
        (name, conteos[name], wt, 100.0 * wt / suma_w if (suma_w and conteos[name]) else 0.0)
        for name, wt, _ in regiones
    ]
    return weights, resumen


# Límites de MU por grado de gantry — de las "Operating Limits" reales de la
# máquina (0.30 a 20.00 MU/deg). pydosert NO tiene campo para esto, así que
# van acá. El PISO importa: el plan de la corrida de 3h se fue a 0.282 MU/deg
# y violaba el mínimo en 18 de 177 control points del arco 2 (el plan clínico
# real está en 0.63-2.12). Bajar las MU es la vía más barata que tiene LBFGS
# para reducir dosis, así que sin este piso las empuja fuera de rango.
MU_PER_DEG_MIN = 0.30
MU_PER_DEG_MAX = 20.0

# Se pide sólo el 95% del presupuesto físico. Las posiciones de colimador
# tienen precisión 1/10 mm en la máquina (ver "Operating Limits"), así que un
# plan justo en el límite puede cruzarlo al redondear en el DICOM, y el chequeo
# propio de Eclipse no es necesariamente idéntico a esta fórmula.
DELIV_SAFETY = 0.95

# El init conformado se suaviza a una fracción MENOR del presupuesto físico que
# la que exige la loss (0.80 vs 0.95), a propósito: si el init arranca pegado al
# borde de lo factible, el gradiente de la penalización ahí es ~0 pero cualquier
# paso de tamaño útil cruza el borde y paga cuadráticamente, así que el line
# search de LBFGS rechaza todo y la optimización queda clavada sin avanzar
# (confirmado empírico con deliv_weight=50: mse idéntico bit a bit en 3 pasos).
# Con 0.80 hay ~1.4mm de holgura en el segmento típico antes de que la
# penalización empiece a morder.
INIT_SPEED_SAFETY = 0.80

# Umbral para CONTAR violaciones en los diagnósticos (no afecta la loss). 0.01mm
# = 10 micrones, muy por debajo de la precisión de 0.1mm de la máquina: filtra
# el ruido de float32 del limitador de velocidad del init, que clampea justo al
# presupuesto y deja residuos de ~1e-5 mm que se contaban como "violación".
VIOL_REPORT_EPS_MM = 0.01


def deliverability_loss(optim_arcs: list, config, safety: float = DELIV_SAFETY) -> dict:
    """Entregabilidad contra los límites FÍSICOS REALES de la máquina, por
    segmento del arco — ya no contra el plan clínico del paciente.

    Presupuesto de tiempo por segmento: `dt = delta_gantry / maximum_gantry_angle_speed`,
    o sea el tiempo MÍNIMO posible (gantry a máxima velocidad). Es la cota más
    ESTRICTA y por lo tanto conservadora: si el gantry va más lento hay más
    tiempo y las láminas tienen MÁS lugar, nunca menos. Con eso:

      - lámina: `|delta_leaf| <= maximum_leaf_speed * dt`
      - MU (techo): `delta_MU <= min(maximum_dose_rate * dt, MU_PER_DEG_MAX * delta_gantry)`
      - MU (piso):  `delta_MU >= MU_PER_DEG_MIN * delta_gantry`
      - mordaza: `|delta_jaw| <= maximum_jaw_speed * dt` (inactivo mientras
        `optimize_jaws=False`, y OJO que `maximum_jaw_speed` sigue siendo el
        default de pydosert, no un valor medido — ver reference_real_machine_limits)

    Historia de por qué esta versión y no la anterior (importa para no repetir
    el error): dos intentos previos con fórmulas de velocidad de máquina se
    descartaron —  `minimum_gantry_angle_speed` daba una cota tan generosa que
    nunca se activaba, y `maximum_gantry_angle_speed` marcaba como violación
    tramos del propio plan clínico real. Lo segundo era un artefacto de
    constantes equivocadas: `machine_config_6MV.json` no tenía NINGÚN límite
    cinemático cargado, así que se usaban los defaults genéricos de pydosert
    (22.5 mm/s, 6.0 deg/s) en vez de los reales de esta máquina (25.0 mm/s,
    4.8 deg/s). Con los valores reales el plan clínico real da 0 violaciones
    de 21240 en ambos arcos, o sea la fórmula física SÍ lo valida. Eso además
    elimina la dependencia de tener el RP del paciente como referencia, que era
    el bloqueo de fondo para el caso real de despliegue sin RP.

    Agregación: se usa suma sobre láminas normalizada por número de segmentos
    (no promedio sobre los ~21240 pares lámina-CP) MÁS el máximo. Con promedio,
    6 violaciones puntuales quedaban diluidas a ~0 en el agregado —
    exactamente lo que pasó: `deliv_raw=0.00012` mientras Eclipse marcaba
    errores. El término de máximo apunta directo a la peor violación, que es
    lo que Eclipse chequea (caso por caso, no en promedio).

    Devuelve dict con los términos + 'total' + 'diag' (diagnósticos en
    unidades FÍSICAS: mm de exceso, cuántos pares violan, rango de MU/deg)
    para poder saber si el plan es entregable ANTES de ir a Eclipse.
    """
    leaf_loss = mu_hi_loss = mu_lo_loss = jaw_loss = torch.tensor(0.0)
    # `leaf_step_frac_max`: mayor |delta_leaf| como fracción del presupuesto.
    # Distingue "cómodamente adentro" (<<1) de "pegado al límite por dentro"
    # (~1) — con sólo el exceso no se puede diferenciar, porque un plan clavado
    # exactamente en la frontera también da exceso 0.
    # `leaf_travel_cm`: camino total recorrido por TODAS las láminas del arco.
    # Es la medida directa de cuánta modulación tiene el plan, y la que
    # importa: el plan clínico real de referencia recorre ~1580 cm por arco,
    # y una version sin modular (conformada al PTV) ~440 cm.
    # `leaf_step_p90_mm`: percentil 90 del |delta| por segmento. Va junto al
    # máximo a propósito — caracterizar el plan sólo con el máximo llevó a una
    # conclusión equivocada (ver project_mimicking_pipeline_status.md): un
    # único outlier puede marcar 100% de uso del presupuesto mientras el 90%
    # del MLC casi no se mueve.
    diag = {"leaf_viol_max_mm": 0.0, "leaf_viol_n": 0, "leaf_step_frac_max": 0.0,
            "leaf_travel_cm": 0.0, "leaf_step_p90_mm": 0.0,
            "mu_per_deg_min": float("inf"), "mu_per_deg_max": 0.0, "mu_per_deg_viol_n": 0}

    for oa in optim_arcs:
        bs = oa.to_beam_sequence()
        dgantry = _wrapped_gantry_diff_deg(oa.gantry_angles).to(bs.mus.device)  # [CP-1] deg
        n_seg = max(int(dgantry.shape[0]), 1)
        dt = dgantry / float(config.maximum_gantry_angle_speed)  # [CP-1] s (cota mínima)

        # --- láminas ---
        leaf_budget = (float(config.maximum_leaf_speed) * dt * safety).unsqueeze(-1)  # [CP-1,1]
        dleft = torch.diff(bs.leaf_positions[..., 0], dim=0).abs()
        dright = torch.diff(bs.leaf_positions[..., 1], dim=0).abs()
        exc = torch.cat([F.relu(dleft - leaf_budget), F.relu(dright - leaf_budget)], dim=1)
        leaf_loss = leaf_loss + (exc ** 2).sum() / n_seg + (exc.max() ** 2)

        # --- MU por segmento ---
        # OJO: `BeamSequence.mus` ya son las MU INCREMENTALES por control point
        # (así las arma `fetch_plan_data`: mu = cum_i - cum_{i-1}), NO
        # acumuladas. Entonces la MU del segmento i->i+1 es `mus[i+1]` directo,
        # y NO `diff(mus)` (que sería la segunda derivada, o sea el cambio de
        # la tasa de MU). La versión anterior de esta función usaba diff(mus) y
        # el error era invisible porque comparaba esa cantidad contra la misma
        # cantidad del plan real. `mus[0]` es 0 por convención DICOM (el primer
        # CP no entrega MU), así que `mus[1:]` alinea 1 a 1 con los segmentos.
        mu_seg = bs.mus[1:]
        mu_hi = torch.minimum(
            float(config.maximum_dose_rate) * dt,
            MU_PER_DEG_MAX * dgantry,
        ) * safety
        exc_hi = F.relu(mu_seg - mu_hi)
        mu_hi_loss = mu_hi_loss + (exc_hi ** 2).sum() / n_seg + (exc_hi.max() ** 2)

        # --- MU: piso por MU/deg (el que se estaba violando) ---
        mu_lo = MU_PER_DEG_MIN * dgantry / safety
        exc_lo = F.relu(mu_lo - mu_seg)
        mu_lo_loss = mu_lo_loss + (exc_lo ** 2).sum() / n_seg + (exc_lo.max() ** 2)

        # --- mordaza (inactivo con optimize_jaws=False) ---
        if oa.optimize_jaws:
            jaw_budget = (float(config.maximum_jaw_speed) * dt * safety).unsqueeze(-1)
            djaw = torch.diff(bs.jaw_positions, dim=0).abs()
            exc_j = F.relu(djaw - jaw_budget)
            jaw_loss = jaw_loss + (exc_j ** 2).sum() / n_seg + (exc_j.max() ** 2)

        with torch.no_grad():
            diag["leaf_viol_max_mm"] = max(diag["leaf_viol_max_mm"], float(exc.max()))
            diag["leaf_viol_n"] += int((exc > VIOL_REPORT_EPS_MM).sum())
            dmax_leaf = torch.maximum(dleft, dright)
            diag["leaf_step_frac_max"] = max(
                diag["leaf_step_frac_max"], float((dmax_leaf / leaf_budget).max())
            )
            todos = torch.cat([dleft, dright], dim=1)
            diag["leaf_travel_cm"] += float(todos.sum()) / 10.0
            diag["leaf_step_p90_mm"] = max(
                diag["leaf_step_p90_mm"], float(torch.quantile(todos.flatten().float(), 0.90))
            )
            mu_per_deg = mu_seg / dgantry.clamp(min=1e-6)
            diag["mu_per_deg_min"] = min(diag["mu_per_deg_min"], float(mu_per_deg.min()))
            diag["mu_per_deg_max"] = max(diag["mu_per_deg_max"], float(mu_per_deg.max()))
            diag["mu_per_deg_viol_n"] += int(
                ((mu_per_deg < MU_PER_DEG_MIN) | (mu_per_deg > MU_PER_DEG_MAX)).sum()
            )

    return {
        "leaf": leaf_loss,
        "mu_hi": mu_hi_loss,
        "mu_lo": mu_lo_loss,
        "jaw": jaw_loss,
        "total": leaf_loss + mu_hi_loss + mu_lo_loss + jaw_loss,
        "diag": diag,
    }


# Pesos internos (violation_weight/slack_weight) de cada término DVH — punto
# de partida, NO calibrado (mismo estado en que quedó deliv_weight=10.0 la
# primera vez, ver project_mimicking_pipeline_status.md). D98 vive en Gy
# (violaciones típicas ~0.1-0.3 Gy, al cuadrado ~0.01-0.1); V70/V50 viven en
# PUNTOS DE PORCENTAJE (violación medida en v5: ~11 puntos, al cuadrado 121) —
# sin bajar el peso interno de los términos de volumen, dominarían la loss
# por pura escala de unidades, no por importancia real (mismo tipo de
# problema que MU-vs-lámina en LBFGS, ver ese hallazgo más arriba). Se baja
# `violation_weight`/`slack_weight` de Vx ~500x respecto del default de
# pydosert (10/0.1) para que, a la violación medida en v5, el orden de
# magnitud del término quede parecido al de D98 en vez de ahogarlo.
DVH_VIOL_WEIGHT_DP = 10.0
DVH_SLACK_WEIGHT_DP = 0.1
DVH_VIOL_WEIGHT_VX = 0.02
DVH_SLACK_WEIGHT_VX = 0.0002

# Fracción de vóxeles del PTV promediados alrededor del rango D98 (ver
# smooth_d98_loss). 0.005 = ~99 vóxeles en el paciente piloto (19823 vóxeles
# de PTV_High), un orden de magnitud menor que el artefacto de borde ya
# conocido (1.8%/356 vóxeles, ver compute_dvh_targets) — pensado para repartir
# el gradiente sin diluirlo tanto que deje de representar D98 puntual.
D98_WINDOW_FRAC = 0.005


def smooth_d98_loss(
    dose_pred: torch.Tensor, mask: torch.Tensor, target: float,
    window_frac: float = D98_WINDOW_FRAC,
    violation_weight: float = DVH_VIOL_WEIGHT_DP, slack_weight: float = DVH_SLACK_WEIGHT_DP,
) -> torch.Tensor:
    """Sustituto de `pydosert.objectives.losses.dvh_Dp_loss(p=2)` para el
    término D98(PTV) — mismo hinge suave (relu**2 en violación/holgura), pero
    Dp se calcula como el PROMEDIO de una ventana de vóxeles alrededor del
    rango del 2%-ilo en vez de un solo vóxel (`vals_sorted[k]`).

    Motivo (diagnóstico 2026-08-18, `commissioning/diagnostics_mimicking_v7/
    diag_d98_onehot_and_chunk.py`): aunque el gradiente de un solo vóxel de
    `dose_pred` SÍ se reparte sobre miles de parámetros al retropropagar (el
    motor suma sobre todos los control points y el kernel de convolución
    esparce sobre una vecindad — no es un one-hot en espacio de parámetros,
    hipótesis inicial descartada), la IDENTIDAD de qué vóxel ocupa
    exactamente el rango del 2%-ilo SÍ es inestable de un paso a otro, y esa
    inestabilidad es ~18x peor con `leaf_margin_mm` ajustado (2mm, salto de 18
    posiciones en el orden tras un paso de Adam) que con margen ancho (5mm,
    salto de 1 posición) — medido directo en el paciente piloto, mismo target
    D98, misma magnitud de paso. Ese "salto" de a qué vóxel individual apunta
    el gradiente de D98 entre pasos es exactamente el tipo de discontinuidad
    que corrompe el par secante (y=Δgrad) que usa LBFGS para su aproximación
    de curvatura — hipótesis consistente con el freeze de v7, no confirmada
    todavía corriendo LBFGS real (ver project_mimicking_pipeline_status.md).
    Promediar una ventana de vóxeles en vez de tomar uno solo hace que el
    gradiente de un paso al siguiente comparta la mayoría de los vóxeles de
    la ventana en vez de poder saltar a una vecindad completamente distinta.
    """
    vals = dose_pred[mask > 0]
    if vals.numel() == 0:
        return torch.tensor(0.0, device=dose_pred.device)
    vals_sorted, _ = torch.sort(vals)
    n = vals_sorted.numel()
    k = int(0.02 * (n - 1))
    half = max(1, int(window_frac * n) // 2)
    lo, hi = max(0, k - half), min(n, k + half + 1)
    Dp = vals_sorted[lo:hi].mean()

    diff = target - Dp
    violation = F.relu(diff)
    slack = F.relu(-diff)
    return violation_weight * violation ** 2 + slack_weight * slack ** 2


def compute_dvh_targets(dose_target: torch.Tensor, patient, rx_frac_gy: float) -> dict:
    """D98(PTV_High) y V70/V50(Rectum/Bladder) calculados sobre la dosis
    TARGET de U-Net (no sobre dose_pred) — son los valores fijos hacia los
    que empujan los términos de `dvh_loss` durante toda la corrida.

    D98 = percentil ASCENDENTE 2 (98% del volumen recibe AL MENOS esa dosis),
    calculado directo con `torch.quantile` acá porque el target es constante
    (no hace falta el hinge diferenciable de `dvh_Dp_loss`, ese se usa sólo
    sobre `dose_pred` dentro de la loss). V70/V50 son la fracción de volumen
    con dosis >= 70%/50% de la dosis de prescripción POR FRACCIÓN (misma
    convención que `dose_target`/`dose_pred`, ver módulo `unet_to_target`).

    D98 se deja CRUDO a propósito (2026-08-16), aunque se sabe que está
    contaminado por un artefacto de borde: en el paciente piloto da 81.5%
    de Rx/fx, no por mala cobertura real sino por 1.8% del volumen del PTV
    (356/19823 vóxels) concentrado en el último corte h del bbox del PTV,
    con dosis target ~62% ahí (ver chk_target_dentro_ptv.py) — casi seguro
    un límite de z_range de la predicción del U-Net. El resto de la curva
    (D95=93.3%, D90=98.2%, D80=101.3%) no muestra este problema.

    Se probaron DOS formas de corregirlo — un piso fijo
    (`D98_TARGET_FLOOR_FRAC=0.95`, target 1.900Gy) y excluir los vóxels de
    borde del cálculo (target 1.731Gy) — y AMBAS rompieron la optimización:
    LBFGS quedaba clavado (mse idéntico bit-a-bit) desde el paso 2, con
    D98/V70/V50 congelados muy lejos de cualquier target. Aislado con
    varios smoke tests de 5 pasos, descartando una variable por vez: no era
    `aperture_extra_mm` (3mm y 8mm dieron el mismo freeze), no era
    `leaf_margin_mm` (2mm y 5mm dieron el mismo freeze), no era
    `--project-aperture` (con y sin esa proyección, mismo freeze). El
    control decisivo — la receta EXACTA de v6 (D98 crudo 81.5%, margen
    5mm, aperture_extra 10mm) — SÍ convergió normalmente (mse bajando paso
    a paso: 2.743→2.709→2.687→2.651→2.637). Es decir, el único factor que
    causa el freeze es subir el target de D98 por ENCIMA de su valor crudo,
    por poco que sea (incluso 81.5%→86.6% ya rompe). Mecanismo probable: la
    dosis cruda del init conformado (MU uniforme, sin optimizar) ya
    sobredosifica masivamente (D98~185% de Rx, V70_Recto~80%); el primer
    paso de LBFGS tiene que bajar MU mucho para corregir eso, y el término
    D98 "at_least" empieza a frenar esa bajada apenas la dosis cruza el
    target — con el valor crudo (81.5%) hay margen de sobra antes de
    chocar con ese freno; con cualquier valor más alto, el freno aparece
    antes, en un punto donde Recto/Vejiga siguen muy violados, y ahí LBFGS
    no encuentra ninguna dirección que mejore todo a la vez.

    Pendiente para una futura sesión (no resuelto acá): cómo corregir el
    artefacto de borde sin este freeze — candidatos no probados: activar
    `dvh_weight` recién después de unos pasos de MSE puro (dejar que baje
    la sobredosis inicial antes de sumar el término D98), o una rampa
    gradual del target en vez de un salto. Por ahora se prioriza tener una
    corrida que converja sobre corregir este artefacto conocido y menor.
    """
    device = dose_target.device
    ptv = patient.structures["PTV_High"].to(device) > 0
    rectum = patient.structures["Rectum"].to(device) > 0
    bladder = patient.structures["Bladder"].to(device) > 0

    d98_gy_raw = float(torch.quantile(dose_target[ptv].float(), 0.02))
    d98_gy = d98_gy_raw

    v70_thr_gy = 0.70 * rx_frac_gy
    v70_target_pct = 100.0 * float((dose_target[rectum] >= v70_thr_gy).float().mean())

    v50_thr_gy = 0.50 * rx_frac_gy
    v50_target_pct = 100.0 * float((dose_target[bladder] >= v50_thr_gy).float().mean())

    return {
        "ptv_d98_gy": d98_gy, "ptv_d98_gy_raw": d98_gy_raw,
        "rectum_v70_thr_gy": v70_thr_gy, "rectum_v70_target_pct": v70_target_pct,
        "bladder_v50_thr_gy": v50_thr_gy, "bladder_v50_target_pct": v50_target_pct,
    }


def dvh_loss(dose_pred: torch.Tensor, patient, dvh_targets: dict) -> dict:
    """Términos DVH sobre puntos específicos del DVH del target (PIPELINE_KBP_PDRT.md
    paso 2), no sobre la curva completa — el MSE por vóxel ya cubre la forma
    general de la distribución; estos términos apuntan justo a las métricas
    de percentil que el MSE deja sin garantizar (ver hallazgo v5 en el
    docstring del módulo).

    D98(PTV) usa `smooth_d98_loss` (este módulo), NO `pydosert.dvh_Dp_loss`
    directamente — ver el docstring de esa función para el porqué (2026-08-18:
    `dvh_Dp_loss` toma un solo vóxel (`vals_sorted[k]`) como Dp, y cuál vóxel
    exacto ocupa ese rango es inestable de un paso a otro, más con margen de
    conformado ajustado — sospechoso de corromper la curvatura de LBFGS y
    causar el freeze de v7). De paso, ojo con la convención de percentil de
    `dvh_Dp_loss` si se la vuelve a usar en otro lado: ordena ASCENDENTE y
    `p` es el índice percentil ascendente — p=98 da la dosis que sólo el 2%
    del volumen SUPERA (D2 clínico), p=2 da D98 clínico (98% del volumen
    ALCANZA O SUPERA esa dosis). Es lo inverso de lo que sugiere el nombre a
    primera vista.
    """
    device = dose_pred.device
    ptv = patient.structures["PTV_High"].to(device) > 0
    rectum = patient.structures["Rectum"].to(device) > 0
    bladder = patient.structures["Bladder"].to(device) > 0

    d98_loss = smooth_d98_loss(
        dose_pred, ptv, target=dvh_targets["ptv_d98_gy"],
        violation_weight=DVH_VIOL_WEIGHT_DP, slack_weight=DVH_SLACK_WEIGHT_DP,
    )
    v70_loss = dvh_Vx_loss(
        dose_pred, rectum, x=dvh_targets["rectum_v70_thr_gy"],
        target_volume_percent=dvh_targets["rectum_v70_target_pct"], direction="at_most",
        violation_weight=DVH_VIOL_WEIGHT_VX, slack_weight=DVH_SLACK_WEIGHT_VX,
    )
    v50_loss = dvh_Vx_loss(
        dose_pred, bladder, x=dvh_targets["bladder_v50_thr_gy"],
        target_volume_percent=dvh_targets["bladder_v50_target_pct"], direction="at_most",
        violation_weight=DVH_VIOL_WEIGHT_VX, slack_weight=DVH_SLACK_WEIGHT_VX,
    )

    with torch.no_grad():
        d98_actual = float(torch.quantile(dose_pred[ptv].float(), 0.02))
        v70_actual = 100.0 * float((dose_pred[rectum] >= dvh_targets["rectum_v70_thr_gy"]).float().mean())
        v50_actual = 100.0 * float((dose_pred[bladder] >= dvh_targets["bladder_v50_thr_gy"]).float().mean())

    return {
        "ptv_d98": d98_loss, "rectum_v70": v70_loss, "bladder_v50": v50_loss,
        "total": d98_loss + v70_loss + v50_loss,
        "diag": {
            "ptv_d98_actual_gy": d98_actual, "ptv_d98_target_gy": dvh_targets["ptv_d98_gy"],
            "rectum_v70_actual_pct": v70_actual, "rectum_v70_target_pct": dvh_targets["rectum_v70_target_pct"],
            "bladder_v50_actual_pct": v50_actual, "bladder_v50_target_pct": dvh_targets["bladder_v50_target_pct"],
        },
    }


def run_pure_mimicking(
    patient_dir,
    pred_npz_path,
    rx_gy_total: float,
    n_fracciones: int,
    struct_names=None,
    num_steps: int = 5,
    lbfgs_kwargs: dict = None,
    deliv_weight: float = 0.0,
    dvh_weight: float = 0.0,
    closed_start: bool = False,
    ptv_conform_start: bool = False,
    leaf_margin_mm: float = 2.0,
    jaw_margin_mm: float = 2.0,
    leaf_max_step_mm: float = 10.0,
    optimize_jaws: bool = False,
    struct_weights: dict = None,
    optimizer_name: str = "lbfgs",
    adam_lr: float = 0.01,
    project_velocity: bool = False,
    project_aperture: bool = False,
    aperture_extra_mm: float = APERTURE_EXTRA_MARGIN_MM,
    project_arc_range: bool = False,
    max_leaf_travel_cm: float = None,
    beam_chunk_size: int = 1,
    max_wall_time_s: float = None,
    on_step=None,
    device=None,
    dtype: torch.dtype = torch.float32,
):
    """Mimicking: loss = MSE(dosis_PDRT, dosis_U-Net) dentro de BODY, sumando
    la dosis de todos los arcos reales del paciente, más (si `deliv_weight`>0)
    la regularización de entregabilidad `deliverability_loss` (paso 1 de
    PIPELINE_KBP_PDRT.md — penaliza moverse más rápido que el plan clínico
    real en cada tramo del arco, ver esa función para el porqué). Con
    `deliv_weight=0.0` (default) es el mimicking puro, sin ese término — NO
    genera un RP deliverable/importable en Eclipse (ver docstring del
    módulo), usar sólo para validar que el loop de dosis converge.

    `dvh_weight` (default 0.0 = desactivado): agrega `dvh_loss` — D98(PTV_High)
    "at_least" y V70(Rectum)/V50(Bladder) "at_most" contra los valores de esos
    mismos puntos calculados sobre la dosis TARGET de U-Net (`compute_dvh_targets`,
    fijos para toda la corrida). Ver docstring del módulo para el hallazgo que
    motivó esto (v5: mse bajo pero D98/V70 lejos del target) y el docstring de
    `dvh_loss` para el ojo con la convención de percentil de `dvh_Dp_loss`.
    Los pesos internos (`DVH_VIOL_WEIGHT_DP/VX`) son un punto de partida sin
    calibrar, igual que `deliv_weight=10.0` cuando se introdujo — puede hacer
    falta ajustar `dvh_weight` viendo cómo compite contra el mse en el log.

    `closed_start=True`: arranca cada arco con MLC cerrado/jaws abiertos en
    vez de la apertura real (ver `OptimizableArc`) — simula el caso real de
    despliegue sin RP del paciente. En la práctica quedó clavado casi de
    entrada (ver `OptimizableArc`) — por eso `ptv_conform_start=True`:
    arranca conformando MLC/jaws a la proyección BEV real del PTV en cada
    control point (con `leaf_margin_mm`/`jaw_margin_mm` de margen), en vez de
    cerrado/abierto a ciegas. Ambos son mutuamente excluyentes con el warm
    start (si `closed_start` es True, `ptv_conform_start` se ignora).
    `deliverability_loss` sigue necesitando la apertura real como referencia
    (no hay todavía una alternativa sin RP), así que con estos modos usar
    `deliv_weight=0.0` — sólo evalúan tiempo/convergencia de la dosis, no
    entregabilidad.

    `optimize_jaws=False` (default): la mordaza Y queda fija (ver
    `OptimizableArc`) — el equipo real no tiene jaw tracking.

    `struct_weights`: pesos por estructura para el MSE de dosis (default
    `STRUCT_WEIGHTS_DEFAULT`, ver ahí por qué el peso uniforme era un
    problema). El MSE pasa a ser un promedio PONDERADO, así que su escala
    cambia respecto de corridas anteriores — `deliv_weight` hay que
    recalibrarlo, los valores de corridas viejas no son comparables.

    `beam_chunk_size` (default 1, igual que antes): en PyDoseRT cada "beam"
    del motor es en realidad un control point del arco — con
    `beam_chunk_size=1` el motor procesa los ~178 CPs de a uno, secuencial
    (gradient-checkpointing por CP), en vez de en un solo batch grande.
    Subirlo (según memoria de GPU disponible) puede acelerar mucho el
    cómputo por paso — no confirmado todavía cuánta memoria hace falta por
    unidad de chunk en este dose_grid_shape/kernel_size.

    `max_wall_time_s`: si se pasa, corta el loop (después de terminar el
    paso en curso, nunca a mitad de un `optimizer.step()`) en cuanto el
    tiempo acumulado de pasos lo supera — para corridas largas exploratorias
    con presupuesto de tiempo fijo en vez de número de pasos fijo.
    `on_step(step_index, history_entry, setup_time_s, optim_arcs)`: si se pasa,
    se llama después de cada paso — para loguear a disco de forma incremental
    y no perder todo si el proceso se corta antes de terminar, y para poder
    escribir un RP intermedio (por eso recibe `optim_arcs`) sin tener que
    matar la corrida para ver un resultado parcial.

    Devuelve (optim_arcs, dose_pred_final, history) — `history` es una lista
    de dicts {mse, deliv, total} por cada llamada a `optimizer.step()`.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_setup_start = time.perf_counter()

    patient, arcs = load_patient_and_arcs(
        patient_dir, struct_names=struct_names or STRUCT_NAMES_DEFAULT, device=device, dtype=dtype,
    )

    dose_target_np, _ = unet_dose_to_pdrt_grid(patient_dir, pred_npz_path, rx_gy_total, n_fracciones)
    dose_target = torch.from_numpy(dose_target_np).to(device=device, dtype=dtype)

    engine = build_dose_engine(patient, beam_template=arcs[0], device=device, dtype=dtype)
    engine.train()
    # Necesaria siempre: `deliverability_loss` saca de acá los límites
    # cinemáticos reales de la máquina (ver reference_real_machine_limits).
    machine_config = load_machine_config()

    if ptv_conform_start:
        ptv_mask = patient.structures.get("PTV_High")
        if ptv_mask is None:
            raise ValueError(
                "ptv_conform_start=True necesita 'PTV_High' en struct_names/patient.structures."
            )
        leaf_widths = machine_config.leaf_widths
        optim_arcs = [
            OptimizableArc(
                a, ptv_conform_start=True, ptv_mask=ptv_mask, resolution=patient.resolution,
                leaf_widths=leaf_widths, leaf_margin_mm=leaf_margin_mm, jaw_margin_mm=jaw_margin_mm,
                leaf_max_step_mm=leaf_max_step_mm, machine_config=machine_config,
                optimize_jaws=optimize_jaws,
            )
            for a in arcs
        ]
    else:
        optim_arcs = [
            OptimizableArc(a, closed_start=closed_start, optimize_jaws=optimize_jaws) for a in arcs
        ]
    setup_time_s = time.perf_counter() - t_setup_start

    params = [p for oa in optim_arcs for p in oa.parameters()]

    kwargs = dict(
        lr=1.0, max_iter=20, max_eval=25,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        history_size=20, line_search_fn="strong_wolfe",
    )
    if lbfgs_kwargs:
        kwargs.update(lbfgs_kwargs)
    inner_iters = int(kwargs.get("max_iter", 20))

    usa_adam = optimizer_name.lower() == "adam"
    if usa_adam:
        # Adam normaliza el paso por parámetro (por la media móvil del cuadrado
        # del gradiente). Eso es exactamente lo que hace falta acá: medido en el
        # init, cada parámetro de MU tiene ~100x más gradiente que cada
        # parámetro de lámina (1.36e-2 vs 1.45e-4), así que LBFGS -- que elige
        # UN largo de paso para todo el vector -- mueve las MU ~100x más que las
        # láminas. Resultado con LBFGS: bajó las MU 13% y dejó el MLC casi
        # intacto (0.47mm de cambio medio en 38 pasos), o sea un plan sin
        # modulación. Ver project_mimicking_pipeline_status.md.
        optimizer = torch.optim.Adam(params, lr=adam_lr)
    else:
        optimizer = torch.optim.LBFGS(params, **kwargs)

    # patient.* queda en CPU tras load_dicom (sólo los BeamSequence se mueven a
    # `device` ahí) — hay que pasarlo explícitamente, igual que ya hace
    # commissioning/recompute_check.py.
    body_mask = patient.structures["BODY"].to(device)
    density_image = patient.density_image.to(device=device, dtype=dtype).unsqueeze(0)

    # MSE ponderado por estructura (ver build_mse_weights). Se precomputan
    # target y pesos ya indexados por BODY — son constantes, no hace falta
    # recalcularlos en cada evaluación del closure.
    weights_vol, w_resumen = build_mse_weights(
        patient, struct_weights or STRUCT_WEIGHTS_DEFAULT, device, dtype
    )
    body_bool = body_mask > 0
    w_body = weights_vol[body_bool]
    w_sum = w_body.sum()
    tgt_body = dose_target[body_bool]
    print("[mimicking] pesos del MSE por estructura (share = % del presupuesto de la loss):")
    for name, n, wt in ((r[0], r[1], r[2]) for r in w_resumen):
        share = next(r[3] for r in w_resumen if r[0] == name)
        print(f"    {name:20s} peso={wt:7.1f}  vox={n:>9,}  share={share:5.1f}%")

    # Targets DVH fijos para toda la corrida (calculados sobre dose_target,
    # no sobre dose_pred) — ver compute_dvh_targets/dvh_loss.
    rx_frac_gy = rx_gy_total / n_fracciones
    dvh_targets = compute_dvh_targets(dose_target, patient, rx_frac_gy)
    print("[mimicking] targets DVH (calculados sobre la dosis target de U-Net):")
    print(f"    PTV_High D98   = {dvh_targets['ptv_d98_gy']:.3f} Gy/fx "
          f"({100 * dvh_targets['ptv_d98_gy'] / rx_frac_gy:.1f}% de Rx/fx)")
    print(f"    Rectum   V70%Rx = {dvh_targets['rectum_v70_target_pct']:.1f}%  "
          f"(umbral {dvh_targets['rectum_v70_thr_gy']:.3f} Gy/fx)")
    print(f"    Bladder  V50%Rx = {dvh_targets['bladder_v50_target_pct']:.1f}%  "
          f"(umbral {dvh_targets['bladder_v50_thr_gy']:.3f} Gy/fx)")

    state = {}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        dose_pred = None
        for oa in optim_arcs:
            bs = oa.to_beam_sequence()
            d = engine.compute_dose(bs, density_image=density_image, beam_chunk_size=beam_chunk_size)
            dose_pred = d if dose_pred is None else dose_pred + d
        dose_pred = dose_pred[0]

        mse = (w_body * (dose_pred[body_bool] - tgt_body) ** 2).sum() / w_sum
        # Los diagnósticos de entregabilidad/DVH se calculan SIEMPRE (son
        # baratos comparados con la dosis) aunque sus pesos sean 0 — así se
        # sabe si el plan es entregable/cubre el DVH incluso en corridas sin
        # esos términos activos.
        dl = deliverability_loss(optim_arcs, machine_config)
        dh = dvh_loss(dose_pred, patient, dvh_targets)
        deliv = dl["total"] if deliv_weight > 0.0 else dl["total"].detach()
        dvh_term = dh["total"] if dvh_weight > 0.0 else dh["total"].detach()
        loss = mse
        if deliv_weight > 0.0:
            loss = loss + deliv_weight * deliv
        if dvh_weight > 0.0:
            loss = loss + dvh_weight * dvh_term
        loss.backward()

        state["mse"] = float(mse.detach())
        state["deliv"] = float(dl["total"].detach())
        state["diag"] = dl["diag"]
        state["dvh"] = float(dh["total"].detach())
        state["dvh_diag"] = dh["diag"]
        state["loss"] = float(loss.detach())
        state["dose_pred"] = dose_pred.detach()
        return loss

    history = []
    cumulative_time_s = 0.0
    for step in range(num_steps):
        t_step_start = time.perf_counter()
        if usa_adam:
            # Un "paso" = `inner_iters` iteraciones de Adam, para que sea
            # comparable en cómputo a un optimizer.step() de LBFGS con
            # max_iter=inner_iters (y el logging/tiempos midan lo mismo).
            for _ in range(inner_iters):
                closure()
                optimizer.step()
                if project_velocity:
                    _proyectar_todo(optim_arcs, machine_config,
                                    project_aperture, aperture_extra_mm, project_arc_range)
        else:
            optimizer.step(closure)
            if project_velocity:
                _proyectar_todo(optim_arcs, machine_config,
                                project_aperture, aperture_extra_mm, project_arc_range)
        step_time_s = time.perf_counter() - t_step_start
        cumulative_time_s += step_time_s
        dg = state["diag"]
        dvhd = state["dvh_diag"]
        history.append({
            "mse": state["mse"], "deliv": state["deliv"], "dvh": state["dvh"], "total": state["loss"],
            "step_time_s": step_time_s, "cumulative_time_s": cumulative_time_s,
            "leaf_viol_max_mm": dg["leaf_viol_max_mm"], "leaf_viol_n": dg["leaf_viol_n"],
            "leaf_step_frac_max": dg["leaf_step_frac_max"],
            "leaf_travel_cm": dg["leaf_travel_cm"], "leaf_step_p90_mm": dg["leaf_step_p90_mm"],
            "mu_per_deg_min": dg["mu_per_deg_min"], "mu_per_deg_viol_n": dg["mu_per_deg_viol_n"],
            "ptv_d98_actual_gy": dvhd["ptv_d98_actual_gy"], "ptv_d98_target_gy": dvhd["ptv_d98_target_gy"],
            "rectum_v70_actual_pct": dvhd["rectum_v70_actual_pct"], "rectum_v70_target_pct": dvhd["rectum_v70_target_pct"],
            "bladder_v50_actual_pct": dvhd["bladder_v50_actual_pct"], "bladder_v50_target_pct": dvhd["bladder_v50_target_pct"],
        })
        print(
            f"[mimicking] step {step + 1}/{num_steps}  "
            f"mse(pond)={state['mse']:.6f}  deliv_raw={state['deliv']:.6f}  dvh_raw={state['dvh']:.6f}  "
            f"total={state['loss']:.6f}  | ENTREGABILIDAD: exceso_lamina_max="
            f"{dg['leaf_viol_max_mm']:.3f}mm ({dg['leaf_viol_n']} pares)  "
            f"recorrido_laminas={dg['leaf_travel_cm']:.0f}cm (clinico ~3100)  "
            f"p90={dg['leaf_step_p90_mm']:.2f}mm  uso_max={100 * dg['leaf_step_frac_max']:.1f}%  "
            f"MU/deg_min={dg['mu_per_deg_min']:.3f} ({dg['mu_per_deg_viol_n']} fuera)  "
            f"| DVH: D98_PTV={dvhd['ptv_d98_actual_gy']:.3f}/{dvhd['ptv_d98_target_gy']:.3f}Gy  "
            f"V70_Recto={dvhd['rectum_v70_actual_pct']:.1f}/{dvhd['rectum_v70_target_pct']:.1f}%  "
            f"V50_Vejiga={dvhd['bladder_v50_actual_pct']:.1f}/{dvhd['bladder_v50_target_pct']:.1f}%  "
            f"| tiempo_paso={step_time_s:.1f}s  "
            f"tiempo_acumulado={cumulative_time_s:.1f}s"
        )
        if on_step is not None:
            # Se pasan tambien los arcos para que el driver pueda escribir un
            # RP intermedio sin esperar a que termine la corrida (el estado del
            # optimizador vive solo en memoria; sin esto, cortar la corrida
            # para mirar un resultado parcial perdia todo el avance).
            on_step(step, history[-1], setup_time_s, optim_arcs)
        if max_leaf_travel_cm is not None and dg["leaf_travel_cm"] >= max_leaf_travel_cm:
            # Parada por COMPLEJIDAD, no por dosis: apuntar a un plan con
            # recorrido de láminas parecido al clínico en vez de a mse mínimo.
            # El mse mínimo es engañoso — se logra sobremodulando, y ahí PDRT
            # deja de coincidir con AAA (ver project_mimicking_pipeline_status).
            print(
                f"[mimicking] recorrido de láminas alcanzó el tope "
                f"({dg['leaf_travel_cm']:.0f}cm >= {max_leaf_travel_cm:.0f}cm) — "
                f"cortando en el paso {step + 1}/{num_steps}."
            )
            break
        if max_wall_time_s is not None and cumulative_time_s >= max_wall_time_s:
            print(
                f"[mimicking] presupuesto de tiempo agotado ({cumulative_time_s:.1f}s >= "
                f"{max_wall_time_s:.1f}s) — cortando en el paso {step + 1}/{num_steps}."
            )
            break

    print(f"[mimicking] tiempo de setup (carga paciente/target/engine)={setup_time_s:.1f}s")

    return optim_arcs, state["dose_pred"], history, setup_time_s
