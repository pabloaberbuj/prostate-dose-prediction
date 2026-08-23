"""Inicialización de MLC/jaws conformada al PTV (PIPELINE_KBP_PDRT.md /
project_cold_start_init_todo.md), como alternativa a `closed_start` en
`OptimizableArc` (mimicking.py) — pensada para el caso real de despliegue,
donde no hay un RP de paciente del que arrancar.

Para cada control point del arco (gantry/colimador propios), proyecta la
máscara 3D del PTV al plano 2D de haz's-eye-view (BEV) que usa el motor de
PyDoseRT para definir leaves/jaws, y calcula la apertura que conforma esa
silueta con margen: MLC a `leaf_margin_mm` (default 5mm), jaws a
`jaw_margin_mm` (default 2mm) — igual al criterio que usa Eclipse para
inicializar la optimización de VMAT.

Geometría (deriva EXACTAMENTE la inversa de lo que hace el motor real, leer
antes de tocar esto — un error de signo/eje aquí da una apertura que no
conforma con el PTV sin que se note a simple vista):

  `pydosert.layers.FluenceVolumeLayer.forward` proyecta el mapa de fluencia
  2D (definido en el plano BEV local del haz, ejes W=leaves/MLCX,
  H=jaws/ASYMY, mm centrados en el eje del haz) a 3D usando divergencia
  cónica: para un vóxel a distancia `depth` de la fuente (a lo largo del eje
  D "canónico", con el haz apuntando a gantry=0), la posición que muestrea
  en el mapa de fluencia es `(w_mm, h_mm) * SID/depth` — la fórmula estándar
  de magnificación con la fuente. Acá se invierte: dado un vóxel real del
  PTV, `fluence_w_mm/h_mm = (w_mm, h_mm)_relativo_al_iso * SID/depth`.

  `pydosert.engine.dose_engine.DoseEngine._forward_core` aplica esa
  proyección DESPUÉS de rotar el mapa de fluencia por el ángulo de colimador
  (`rotate_2d_images`), y el volumen resultante (en el frame "canónico",
  como si gantry=0) se rota recién al final por el ángulo de gantry real
  (`BeamRotationLayer`, que usa `build_rotation_grids`). Ambas rotaciones
  usan la misma convención de matriz af��n (mismo signo) — para invertir:
    - Gantry: se rota la máscara 3D REAL del PTV por -gantry_angle (con la
      MISMA función `build_rotation_grids`/`grid_sample` que usa el motor,
      no una reimplementación manual — así no hay riesgo de firmar mal la
      rotación con offset de isocentro) para llevarla al frame canónico.
    - Colimador: dado que en el frame canónico ya se tienen puntos 2D
      discretos (no una imagen a resamplear), se aplica directo la rotación
      de coordenadas equivalente (derivada de la misma convención de matriz)
      para llevarlos al frame local de leaves/jaws.

Validado (ver test manual) comparando contra la apertura real del plan
clínico en sus propios ángulos de gantry/colimador — el resultado cae en el
mismo orden de magnitud y sigue la envolvente de la apertura real.

Post-procesado (2026-08-11, a pedido de Pablo — la proyección BEV pura, CP a
CP independiente, no es suave: `deliverability_loss` (comparada contra un
plan real) medía una violación de ~1.2 ya en el init, antes de optimizar):
  - MLC: se suaviza la secuencia de cada lámina a lo largo del arco con un
    limitador de velocidad (`_rate_limit_sequence`, pasada adelante + atrás)
    que no deja que se mueva más de `leaf_max_step_mm` entre CPs
    consecutivos — acota la distancia entre láminas consecutivas a algo
    "aceptable para iniciar" en vez de la proyección cruda.
  - Jaws: en vez de un valor por CP, se usa UN solo valor para todo el arco
    (igual al equipo real, que no tiene jaw tracking — ver
    `OptimizableArc(optimize_jaws=False)`): el mínimo/máximo de la lámina más
    extendida en el CP más extendido de todo el arco, con margen.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pydosert.geometry.rotations import build_rotation_grids

LEAF_TRAVEL_LIMIT_MM = 200.0  # límite físico del modelo (ver COMISIONAMIENTO_PDRT.md)
JAW_TRAVEL_LIMIT_MM = 200.0


def _crop_bounds(mask: torch.Tensor, iso_center, resolution, buffer_mm: float = 20.0):
    """Cubo centrado en el ÍNDICE de vóxel del isocentro, con radio = máxima
    distancia (mm) de cualquier vóxel del PTV al isocentro + buffer. La
    rotación de `build_rotation_grids` es siempre alrededor del isocentro,
    así que a cualquier ángulo los vóxels del PTV rotados siguen a la misma
    distancia del iso — este cubo los contiene sea cual sea el ángulo.
    """
    H, D, W = mask.shape
    idx = torch.nonzero(mask > 0.5, as_tuple=False)  # [n, 3] en (h,d,w)
    if idx.numel() == 0:
        raise ValueError("Máscara de PTV vacía — no se puede conformar la apertura.")

    res = torch.tensor(resolution, dtype=torch.float32)
    iso = torch.tensor(iso_center, dtype=torch.float32)
    mm = idx.to(torch.float32) * res.view(1, 3) - iso.view(1, 3)  # [n,3] (h,d,w) mm relativo al iso
    radius_mm = float(torch.sqrt((mm ** 2).sum(dim=1)).max()) + buffer_mm

    iso_idx = iso / res  # posición del iso en índice de vóxel (float), por eje
    half = torch.ceil(radius_mm / res).to(torch.int64)  # radio en vóxeles, por eje

    starts, ends = [], []
    for k, size in enumerate((H, D, W)):
        c = int(round(float(iso_idx[k])))
        s = max(0, c - int(half[k]))
        e = min(size, c + int(half[k]) + 1)
        starts.append(s)
        ends.append(e)
    return tuple(starts), tuple(ends)


def _rotate_mask_to_canonical(mask_crop: torch.Tensor, gantry_angle_rad: float,
                               iso_center_crop, resolution, device, dtype) -> torch.Tensor:
    """Rota la máscara (recortada) por -gantry_angle usando exactamente
    `build_rotation_grids` (mismo código que `BeamRotationLayer`), para
    llevarla del frame real al frame canónico (como si gantry=0). Ver
    docstring del módulo para la derivación del signo.
    """
    H, D, W = mask_crop.shape
    angles = torch.tensor([-gantry_angle_rad], dtype=dtype, device=device)
    grid = build_rotation_grids(
        (1, 1, D, H, W), angles, device, dtype,
        iso_center=iso_center_crop, resolution=resolution,
    )  # [1, 1, 1, D, W, 2]
    grid = grid.repeat(1, 1, H, 1, 1, 1).reshape(H, D, W, 2)

    vol = mask_crop.to(device=device, dtype=dtype).unsqueeze(1)  # (H,D,W) -> [H,1,D,W]
    rotated = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return rotated.squeeze(1)  # [H,D,W], frame canónico


def wrapped_gantry_diff_deg(gantry_angles_rad: torch.Tensor) -> torch.Tensor:
    """|delta| entre ángulos de gantry consecutivos (grados), corrigiendo el
    salto artificial al cruzar el límite 0/360 (359 -> 1 es un giro de 2, no
    de 358). Devuelve [CP-1]. Vive acá y no en mimicking.py para que las dos
    lo compartan sin import circular (mimicking ya importa de este módulo).
    """
    deg = torch.rad2deg(gantry_angles_rad)
    diff = torch.diff(deg)
    return (torch.remainder(diff + 180.0, 360.0) - 180.0).abs()


def _rate_limit_sequence(x: torch.Tensor, max_step, n_passes: int = 2) -> torch.Tensor:
    """Limita el cambio entre elementos consecutivos de `x` ([CP, ...]) a lo
    largo del eje 0, sin alejarse más de lo necesario del valor original --
    pasada hacia adelante y hacia atrás (`n_passes` veces cada una) en vez de
    una sola dirección, para no dejar un "rezago" permanente del lado en que
    arranca la pasada.

    `max_step` puede ser un escalar o un tensor [CP-1] con el tope de cada
    segmento (el caso real: el presupuesto físico depende del delta de gantry
    de cada segmento, que no es uniforme -- el primero y el último del arco
    son medio paso).
    """
    x = x.clone()
    n_cp = x.shape[0]
    escalar = not torch.is_tensor(max_step)

    def paso(i_seg):
        return max_step if escalar else max_step[i_seg]

    for _ in range(n_passes):
        for i in range(1, n_cp):
            s = paso(i - 1)
            x[i] = torch.clamp(x[i], x[i - 1] - s, x[i - 1] + s)
        for i in range(n_cp - 2, -1, -1):
            s = paso(i)
            x[i] = torch.clamp(x[i], x[i + 1] - s, x[i + 1] + s)
    return x


def compute_ptv_conforming_aperture(
    ptv_mask: torch.Tensor,
    gantry_angles_rad: torch.Tensor,
    collimator_angles_rad: torch.Tensor,
    iso_center,
    sid: float,
    resolution,
    leaf_widths: list,
    field_size,
    leaf_margin_mm: float = 2.0,
    jaw_margin_mm: float = 2.0,
    leaf_min_opening_mm: float = 1.0,
    leaf_max_step_mm: float = None,
    max_leaf_speed_mm_s: float = 22.5,
    max_gantry_speed_deg_s: float = 4.8,
    speed_safety: float = 0.95,
    mask_threshold: float = 0.1,
    device=None,
    dtype: torch.dtype = torch.float32,
):
    """Apertura de MLC/jaws que conforma el PTV (+margen) en cada control
    point de un arco, proyectando la máscara 3D real a BEV por gantry/colimador,
    con las láminas suavizadas a lo largo del arco (`leaf_max_step_mm`, ver
    `_rate_limit_sequence`) y una mordaza única para todo el arco (sin jaw
    tracking, ver docstring del módulo).

    Args:
        ptv_mask: [H, D, W] (misma grilla que patient.density_image).
        gantry_angles_rad, collimator_angles_rad: [CP], ángulos reales del arco.
        iso_center: (H, D, W) mm — igual convención que BeamSequence.iso_center.
        sid, resolution, leaf_widths, field_size: igual que el motor comisionado.
        max_leaf_speed_mm_s, max_gantry_speed_deg_s, speed_safety: el suavizado
            usa el MISMO presupuesto físico por segmento que
            `mimicking.deliverability_loss`
            (`max_leaf_speed * delta_gantry / max_gantry_speed * safety`), para
            que no haya dos números que puedan quedar desincronizados. Los
            defaults son los límites reales de esta máquina (ver
            reference_real_machine_limits: 22.5 mm/s es el efectivo de Eclipse
            en arco dinámico, inferido de sus propios avisos, NO los 25.0 mm/s
            nominales de la pantalla del MLC).
        leaf_max_step_mm: tope absoluto EXTRA opcional (mm). None = sólo el
            presupuesto físico. Antes era un valor plano de 10mm, que resultó
            ser mayor que el presupuesto real (~9.1mm en el segmento típico de
            2.03 deg) y por lo tanto no entregable.

    Returns:
        (leaf_positions [CP, N, 2], jaw_positions [CP, 2]) — mm, mismo
        formato que `BeamSequence.leaf_positions/jaw_positions`. `jaw_positions`
        repite el mismo (lo, hi) en las CP filas (mordaza fija, ver arriba).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(leaf_widths)
    H_field, W_field = field_size

    leaf_widths_t = torch.tensor(leaf_widths, dtype=torch.float64)
    total_length = float(leaf_widths_t.sum())
    band_start = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.float64), leaf_widths_t[:-1]]), dim=0) - total_length / 2
    band_end = band_start + leaf_widths_t  # [N], mm, centrado en 0 (misma convención que jaw_indices)

    (h0, d0, w0), (h1, d1, w1) = _crop_bounds(ptv_mask, iso_center, resolution)
    mask_crop = ptv_mask[h0:h1, d0:d1, w0:w1].to(device=device, dtype=dtype)
    iso_center_crop = (
        iso_center[0] - h0 * resolution[0],
        iso_center[1] - d0 * resolution[1],
        iso_center[2] - w0 * resolution[2],
    )

    n_cp = int(gantry_angles_rad.shape[0])
    leaf_positions = torch.zeros((n_cp, N, 2), dtype=torch.float32)
    jaw_positions = torch.zeros((n_cp, 2), dtype=torch.float32)

    Hc, Dc, Wc = mask_crop.shape
    h_idx = torch.arange(Hc, dtype=torch.float64)
    d_idx = torch.arange(Dc, dtype=torch.float64)
    w_idx = torch.arange(Wc, dtype=torch.float64)

    for cp in range(n_cp):
        canonical = _rotate_mask_to_canonical(
            mask_crop, float(gantry_angles_rad[cp]), iso_center_crop, resolution, device, dtype,
        )
        on = torch.nonzero(canonical > mask_threshold, as_tuple=False)  # [n, 3] (h,d,w) idx en el crop
        if on.numel() == 0:
            # Ningún vóxel del PTV visible en este CP (no debería pasar si el
            # crop está bien dimensionado) -> deja el leaf/jaw cerrado.
            leaf_positions[cp, :, 0] = 0.0
            leaf_positions[cp, :, 1] = 0.0
            jaw_positions[cp, 0] = 0.0
            jaw_positions[cp, 1] = 0.0
            continue

        on = on.to(torch.float64).cpu()
        h_mm = on[:, 0] * resolution[0] - iso_center_crop[0]
        d_mm = on[:, 1] * resolution[1] - iso_center_crop[1]
        w_mm = on[:, 2] * resolution[2] - iso_center_crop[2]
        # FluenceVolumeLayer: depths[d] = SID - iso_center[1] + d*resolution[1],
        # con d el ÍNDICE de vóxel (no la posición relativa al iso). Como
        # d*resolution[1] = d_mm + iso_center_crop[1], depth = SID + d_mm.
        depth = sid + d_mm

        fluence_w_mm = w_mm * sid / depth
        fluence_h_mm = h_mm * sid / depth

        colim = float(collimator_angles_rad[cp])
        cos_c, sin_c = torch.cos(torch.tensor(colim, dtype=torch.float64)), torch.sin(torch.tensor(colim, dtype=torch.float64))
        local_w = cos_c * fluence_w_mm + sin_c * fluence_h_mm
        local_h = -sin_c * fluence_w_mm + cos_c * fluence_h_mm

        jaw_lo = float(local_h.min()) - jaw_margin_mm
        jaw_hi = float(local_h.max()) + jaw_margin_mm
        jaw_positions[cp, 0] = max(-JAW_TRAVEL_LIMIT_MM, jaw_lo)
        jaw_positions[cp, 1] = min(JAW_TRAVEL_LIMIT_MM, jaw_hi)

        for n in range(N):
            in_band = (local_h >= band_start[n]) & (local_h <= band_end[n])
            if not bool(in_band.any()):
                leaf_positions[cp, n, 0] = 0.0
                leaf_positions[cp, n, 1] = 0.0
                continue
            left = float(local_w[in_band].min()) - leaf_margin_mm
            right = float(local_w[in_band].max()) + leaf_margin_mm
            left = max(-LEAF_TRAVEL_LIMIT_MM, left)
            right = min(LEAF_TRAVEL_LIMIT_MM, right)
            if right - left < leaf_min_opening_mm:
                center = 0.5 * (left + right)
                left = center - leaf_min_opening_mm / 2
                right = center + leaf_min_opening_mm / 2
            leaf_positions[cp, n, 0] = left
            leaf_positions[cp, n, 1] = right

    # Mordaza única para todo el arco (sin jaw tracking, ver docstring del
    # módulo): el mínimo/máximo entre TODOS los CPs -- nunca clipea la
    # lámina más extendida del CP más extendido de todo el arco.
    jaw_lo_global = float(jaw_positions[:, 0].min())
    jaw_hi_global = float(jaw_positions[:, 1].max())
    jaw_positions[:, 0] = jaw_lo_global
    jaw_positions[:, 1] = jaw_hi_global

    # Suavizado de la secuencia de cada lámina a lo largo del arco -- la
    # proyección BEV pura no es suave CP a CP (ver docstring del módulo).
    # Presupuesto FÍSICO por segmento, el mismo que usa deliverability_loss.
    dgantry = wrapped_gantry_diff_deg(gantry_angles_rad).to(torch.float32).cpu()
    paso_max = max_leaf_speed_mm_s * (dgantry / max_gantry_speed_deg_s) * speed_safety
    if leaf_max_step_mm is not None:
        paso_max = torch.clamp(paso_max, max=float(leaf_max_step_mm))
    paso_max = paso_max.unsqueeze(-1)  # [CP-1, 1] -> broadcast sobre láminas
    leaf_positions[:, :, 0] = _rate_limit_sequence(leaf_positions[:, :, 0], paso_max)
    leaf_positions[:, :, 1] = _rate_limit_sequence(leaf_positions[:, :, 1], paso_max)
    # El suavizado (clamps independientes en left/right) no garantiza en
    # todos los casos left<=right-min_opening -- red de seguridad:
    width = leaf_positions[:, :, 1] - leaf_positions[:, :, 0]
    too_narrow = width < leaf_min_opening_mm
    if too_narrow.any():
        center = 0.5 * (leaf_positions[:, :, 0] + leaf_positions[:, :, 1])
        leaf_positions[:, :, 0] = torch.where(too_narrow, center - leaf_min_opening_mm / 2, leaf_positions[:, :, 0])
        leaf_positions[:, :, 1] = torch.where(too_narrow, center + leaf_min_opening_mm / 2, leaf_positions[:, :, 1])

    return leaf_positions, jaw_positions
