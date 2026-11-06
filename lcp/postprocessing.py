# lcp/postprocessing.py

import os

# Mensaje de error comun de Fase 2 (misma forma para las dos direcciones).
_ERROR_FASE2 = "ERROR Fase 2, origen {id}: {exc}"


def _load_traceback(npz_path):
    """
    Carga el traceback de Fase 1 desde disco y devuelve
    (traceback, row_offset, col_offset, offsets).
    """
    import numpy as np

    import lcp.pathfinder as pf

    # Cerrar el NpzFile tras leer (np.load es perezoso: deja el ZIP abierto y
    # acumularía descriptores -> "Too many open files" con miles de orígenes).
    with np.load(npz_path) as data:
        traceback = data["traceback"].copy()
        row_offset = int(data["row_offset"]) if "row_offset" in data.files else 0
        col_offset = int(data["col_offset"]) if "col_offset" in data.files else 0
        if "offsets" in data.files:
            offsets = data["offsets"].astype(np.int32)
        else:
            # Respaldo para .npz generados por una versión anterior.
            offsets = pf.DEFAULT_OFFSETS.copy()
    return traceback, row_offset, col_offset, offsets


def _reconstruct_records(args, inverted):
    """
    Núcleo compartido de Fase 2.

    ``args`` = (root_id, root_coords, target_items, cost_scalars, transform,
    phase1_dir[, (origin_layer, dest_layer)]), donde ``root_id`` es la raíz del
    MCP ya calculado. Con ``inverted=False`` la raíz es el origen y los targets
    son los destinos; con ``inverted=True`` la raíz es un destino (punto de la
    capa de fuentes) y los targets son los orígenes (puntos de la capa de
    sitios), de modo que la ruta se reconstruye raíz->target, se invierte y se
    registra con la semántica sitio->fuente.

    S9: se descarta todo registro cuyo coste no sea finito. Las celdas que el
    MCP no finalizó dejan el traceback sin predecesor real y, para un destino al
    noroeste del origen sobre la misma diagonal, la reconstrucción devuelve una
    ruta recta falsa con ``cost_total = inf``. Antes se escribía tal cual.
    """
    import numpy as np

    import lcp.pathfinder as pf
    import lcp.processing as proc
    import lcp.utils as utils

    root_id, root_coords, target_items, cost_scalars, transform, phase1_dir = args[:6]
    layers = args[6] if len(args) > 6 else None
    if layers is None:
        origin_layer, dest_layer = None, None
    else:
        # Orden semántico del llamador: (capa de origenes, capa de destinos).
        origin_layer, dest_layer = layers

    npz_path = os.path.join(phase1_dir, f"traceback_{root_id}.npz")
    traceback, row_offset, col_offset, offsets = _load_traceback(npz_path)

    start_pixel = proc.world_to_pixel(transform, root_coords[0], root_coords[1])
    start_pixel_local = (start_pixel[0] - row_offset, start_pixel[1] - col_offset)

    records = []
    for target_id, target_coords in target_items:
        # S9: coste no finito (inf o nan) = ruta no valida -> no se construye.
        cost_total = float(cost_scalars.get(target_id, float("nan")))
        if not np.isfinite(cost_total):
            continue

        end_pixel = proc.world_to_pixel(transform, target_coords[0], target_coords[1])
        end_pixel_local = (end_pixel[0] - row_offset, end_pixel[1] - col_offset)

        path_pixels_local = pf.reconstruct_path_from_traceback(
            traceback, start_pixel_local, end_pixel_local, offsets
        )

        if path_pixels_local is None:
            continue

        path_pixels = path_pixels_local + np.array(
            [[row_offset, col_offset]], dtype=np.int32
        )

        if inverted:
            # Ruta raíz->target invertida: la polilínea y los ids quedan con la
            # semántica de negocio (origin_id = sitio, dest_id = fuente).
            path_pixels = path_pixels[::-1]
            origin_id, dest_id = target_id, root_id
        else:
            origin_id, dest_id = root_id, target_id

        record = utils.build_route_record(
            path_pixels,
            transform,
            origin_id,
            dest_id,
            cost_total,
            origin_layer,
            dest_layer,
        )
        if record is not None:
            records.append(record)

    del traceback
    return records


def reconstruct_routes_from_disk(args):
    """
    Worker de Fase 2: carga el traceback comprimido desde disco y reconstruye
    todas las rutas desde un origen hacia sus destinos (dest_id > origin_id).

    ``args`` = (origin_id, origin_coords, dest_items, cost_scalars, transform,
    phase1_dir) — el 7.º elemento (origin_layer, dest_layer) es opcional (S6).
    Los destinos con coste no finito se descartan (S9).

    Retorna (origin_id, records) o (origin_id, []) ante error.
    """
    origin_id = args[0]

    try:
        return (origin_id, _reconstruct_records(args, inverted=False))

    except Exception as exc:
        print(_ERROR_FASE2.format(id=origin_id, exc=exc))
        import traceback as tb

        tb.print_exc()
        return (origin_id, [])


def reconstruct_inverted_routes_from_disk(args):
    """
    Worker de Fase 2 con dirección invertida (S3): el MCP se calculó DESDE la
    raíz (un punto de la capa de destinos) hacia los targets (puntos de la capa
    de orígenes). Cada ruta se reconstruye raíz->target, se invierte y se
    registra como origin_id = target (sitio) y dest_id = raíz (fuente).

    ``args`` = (root_id, root_coords, target_items, cost_scalars, transform,
    phase1_dir) y, opcionalmente, (origin_layer, dest_layer) en el 7.º lugar con
    el orden semántico del llamador: capa de los targets primero y capa de la
    raíz después. También descarta los costes no finitos (S9).

    Retorna (root_id, records) o (root_id, []) ante error.
    """
    root_id = args[0]

    try:
        return (root_id, _reconstruct_records(args, inverted=True))

    except Exception as exc:
        print(_ERROR_FASE2.format(id=root_id, exc=exc))
        import traceback as tb

        tb.print_exc()
        return (root_id, [])
