# lcp/pathfinder.py

import os

import numpy as np
from numba import njit


# Orden por defecto de los offsets de skimage (producto de [-1,0,1]^2 sin (0,0)).
# Se usa solo como respaldo si un .npz antiguo no almacenó mcp.offsets.
DEFAULT_OFFSETS = np.array(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
    dtype=np.int32,
)


@njit(cache=True)
def reconstruct_path_from_traceback(traceback, start_pixel, end_pixel, offsets):
    """
    Reconstruye una ruta usando el traceback DIRECCIONAL de scikit-image.

    Cada celda almacena un código que indexa ``offsets``; el predecesor de (r, c)
    es (r, c) - offsets[code] (semántica oficial de skimage: si offsets[code] es
    (-1,-1), el predecesor de [x,y] es [x+1,y+1]).

    Implementación en DOS PASADAS:
      - Pasada 1: cuenta los pasos siguiendo el traceback SIN asignar memoria.
      - Pasada 2: asigna un buffer del TAMAÑO EXACTO y lo rellena.

    Esto evita por completo dos defectos:
      1. Reservar traceback.size (~GB para rasters grandes).
      2. La cota errónea (rows + cols), que trunca rutas largas no monótonas
         —las que rodean barreras de alto coste— devolviendo None en silencio.

    La única cota de seguridad es H*W (un árbol de caminos mínimos no revisita
    celdas); alcanzarla implica un traceback corrupto.
    """
    start_row, start_col = start_pixel
    hard_cap = traceback.shape[0] * traceback.shape[1]

    # --- Pasada 1: contar ---
    r, c = end_pixel
    n = 0
    while not (r == start_row and c == start_col) and n < hard_cap:
        code = traceback[r, c]
        if code < 0:
            return None
        r -= offsets[code, 0]
        c -= offsets[code, 1]
        n += 1

    if n >= hard_cap:
        return None

    # --- Pasada 2: rellenar tamaño exacto ---
    path = np.empty((n + 1, 2), dtype=np.int32)
    r, c = end_pixel
    for k in range(n):
        path[k, 0] = r
        path[k, 1] = c
        code = traceback[r, c]
        r -= offsets[code, 0]
        c -= offsets[code, 1]
    path[n, 0] = start_row
    path[n, 1] = start_col

    return path[::-1]


def compute_and_save_cost_surface(args):
    """
    Worker de Fase 1: calcula MCP desde un origen, extrae los costes escalares
    para cada destino y guarda el traceback comprimido (+ los offsets reales del
    MCP) en disco.

    Parámetros (empaquetados en args):
        origin_id, origin_pixel, dest_pixels, shared_data, phase1_dir.
        shared_data acepta opcionalmente 'bbox_buffer' (int) para recortar el
        raster; None = raster completo (recomendado para garantizar optimalidad).

    Retorna (origin_id, cost_scalars) o (origin_id, {}) ante error.
    """
    origin_id, origin_pixel, dest_pixels, shared_data, phase1_dir = args

    try:
        import numpy as np
        from skimage.graph import MCP_Geometric

        cost_raster_full = np.memmap(
            shared_data["memmap_path"],
            dtype=shared_data["raster_dtype"],
            mode="r",
            shape=shared_data["raster_shape"],
        )

        bbox_buffer = shared_data.get("bbox_buffer", None)

        if bbox_buffer is not None and dest_pixels:
            # --- Recorte de bounding-box (NOTA: un buffer fijo NO garantiza la
            #     optimalidad global; el camino mínimo puede salir del recorte). ---
            all_pixels = [origin_pixel] + list(dest_pixels.values())
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            H, W = cost_raster_full.shape
            r_min = max(0, min(rows) - bbox_buffer)
            r_max = min(H, max(rows) + bbox_buffer + 1)
            c_min = max(0, min(cols) - bbox_buffer)
            c_max = min(W, max(cols) + bbox_buffer + 1)

            cost_raster = np.array(
                cost_raster_full[r_min:r_max, c_min:c_max]
            )  # copia explícita
            del cost_raster_full

            origin_pixel_local = (origin_pixel[0] - r_min, origin_pixel[1] - c_min)
            dest_pixels_local = {
                did: (r - r_min, c - c_min) for did, (r, c) in dest_pixels.items()
            }
        else:
            # Sin recorte: copia explícita del raster completo (libera el memmap).
            r_min, c_min = 0, 0
            cost_raster = np.array(cost_raster_full)
            del cost_raster_full
            origin_pixel_local = origin_pixel
            dest_pixels_local = dict(dest_pixels)

        # --- MCP con terminación temprana en los destinos ---
        mcp = MCP_Geometric(cost_raster, fully_connected=True)
        ends = list(dest_pixels_local.values())
        costs, traceback = mcp.find_costs(
            [origin_pixel_local], ends=ends, find_all_ends=True
        )
        # Capturar los offsets REALES del MCP ANTES de destruirlo. Así la Fase 2
        # no depende de un orden de direcciones hardcodeado.
        offsets = np.asarray(mcp.offsets, dtype=np.int32)
        del cost_raster, mcp

        cost_scalars = {}
        for dest_id, (r, c) in dest_pixels_local.items():
            cost_scalars[dest_id] = float(costs[r, c])
        del costs

        npz_path = os.path.join(phase1_dir, f"traceback_{origin_id}.npz")
        np.savez_compressed(
            npz_path,
            traceback=traceback,
            offsets=offsets,
            row_offset=np.int32(r_min),
            col_offset=np.int32(c_min),
        )
        del traceback

        return (origin_id, cost_scalars)

    except Exception as exc:
        print(f"ERROR Fase 1, origen {origin_id}: {exc}")
        import traceback as tb

        tb.print_exc()
        return (origin_id, {})