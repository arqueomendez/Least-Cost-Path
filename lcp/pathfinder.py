# lcp/pathfinder.py

import os

import numpy as np
from numba import njit


@njit
def reconstruct_path_from_traceback(traceback, start_pixel, end_pixel):
    """
    Reconstruye una ruta utilizando el ráster de traceback DIRECCIONAL de scikit-image.
    Cada celda almacena el código de dirección (0-7) que indica desde qué vecino llegó
    el frente de onda.  Para seguir el camino hacia atrás, restamos el delta.

    El buffer de path se dimensiona como rows + cols, que es la cota superior
    práctica de cualquier camino sin ciclos en una cuadrícula.  Esto evita la
    asignación de arrays de tamaño traceback.size (~GB para rasters grandes).
    """
    # Mapeo de los códigos de dirección (0-7) de scikit-image a cambios en (fila, columna)
    # Índice: 0   1   2   3   4   5   6   7
    # Vecino: NW, N, NE, W, E, SW, S, SE
    dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
    dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int32)

    curr_row, curr_col = end_pixel
    start_row, start_col = start_pixel

    # Cota superior real de un camino en una cuadrícula: filas + columnas.
    # Muy inferior a traceback.size (~158M para el raster completo).
    max_path_len = traceback.shape[0] + traceback.shape[1]
    path = np.zeros((max_path_len, 2), dtype=np.int32)
    count = 0
    max_steps = max_path_len

    while not (curr_row == start_row and curr_col == start_col) and count < max_steps:
        path[count] = np.array([curr_row, curr_col])
        count += 1

        direction_code = traceback[curr_row, curr_col]

        if direction_code < 0:
            return None

        # El traceback en (r, c) indica la dirección DESDE la que se llegó.
        # Para volver al padre, nos movemos en la dirección OPUESTA (restamos el delta).
        curr_row = curr_row - dr[direction_code]
        curr_col = curr_col - dc[direction_code]

    if count >= max_steps:
        return None

    path[count] = np.array([start_row, start_col])
    return path[: count + 1][::-1]


def compute_and_save_cost_surface(args):
    """
    Worker de Fase 1: calcula MCP desde un origen, extrae los costes escalares
    para cada destino y guarda el traceback comprimido en disco.

    Parámetros (empaquetados en args):
        origin_id       : identificador del punto de origen.
        origin_pixel    : tupla (row, col) del origen en el raster.
        dest_pixels     : dict {dest_id: (row, col)} de todos los destinos > origin_id.
        shared_data     : dict con memmap_path, raster_dtype, raster_shape, src_transform
                          y, opcionalmente, bbox_buffer (int, píxeles de margen alrededor
                          del bounding-box de origen+destinos; None = raster completo).
        phase1_dir      : ruta al directorio donde se guardarán los .npz de traceback.

    Retorna:
        (origin_id, cost_scalars) donde cost_scalars es {dest_id: float}.
        En caso de error retorna (origin_id, {}).

    Optimizaciones aplicadas:
        1. Recorte de bounding-box: el raster se recorta al rectángulo mínimo que
           contiene el origen y todos los destinos, más un margen ``bbox_buffer``.
           Reduce el uso de RAM por worker y el tiempo de Dijkstra en proporción
           al área recortada (puede ser 10-100× para rasters grandes con puntos
           concentrados).
        2. Terminación temprana: se pasa ``ends`` a ``find_costs`` para que Dijkstra
           se detenga en cuanto haya liquidado todos los píxeles destino, en lugar
           de explorar el raster entero.
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
            # --- Recorte de bounding-box ---
            all_pixels = [origin_pixel] + list(dest_pixels.values())
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            H, W = cost_raster_full.shape
            r_min = max(0, min(rows) - bbox_buffer)
            r_max = min(H, max(rows) + bbox_buffer + 1)
            c_min = max(0, min(cols) - bbox_buffer)
            c_max = min(W, max(cols) + bbox_buffer + 1)

            # Copia contigua del sub-raster; libera la referencia al memmap completo.
            cost_raster = np.ascontiguousarray(
                cost_raster_full[r_min:r_max, c_min:c_max]
            )
            del cost_raster_full

            # Re-indexar todos los píxeles al sistema de coordenadas del sub-raster.
            origin_pixel_local = (origin_pixel[0] - r_min, origin_pixel[1] - c_min)
            dest_pixels_local = {
                did: (r - r_min, c - c_min) for did, (r, c) in dest_pixels.items()
            }
        else:
            # Sin recorte: comportamiento original (raster completo).
            r_min, c_min = 0, 0
            cost_raster = np.ascontiguousarray(cost_raster_full)
            del cost_raster_full
            origin_pixel_local = origin_pixel
            dest_pixels_local = dict(dest_pixels)

        # --- MCP con terminación temprana en los píxeles destino ---
        mcp = MCP_Geometric(cost_raster, fully_connected=True)
        ends = list(dest_pixels_local.values())
        costs, traceback = mcp.find_costs(
            [origin_pixel_local], ends=ends, find_all_ends=True
        )
        del cost_raster, mcp

        # Extraer escalares de coste para cada destino sin persistir el array completo.
        cost_scalars = {}
        for dest_id, (r, c) in dest_pixels_local.items():
            cost_scalars[dest_id] = float(costs[r, c])
        del costs

        # Guardar traceback comprimido junto con los offsets del recorte.
        # row_offset / col_offset permiten que la Fase 2 convierta las coordenadas
        # del sub-raster de vuelta a coordenadas del raster completo.
        npz_path = os.path.join(phase1_dir, f"traceback_{origin_id}.npz")
        np.savez_compressed(
            npz_path,
            traceback=traceback,
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
