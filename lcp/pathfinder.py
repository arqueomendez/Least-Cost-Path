# lcp/pathfinder.py

import math
import os
from itertools import pairwise

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

    Todo acceso a ``traceback`` está ACOTADO: se validan los extremos, el código
    de offset (``0 <= code < len(offsets)``) y la celda destino de cada paso. Sin
    esas comprobaciones, un traceback corrupto (todo ceros, un código inventado o
    un camino que sale del ráster) hacía que el ``@njit`` leyera fuera del array y
    el proceso muriera con una violación de acceso en lugar de devolver ``None``.
    """
    filas = traceback.shape[0]
    columnas = traceback.shape[1]
    n_offsets = offsets.shape[0]
    start_row, start_col = start_pixel
    hard_cap = filas * columnas

    if start_row < 0 or start_row >= filas or start_col < 0 or start_col >= columnas:
        return None
    fin_r, fin_c = end_pixel
    if fin_r < 0 or fin_r >= filas or fin_c < 0 or fin_c >= columnas:
        return None

    # --- Pasada 1: contar ---
    r, c = fin_r, fin_c
    n = 0
    while not (r == start_row and c == start_col) and n < hard_cap:
        code = traceback[r, c]
        if code < 0 or code >= n_offsets:
            return None
        r -= offsets[code, 0]
        c -= offsets[code, 1]
        if r < 0 or r >= filas or c < 0 or c >= columnas:
            return None
        n += 1

    if n >= hard_cap:
        return None

    # --- Pasada 2: rellenar tamaño exacto ---
    path = np.empty((n + 1, 2), dtype=np.int32)
    r, c = fin_r, fin_c
    for k in range(n):
        path[k, 0] = r
        path[k, 1] = c
        code = traceback[r, c]
        if code < 0 or code >= n_offsets:
            return None
        r -= offsets[code, 0]
        c -= offsets[code, 1]
        if r < 0 or r >= filas or c < 0 or c >= columnas:
            return None
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
            all_pixels = [origin_pixel, *dest_pixels.values()]
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            H, W = cost_raster_full.shape
            r_min = max(0, min(rows) - bbox_buffer)
            r_max = min(H, max(rows) + bbox_buffer + 1)
            c_min = max(0, min(cols) - bbox_buffer)
            c_max = min(W, max(cols) + bbox_buffer + 1)

            # S3 (perf-fast-io): vista directa del memmap. MCP solo lee la
            # entrada (verificado: no la muta) y el slice basico de un
            # C-contiguo es contiguo -> sin copia de 1,27 GB por worker.
            cost_raster = cost_raster_full[r_min:r_max, c_min:c_max]

            origin_pixel_local = (origin_pixel[0] - r_min, origin_pixel[1] - c_min)
            dest_pixels_local = {
                did: (r - r_min, c - c_min) for did, (r, c) in dest_pixels.items()
            }
        else:
            # Sin recorte: vista directa del memmap (S3: sin copia; liberar el
            # nombre no aporta porque MCP mantiene la referencia durante el cálculo).
            r_min, c_min = 0, 0
            cost_raster = cost_raster_full
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
        # dest_ids/cost_scalars permiten reanudar (S9) sin recomputar el MCP:
        # la Fase 2 puede reconstruirse desde el propio npz.
        # S2 (perf-fast-io): comprimir solo si los npz van a persistir
        # (KEEP_PHASE1_FILES=True -> archivos chicos para reanudar); si son
        # temporales, np.savez plano ahorra ~29 s de zlib por origen en
        # rasters grandes. np.load lee los dos formatos.
        saver = (
            np.savez_compressed if shared_data.get("compress_npz", True) else np.savez
        )
        saver(
            npz_path,
            traceback=traceback,
            offsets=offsets,
            row_offset=np.int32(r_min),
            col_offset=np.int32(c_min),
            dest_ids=np.array(list(cost_scalars.keys()), dtype=np.int64),
            cost_scalars=np.array(list(cost_scalars.values()), dtype=np.float64),
        )
        del traceback

        return (origin_id, cost_scalars)

    except Exception as exc:
        print(f"ERROR Fase 1, origen {origin_id}: {exc}")
        import traceback as tb

        tb.print_exc()
        return (origin_id, {})


# =====================================================================
# Kernel alternativo: A* jerárquico con corredor (puerto de main)
# ---------------------------------------------------------------------
# Dijkstra (MCP_Geometric) expande uniformemente desde el origen y, para
# una unica ruta lejana en un raster grande, explora casi toda la
# superficie. Este kernel es dirigido al objetivo: Fase 1 busca en baja
# resolucion para delimitar un corredor estrategico y Fase 2 busca en
# alta resolucion SOLO dentro de ese corredor. Se expone como OPCION de
# calculo seleccionable por el usuario (KERNEL en el notebook); el
# pipeline MCP permanece intacto como default. Modelo de coste: el
# mismo de main (media de costes de celda * distancia euclidiana entre
# centros de pixel); difiere del modelo geometrico de skimage.
# =====================================================================


@njit(cache=True)
def heuristic_numba(r1, c1, r2, c2, dx, dy):
    """Heuristica de distancia euclidiana (en unidades del CRS)."""
    return math.sqrt(((r2 - r1) * dy) ** 2 + ((c2 - c1) * dx) ** 2)


@njit(cache=True)
def _heap_sift_up(heap, pos):
    # S7 (revision): heap binario manual del open set (numba no soporta heapq
    # con listas tipadas de forma fiable). Mueve heap[pos] hacia arriba
    # mientras su padre tenga f (columna 0) mayor.
    while pos > 0:
        parent = (pos - 1) // 2
        if heap[parent, 0] > heap[pos, 0]:
            for k in range(4):
                tmp = heap[parent, k]
                heap[parent, k] = heap[pos, k]
                heap[pos, k] = tmp
            pos = parent
        else:
            break


@njit(cache=True)
def _heap_sift_down(heap, n, pos):
    # Hundir heap[pos] dentro del heap de tamano n mientras algun hijo tenga
    # f menor.
    while True:
        left = 2 * pos + 1
        if left >= n:
            break
        smallest = left
        right = left + 1
        if right < n and heap[right, 0] < heap[left, 0]:
            smallest = right
        if heap[smallest, 0] < heap[pos, 0]:
            for k in range(4):
                tmp = heap[pos, k]
                heap[pos, k] = heap[smallest, k]
                heap[smallest, k] = tmp
            pos = smallest
        else:
            break


# Capacidad inicial del heap del open set; crece por duplicacion cuando se
# llena (el numero de push depende del area explorada, no se conoce a priori).
_HEAP_INITIAL_CAPACITY = 1024


@njit(cache=True)
def a_star_search(cost_array, nodata_value, start_pixel, end_pixel, dx, dy, weight, search_mask):
    """
    A* sobre el raster de coste, optimizado con Numba (puerto fiel de main).

    `came_from` guarda un codigo direccional 0..8: el predecesor de una
    celda es (dr, dc) con dr = direction // 3 - 1 y dc = direction % 3 - 1.
    Devuelve (path_found, came_from); la ruta se reconstruye con
    `reconstruct_path`.

    S7 (revision): el open set es un heap binario manual (array (n,4) float64
    con columnas [f, g, r, c], sift-up/sift-down manuales y crecimiento por
    duplicacion); antes se hacia vstack + argmin por pop, cuadratico en celdas
    exploradas. Misma semantica de f/g: se extrae siempre la entrada de f
    minimo; entre empates el orden de desempate puede diferir, lo que puede
    cambiar la ruta elegida entre rutas de coste identico (igual que el
    comportamiento documentado del kernel para empates).
    """
    height, width = cost_array.shape
    g_cost = np.full(cost_array.shape, np.inf, dtype=np.float64)
    came_from = np.full(cost_array.shape, -1, dtype=np.int16)

    capacity = _HEAP_INITIAL_CAPACITY
    heap = np.empty((capacity, 4), dtype=np.float64)
    n_heap = 0

    h_initial = heuristic_numba(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1], dx, dy)
    heap[0, 0] = h_initial * weight
    heap[0, 1] = 0.0
    heap[0, 2] = float(start_pixel[0])
    heap[0, 3] = float(start_pixel[1])
    n_heap = 1
    g_cost[start_pixel] = 0.0
    path_found = False

    while n_heap > 0:
        g = heap[0, 1]
        current_pos = (int(heap[0, 2]), int(heap[0, 3]))

        # Pop de la raiz: la ultima entrada ocupa su lugar y se hunde.
        n_heap -= 1
        if n_heap > 0:
            for k in range(4):
                heap[0, k] = heap[n_heap, k]
            _heap_sift_down(heap, n_heap, 0)

        if current_pos == end_pixel:
            path_found = True
            break

        if g > g_cost[current_pos]:
            continue

        cost_current = cost_array[current_pos]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue

                neighbor_pos = (current_pos[0] + dr, current_pos[1] + dc)

                if not (0 <= neighbor_pos[0] < height and 0 <= neighbor_pos[1] < width):
                    continue
                if not search_mask[neighbor_pos]:
                    continue

                cost_neighbor = cost_array[neighbor_pos]
                if cost_neighbor == nodata_value or np.isinf(cost_neighbor) or np.isnan(cost_neighbor):
                    continue

                dist_m = math.sqrt((dr * dy) ** 2 + (dc * dx) ** 2)
                avg_cost = (cost_current + cost_neighbor) / 2.0
                tentative_g_cost = g + (avg_cost * dist_m)

                if tentative_g_cost < g_cost[neighbor_pos]:
                    direction = (dr + 1) * 3 + (dc + 1)
                    came_from[neighbor_pos] = direction
                    g_cost[neighbor_pos] = tentative_g_cost
                    h = heuristic_numba(neighbor_pos[0], neighbor_pos[1], end_pixel[0], end_pixel[1], dx, dy)
                    new_f_cost = tentative_g_cost + (h * weight)

                    # Push con crecimiento por duplicacion cuando el heap
                    # esta lleno (se copian solo las n_heap entradas vivas).
                    if n_heap == capacity:
                        new_heap = np.empty((capacity * 2, 4), dtype=np.float64)
                        for i in range(n_heap):
                            for k in range(4):
                                new_heap[i, k] = heap[i, k]
                        heap = new_heap
                        capacity = capacity * 2
                    heap[n_heap, 0] = new_f_cost
                    heap[n_heap, 1] = tentative_g_cost
                    heap[n_heap, 2] = float(neighbor_pos[0])
                    heap[n_heap, 3] = float(neighbor_pos[1])
                    n_heap += 1
                    _heap_sift_up(heap, n_heap - 1)

    return path_found, came_from


@njit(cache=True)
def reconstruct_path(came_from_array, start_pixel, end_pixel):
    """
    Reconstruye la ruta a partir del array `came_from` direccional.

    Cota de seguridad: came_from_array.size (un arbol de caminos minimos
    no revisita celdas). Un bucle o un -1 intermedio devuelve None en vez
    de colgarse.
    """
    path = np.zeros((came_from_array.size, 2), dtype=np.int32)
    current_pos_r, current_pos_c = end_pixel
    count = 0
    limit = came_from_array.size

    while (current_pos_r, current_pos_c) != start_pixel and count < limit:
        path[count] = np.array([current_pos_r, current_pos_c])
        direction = came_from_array[current_pos_r, current_pos_c]
        if direction == -1:
            return None

        dc = (direction % 3) - 1
        dr = (direction // 3) - 1
        current_pos_r -= dr
        current_pos_c -= dc
        count += 1

    # S8/R3-reconstruct-path-limit-oob (revision): si el bucle agoto la cota
    # (came_from corrupto, p. ej. un ciclo) sin llegar al inicio, count == limit
    # y el write final caeria fuera del array (indices 0..limit-1). Devolver
    # None en vez de escribir fuera de rango bajo numba sin bounds checking.
    if count >= limit:
        return None

    path[count] = np.array([start_pixel[0], start_pixel[1]])
    return path[: count + 1][::-1]


def accumulate_cost(cost_array, nodata_value, res_x, res_y, pixel_path):
    """
    Coste acumulado de una ruta de pixeles con el modelo del kernel A*:
    suma de (coste medio entre celdas consecutivas * distancia entre
    centros). Reproduce el g_cost que el propio A* acumula para la ruta
    final, de modo que `cost_total` es comparable entre fases.
    """
    total = 0.0
    for (r0, c0), (r1, c1) in pairwise(pixel_path):
        cost_0 = float(cost_array[r0, c0])
        cost_1 = float(cost_array[r1, c1])
        if cost_1 == nodata_value or not np.isfinite(cost_1) or not np.isfinite(cost_0):
            return float("inf")
        dist_m = math.sqrt(((r1 - r0) * res_y) ** 2 + ((c1 - c0) * res_x) ** 2)
        total += ((cost_0 + cost_1) / 2.0) * dist_m
    return total


def corridor_astar_route(
    cost_data,
    nodata_value,
    res_x,
    res_y,
    start_pixel,
    end_pixel,
    search_mask,
    downsampling_factors=(32, 20, 10),
    corridor_buffer_pixels=150,
    heuristic_weight=1.0,
):
    """
    Ruta 1->1 con el kernel A* jerarquico (comportamiento de main).

    Fase 1: intenta A* en baja resolucion con cada factor de
    downsampling (de mayor a menor granularidad, primer exito gana).
    Fase 2: A* en alta resolucion restringido al corredor alrededor de
    la ruta de baja resolucion INTERSECTADO con la mascara de estudio;
    si el corredor no produce ruta, PLAN B: A* en toda la mascara.

    Devuelve (path_pixels, cost_total): path_pixels es None y cost_total
    inf cuando no hay ruta alcanzable. `cost_data` debe ser un array
    float con np.inf (o nodata_value) en las celdas prohibidas; el
    caller aplica la mascara de estudio igual que en el pipeline MCP.

    Defaults heredados de main y su rol: `downsampling_factors` es la
    escalera de granularidad (32 = exploracion mas gruesa; 20 y 10
    refinan si la anterior no encuentra ruta); `corridor_buffer_pixels`
    es el ancho del corredor de Fase 2 alrededor de la ruta LR, y es el
    parametro que mas afecta si el corredor contiene la ruta optima.
    Cambiarlos cambia la calidad de la ruta silenciosamente.

    R3-astar-heuristic-admissibility (revision): la heuristica euclidiana
    ponderada es admisible solo si heuristic_weight <= minimo de coste
    transitable de la superficie; si weight lo excede puede sobreestimar
    y la ruta puede ser suboptima bajo el propio modelo del kernel. La
    deteccion es runtime: con weight > 0 y weight > min_coste se emite
    una ADVERTENCIA que sugiere bajar HEURISTIC_WEIGHT (R2/R3 del quinto
    review: la condicion es weight-relativa, no el umbral fijo 1.0, que
    dejaba sin aviso a weight > 1.0 con costes en [1.0, weight) y avisaba
    en vano con 0 < weight < min_coste). `heuristic_weight=0` convierte
    la busqueda en Dijkstra exacto (mas lento, sin direccion y siempre
    admisible). Un weight negativo levanta ValueError: premia alejarse
    del objetivo y produce rutas arbitrarias.
    """
    import lcp.processing as proc

    start_pixel = (int(start_pixel[0]), int(start_pixel[1]))
    end_pixel = (int(end_pixel[0]), int(end_pixel[1]))

    # R3-heuristic-weight-unvalidated (quinto review): un weight negativo
    # hace f = g + w*h premiar la lejania del objetivo y bypassuea el guard
    # de abajo (que exige weight > 0): rechazo explicito.
    if heuristic_weight < 0:
        raise ValueError(
            f"HEURISTIC_WEIGHT no puede ser negativo: {heuristic_weight}. "
            "Un peso negativo premia alejarse del objetivo y produce rutas "
            "arbitrarias; use 0 (Dijkstra exacto) o un valor positivo."
        )

    # S6/R3-heuristic-admissibility-no-runtime-guard: con weight > 0, la
    # heuristica euclidiana sobreestima el coste restante cuando weight
    # excede el minimo de coste transitable (la ruta puede ser suboptima
    # bajo el propio modelo del kernel). Deteccion runtime weight-relativa
    # con un solo pase sobre las celdas transitables (excluye
    # inf/nodata/NaN): aceptable porque corridor_astar solo corre 1->1.
    # weight=0 equivale a Dijkstra exacto y siempre es admisible: sin aviso.
    if heuristic_weight > 0 and search_mask.any():
        _traversable = cost_data[search_mask]
        _traversable = _traversable[
            np.isfinite(_traversable) & (_traversable != nodata_value)
        ]
        if _traversable.size:
            _min_cost = float(np.nanmin(_traversable))
            if _min_cost < heuristic_weight:
                print(
                    f"  ADVERTENCIA: el minimo de coste transitable "
                    f"({_min_cost:.4g}) es menor que heuristic_weight="
                    f"{heuristic_weight}: la heuristica euclidiana puede "
                    "sobreestimar y la ruta puede ser suboptima. Baje "
                    f"HEURISTIC_WEIGHT a <= {_min_cost:.4g} (o 0 = Dijkstra "
                    "exacto, mas lento) si la optimalidad importa."
                )

    path_pixels_lr = None
    successful_factor = None
    for factor in downsampling_factors:
        cost_lr, dx_lr, dy_lr = proc.create_low_res_data(cost_data, res_x, res_y, factor)
        mask_lr = search_mask[::factor, ::factor]
        # Clamp al grid LR recortado (floor(dim/factor)): un pixel dentro de
        # las ultimas factor-1 filas/columnas del raster caeria fuera del grid
        # y, con bounds checking deshabilitado en numba, g_cost[start_lr] = 0.0
        # seria una escritura fuera de rango antes del PLAN B (hallazgos R2/R3
        # de la revision nativa).
        start_lr = (
            min(start_pixel[0] // factor, cost_lr.shape[0] - 1),
            min(start_pixel[1] // factor, cost_lr.shape[1] - 1),
        )
        end_lr = (
            min(end_pixel[0] // factor, cost_lr.shape[0] - 1),
            min(end_pixel[1] // factor, cost_lr.shape[1] - 1),
        )
        found_lr, came_from_lr = a_star_search(
            cost_lr, nodata_value, start_lr, end_lr, dx_lr, abs(dy_lr), heuristic_weight, mask_lr
        )
        if found_lr:
            path_pixels_lr = reconstruct_path(came_from_lr, start_lr, end_lr)
            successful_factor = factor
            break

    corridor_mask = None
    if path_pixels_lr is not None:
        corridor_mask = proc.create_search_corridor(
            path_pixels_lr, cost_data.shape, successful_factor, corridor_buffer_pixels
        )
        final_mask = np.logical_and(corridor_mask, search_mask)
        found_hr, came_from_hr = a_star_search(
            cost_data, nodata_value, start_pixel, end_pixel, res_x, abs(res_y), heuristic_weight, final_mask
        )
        if found_hr:
            path_hr = reconstruct_path(came_from_hr, start_pixel, end_pixel)
            if path_hr is not None:
                cost_total = accumulate_cost(cost_data, nodata_value, res_x, res_y, path_hr)
                return path_hr, cost_total

    # PLAN B: busqueda en toda la mascara de estudio (igual que main).
    found_fb, came_from_fb = a_star_search(
        cost_data, nodata_value, start_pixel, end_pixel, res_x, abs(res_y), heuristic_weight, search_mask
    )
    if found_fb:
        path_fb = reconstruct_path(came_from_fb, start_pixel, end_pixel)
        if path_fb is not None:
            cost_total = accumulate_cost(cost_data, nodata_value, res_x, res_y, path_fb)
            return path_fb, cost_total

    return None, float("inf")

