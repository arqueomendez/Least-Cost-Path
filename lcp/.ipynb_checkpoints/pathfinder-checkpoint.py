# lcp/pathfinder.py

import numpy as np
from skimage.graph import MCP_Geometric
from numba import njit

def cost_surface_calculator(cost_raster, start_coords):
    """
    Calcula la superficie de coste acumulado y el traceback desde un punto de origen.
    """
    # Forzar el tipo de datos a float64 para máxima compatibilidad con el backend de C
    cost_raster_float64 = cost_raster.astype(np.float64)

    # MCP_Geometric es el motor de cálculo principal.
    mcp = MCP_Geometric(cost_raster_float64, fully_connected=True)
    
    # find_costs calcula el coste desde el(los) punto(s) de inicio a todos los demás.
    costs, traceback = mcp.find_costs([start_coords])
    
    return costs, traceback


@njit
def reconstruct_path_from_traceback(traceback, start_pixel, end_pixel):
    """
    Reconstruye una ruta utilizando el ráster de traceback DIRECCIONAL de scikit-image.
    --- VERSIÓN FINAL CON LA LÓGICA DE RECONSTRUCCIÓN INVERSA CORRECTA ---
    """
    # Mapeo de los códigos de dirección (0-7) de scikit-image a cambios en (fila, columna)
    # Índice: 0   1   2   3   4   5   6   7
    # Vecino: NW, N, NE, W, E, SW, S, SE
    dr = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dc = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    curr_row, curr_col = end_pixel
    start_row, start_col = start_pixel
    
    path = np.zeros((traceback.size, 2), dtype=np.int32)
    count = 0
    max_steps = traceback.size 

    while not (curr_row == start_row and curr_col == start_col) and count < max_steps:
        path[count] = np.array([curr_row, curr_col])
        count += 1

        direction_code = traceback[curr_row, curr_col]
        
        if direction_code < 0:
            return None 

        # --- INICIO DE LA CORRECCIÓN FINAL Y DEFINITIVA ---
        # El traceback en (r, c) nos dice la dirección DESDE la que se llegó.
        # Para volver al padre, debemos movernos en la dirección OPUESTA.
        # Por lo tanto, RESTAMOS el delta de la leyenda de dirección.
        curr_row = curr_row - dr[direction_code]
        curr_col = curr_col - dc[direction_code]
        # --- FIN DE LA CORRECCIÓN FINAL Y DEFINITIVA ---

    if count >= max_steps:
        return None

    path[count] = np.array([start_row, start_col])
    return path[:count + 1][::-1]