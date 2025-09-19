# lcp_network/pathfinder.py

import numpy as np
import math
from numba import njit

@njit
def heuristic_numba(r1, c1, r2, c2, dx, dy):
    return math.sqrt((r2 - r1)**2 + (c2 - c1)**2) * math.sqrt(dx * dy)

@njit
def a_star_numba_compiled(cost_array, nodata_value, start_pixel, end_pixel, dx, dy, weight, search_mask):
    # (El código del algoritmo A* completo va aquí, sin cambios)
    height, width = cost_array.shape; g_cost = np.full(cost_array.shape, np.inf); came_from = np.full(cost_array.shape, -1, dtype=np.int16)
    open_set = np.zeros((1, 4)); h_initial = heuristic_numba(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1], dx, dy)
    open_set[0] = [h_initial*weight, 0.0, start_pixel[0], start_pixel[1]]; g_cost[start_pixel] = 0; path_found = False
    while open_set.shape[0] > 0:
        min_idx = np.argmin(open_set[:, 0]); f, g, r, c = open_set[min_idx]; current_pos = (int(r), int(c))
        if open_set.shape[0] == 1: open_set = np.zeros((0, 4))
        else: open_set = np.vstack((open_set[:min_idx], open_set[min_idx + 1:]))
        if current_pos == end_pixel: path_found = True; break
        if g > g_cost[current_pos]: continue
        cost_current = cost_array[current_pos]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                neighbor_pos = (current_pos[0] + dr, current_pos[1] + dc)
                if not (0 <= neighbor_pos[0] < height and 0 <= neighbor_pos[1] < width): continue
                if not search_mask[neighbor_pos]: continue
                cost_neighbor = cost_array[neighbor_pos];
                if cost_neighbor == nodata_value: continue
                dist_m = math.sqrt((dr*dy)**2 + (dc*dx)**2); avg_cost = (cost_current + cost_neighbor) / 2.0
                tentative_g_cost = g + (avg_cost*dist_m)
                if tentative_g_cost < g_cost[neighbor_pos]:
                    direction = (dr+1)*3 + (dc+1); came_from[neighbor_pos] = direction; g_cost[neighbor_pos] = tentative_g_cost
                    h = heuristic_numba(neighbor_pos[0], neighbor_pos[1], end_pixel[0], end_pixel[1], dx, dy)
                    new_f_cost = tentative_g_cost + (h*weight)
                    new_entry = np.array([[new_f_cost, tentative_g_cost, neighbor_pos[0], neighbor_pos[1]]])
                    open_set = np.vstack((open_set, new_entry))
    return path_found, came_from, g_cost


@njit
def reconstruct_path_pixels_numba(came_from_array, start_pixel, end_pixel):
    # (El código de reconstrucción de ruta va aquí, sin cambios)
    path = np.zeros((came_from_array.size, 2), dtype=np.int32); current_pos_r, current_pos_c = end_pixel; count = 0; limit = came_from_array.size
    while (current_pos_r, current_pos_c) != start_pixel and count < limit:
        path[count] = np.array([current_pos_r, current_pos_c]); direction = came_from_array[current_pos_r, current_pos_c]
        if direction == -1: return None
        dc = (direction % 3) - 1; dr = (direction // 3) - 1; current_pos_r -= dr; current_pos_c -= dc; count += 1
    path[count] = np.array([start_pixel[0], start_pixel[1]]); return path[:count+1][::-1]