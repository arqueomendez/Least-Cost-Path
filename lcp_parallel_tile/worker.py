# lcp_network/worker.py

import os
import numpy as np
import rasterio

from . import geoutils
from . import pathfinder

def tile_worker(task, config):
    tile_id = task['tile_id']; window = task['window']; border_nodes = task['border_nodes']
    try:
        with rasterio.open(config['cost_raster_path']) as src:
            tile_data = src.read(1, window=window)
            tile_transform = src.window_transform(window)
            nodata = src.nodata
            res = (abs(src.transform.a), abs(src.transform.e))
            crs = src.crs

            # --- [MODIFICACIÓN] Crear máscara rasterizada para el tile ---
            search_mask_hr = None
            if config['mask_shapefile_path']:
                # Creamos una máscara que se alinea perfectamente con el tile que hemos leído
                search_mask_hr = geoutils.create_mask_from_vector(config['mask_shapefile_path'], src, window=window)
            
            # Si no hay máscara, creamos una que permite todo el espacio
            if search_mask_hr is None:
                search_mask_hr = np.ones_like(tile_data, dtype=bool)

        nodes_list = list(border_nodes.items()); num_rutas_calculadas = 0
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                id_start, start_pixel_hr = nodes_list[i]; id_end, end_pixel_hr = nodes_list[j]
                
                path_found_lr, path_pixels_lr, successful_factor, trans_low = False, None, None, None
                for factor in config['downsampling_factors']:
                    cost_lr = geoutils.create_low_res_data(tile_data, factor)
                    # --- [MODIFICACIÓN] Usar máscara de baja resolución ---
                    mask_lr = search_mask_hr[::factor, ::factor]
                    trans_lr = tile_transform * tile_transform.scale(factor, factor); dx_lr, dy_lr = res[0] * factor, res[1] * factor
                    start_lr = (start_pixel_hr[0] // factor, start_pixel_hr[1] // factor); end_lr = (end_pixel_hr[0] // factor, end_pixel_hr[1] // factor)
                    
                    found, came_from_lr, _ = pathfinder.a_star_numba_compiled(cost_lr, nodata, start_lr, end_lr, dx_lr, abs(dy_lr), config['heuristic_weight'], mask_lr)
                    if found:
                        path_found_lr, successful_factor, trans_low = True, factor, trans_lr
                        path_pixels_lr = pathfinder.reconstruct_path_pixels_numba(came_from_lr, start_lr, end_lr); break
                
                path_found_hr, came_from_hr = False, None
                if path_found_lr:
                    corridor_mask = geoutils.create_search_corridor(path_pixels_lr, tile_data.shape, successful_factor, config['corridor_buffer_pixels'])
                    # --- [MODIFICACIÓN] La máscara final es la intersección del corredor y la máscara principal ---
                    final_mask = np.logical_and(corridor_mask, search_mask_hr)
                    path_found_hr, came_from_hr, _ = pathfinder.a_star_numba_compiled(tile_data, nodata, start_pixel_hr, end_pixel_hr, res[0], abs(res[1]), config['heuristic_weight'], final_mask)
                
                if path_found_hr:
                    path_pixels_hr = pathfinder.reconstruct_path_pixels_numba(came_from_hr, start_pixel_hr, end_pixel_hr)
                    output_path = os.path.join(config['session_output_dir'], f"ruta_{tile_id}__{id_start}__{id_end}.shp")
                    geoutils.save_path_to_shapefile(path_pixels_hr, tile_transform, crs, output_path)
                    num_rutas_calculadas += 1

        return ("Éxito", tile_id, num_rutas_calculadas)
    except Exception as e: return ("Error", tile_id, str(e))