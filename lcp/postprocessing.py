# lcp/postprocessing.py

import os
import lcp.pathfinder as pf
import lcp.utils as utils
import lcp.processing as proc

def extract_and_save_routes(traceback, origin_id, origin_coords, all_points, transform, crs, output_dir):
    """
    Itera sobre todos los destinos, reconstruye cada ruta y la guarda.
    VERSIÓN DE DEPURACIÓN PROFUNDA.
    """
    saved_routes_count = 0
    start_pixel = proc.world_to_pixel(transform, origin_coords[0], origin_coords[1])
    
    print("\n  --- Dentro de extract_and_save_routes ---")
    print(f"    > Procesando rutas desde el origen {origin_id} (píxel {start_pixel})")

    for dest_id, dest_coords in all_points.items():
        if origin_id == dest_id:
            continue
            
        end_pixel = proc.world_to_pixel(transform, dest_coords[0], dest_coords[1])
        
        # --- Diagnóstico de la Reconstrucción ---
        print(f"      > Intentando reconstruir ruta a destino {dest_id} (píxel {end_pixel})...")
        path_pixels = pf.reconstruct_path_from_traceback(traceback, start_pixel, end_pixel)
        
        if path_pixels is not None:
            output_path = os.path.join(output_dir, f"ruta_final_{origin_id}_a_{dest_id}.shp")
            utils.save_path_to_shapefile(path_pixels, transform, crs, output_path)
            saved_routes_count += 1
            print(f"        -> ÉXITO. Ruta guardada.")
        else:
            print(f"        -> FALLO. La función de reconstrucción devolvió None.")
            
    return saved_routes_count