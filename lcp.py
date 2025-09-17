# run_analysis.py
# Este script reemplaza al Jupyter Notebook y orquesta todo el proceso.

import os
import numpy as np
import rasterio
from datetime import datetime

# Importar los módulos del paquete lcp
import lcp.data_loader as dl
import lcp.processing as proc
import lcp.pathfinder as pf
import lcp.utils as utils

def main():
    # ==============================================================================
    # --- PARÁMETROS CONFIGURABLES POR EL USUARIO ---
    # ==============================================================================
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Crear una carpeta de sesión única para cada ejecución
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}')

    # --- Rutas a los datos de entrada ---
    COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')
    ALL_POINTS_SHAPEFILE = os.path.join(DATA_DIR, 'points.shp')
    MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp') # Puede ser None si no se usa máscara

    # --- Parámetros de cálculo (¡MODIFICAR AQUÍ!) ---
    ORIGIN_POINT_ID = 5
    ID_FIELD_NAME = 'id' # Nombre del campo/columna con los IDs de los puntos
    DOWNSAMPLING_FACTORS = [32, 20, 10]
    CORRIDOR_BUFFER_PIXELS = 150
    HEURISTIC_WEIGHT = 1.0

    # ==============================================================================
    # --- EJECUCIÓN DEL ANÁLISIS ---
    # ==============================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Los resultados se guardarán en: {OUTPUT_DIR}")

    try:
        all_points, points_crs = dl.load_points_as_dict(ALL_POINTS_SHAPEFILE, ID_FIELD_NAME)
        
        with dl.load_raster(COST_RASTER_PATH) as src:
            print("Cargando superficie de costo a memoria...")
            cost_data_high_res = src.read(1)
            
            if MASK_SHAPEFILE_PATH and os.path.exists(MASK_SHAPEFILE_PATH):
                main_search_mask = proc.create_mask_from_vector(MASK_SHAPEFILE_PATH, src)
            else:
                print("No se proporcionó máscara de polígono; se buscará en todo el raster.")
                main_search_mask = np.ones_like(cost_data_high_res, dtype=bool)

            origin_coords = all_points.get(ORIGIN_POINT_ID)
            if not origin_coords:
                raise ValueError(f"El ID de origen '{ORIGIN_POINT_ID}' no se encontró en el archivo de puntos.")
            
            start_pixel_hr = proc.world_to_pixel(src.transform, origin_coords[0], origin_coords[1])
            print(f"\nAnálisis desde el punto ID {ORIGIN_POINT_ID} (Píxel de alta res: {start_pixel_hr})")

            for dest_id, dest_coords in all_points.items():
                if dest_id == ORIGIN_POINT_ID: continue

                print(f"\n--- Calculando ruta: {ORIGIN_POINT_ID} -> {dest_id} ---")
                end_pixel_hr = proc.world_to_pixel(src.transform, dest_coords[0], dest_coords[1])
                
                # FASE 1: Búsqueda a baja resolución
                path_found_lr, path_pixels_lr, successful_factor, trans_low = False, None, None, None
                print("-> FASE 1: Buscando en baja resolución...")
                for factor in DOWNSAMPLING_FACTORS:
                    print(f"  Intentando con factor de remuestreo {factor}x...")
                    cost_lr, trans_lr, dx_lr, dy_lr = proc.create_low_res_data(src, factor)
                    mask_lr = main_search_mask[::factor, ::factor]
                    start_lr = (start_pixel_hr[0] // factor, start_pixel_hr[1] // factor)
                    end_lr = (end_pixel_hr[0] // factor, end_pixel_hr[1] // factor)
                    
                    found, came_from_lr = pf.a_star_search(cost_lr, src.nodata, start_lr, end_lr, dx_lr, abs(dy_lr), HEURISTIC_WEIGHT, mask_lr)
                    
                    if found:
                        print(f"  Éxito con factor {factor}.")
                        path_found_lr, successful_factor, trans_low = True, factor, trans_lr
                        path_pixels_lr = pf.reconstruct_path(came_from_lr, start_lr, end_lr)
                        break
                    else:
                        print(f"  Falló con factor {factor}.")
                
                # FASE 2: Búsqueda a alta resolución
                print("\n-> FASE 2: Buscando en alta resolución...")
                path_found_hr, came_from_hr = False, None

                if path_found_lr and path_pixels_lr is not None:
                    print("  Creando corredor a partir de la ruta de baja resolución...")
                    corridor_mask = proc.create_search_corridor(path_pixels_lr, cost_data_high_res.shape, successful_factor, CORRIDOR_BUFFER_PIXELS)
                    final_search_mask = np.logical_and(corridor_mask, main_search_mask)
                    path_found_hr, came_from_hr = pf.a_star_search(cost_data_high_res, src.nodata, start_pixel_hr, end_pixel_hr, src.res[0], abs(src.res[1]), HEURISTIC_WEIGHT, final_search_mask)

                if not path_found_hr:
                    print("  Búsqueda en corredor fallida o no realizada. Iniciando PLAN B: búsqueda en toda la máscara.")
                    path_found_hr, came_from_hr = pf.a_star_search(cost_data_high_res, src.nodata, start_pixel_hr, end_pixel_hr, src.res[0], abs(src.res[1]), HEURISTIC_WEIGHT, main_search_mask)

                # Guardar resultados
                if path_found_hr:
                    print(f"  ÉXITO FINAL: Se encontró la ruta para {ORIGIN_POINT_ID} -> {dest_id}.")
                    path_pixels_hr = pf.reconstruct_path(came_from_hr, start_pixel_hr, end_pixel_hr)
                    
                    if path_pixels_lr is not None:
                        p1_path = os.path.join(OUTPUT_DIR, f"ruta_fase1_{ORIGIN_POINT_ID}_a_{dest_id}.shp")
                        utils.save_path_to_shapefile(path_pixels_lr, trans_low, src.crs, p1_path)
                    
                    final_path_shp = os.path.join(OUTPUT_DIR, f"ruta_final_{ORIGIN_POINT_ID}_a_{dest_id}.shp")
                    utils.save_path_to_shapefile(path_pixels_hr, src.transform, src.crs, final_path_shp)
                else:
                    print(f"  ERROR CRÍTICO: No se pudo encontrar ninguna ruta para {ORIGIN_POINT_ID} -> {dest_id}.")

    except Exception as e:
        print(f"\nOcurrió un error fatal en la ejecución: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n--- ANÁLISIS COMPLETADO ---")

if __name__ == '__main__':
    main()