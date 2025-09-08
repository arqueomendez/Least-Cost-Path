import numpy as np
import rasterio
import math
import os
from tqdm import tqdm
import fiona
from shapely.geometry import Point, LineString, mapping, shape
from shapely.ops import unary_union
from numba import njit
from datetime import datetime
from skimage.draw import disk
from rasterio.enums import Resampling
from rasterio.features import rasterize

# --- 1. CONFIGURACIÓN DEL PROYECTO Y PARÁMETROS ---
BASE_PATH = R"C:\Users\User\Desktop\SofiHanna"
POINTS_SUBFOLDER = os.path.join(BASE_PATH, 'puntos_por_100km')
COST_RASTER_PATH = os.path.join(BASE_PATH, 'final_total_cost.tif')

# --- PARÁMETROS DE ENTRADA Y SALIDA ---
# Nombre del Shapefile que contiene TODOS los 11 puntos
ALL_POINTS_SHAPEFILE = os.path.join(POINTS_SUBFOLDER, 'puntos_por_100km.shp') # <--- CONFIRMA ESTE NOMBRE

# ID del punto que será el ORIGEN de todas las rutas
ORIGIN_POINT_ID = 5 # <--- PUNTO DE INICIO FIJO

# Nombre del campo en el shapefile que contiene los IDs de los puntos
ID_FIELD_NAME = 'rand_point' # <--- CONFIRMA EL NOMBRE DE ESTE CAMPO

# --- NUEVO: PARÁMETRO PARA LA MÁSCARA DE BÚSQUEDA ---
# Dejar como None si no se quiere usar una máscara.
MASK_SHAPEFILE_PATH = os.path.join(BASE_PATH, 'poligono_extendido_cortado_disuelto.shp')

# --- PARÁMETROS DE LA BÚSQUEDA JERÁRQUICA ---
DOWNSAMPLING_FACTORS = [32, 20, 10] # Factores a probar, en orden
CORRIDOR_BUFFER_PIXELS = 150
HEURISTIC_WEIGHT = 1.0

# --- 2. FUNCIONES AUXILIARES ---

def load_all_points(shapefile_path, id_field):
    """Carga todos los puntos y sus IDs desde un shapefile a un diccionario."""
    points = {}
    crs = None
    try:
        with fiona.open(shapefile_path, 'r') as collection:
            crs = collection.crs
            if len(collection) == 0:
                print(f"Error: El shapefile '{os.path.basename(shapefile_path)}' está vacío.")
                return None, None
            for feature in collection:
                # Asegurarse de que el ID es un tipo de dato consistente (entero)
                point_id = int(feature['properties'][id_field])
                coords = feature['geometry']['coordinates']
                points[point_id] = coords
        return points, crs
    except Exception as e:
        print(f"Error cargando los puntos desde {shapefile_path}: {e}")
        return None, None

def load_polygon_geometry(shapefile_path):
    """Carga y disuelve la geometría de un shapefile de polígonos."""
    if not shapefile_path or not os.path.exists(shapefile_path):
        return None
    try:
        with fiona.open(shapefile_path, 'r') as collection:
            if len(collection) == 0:
                print(f"Advertencia: El shapefile de máscara '{os.path.basename(shapefile_path)}' está vacío.")
                return None
            geometries = [shape(feature['geometry']) for feature in collection]
            return unary_union(geometries)
    except Exception as e:
        print(f"Error cargando la geometría del polígono desde {shapefile_path}: {e}")
        return None

def world_to_pixel(transform, x, y):
    col, row = ~transform * (x, y); return int(row), int(col)

def create_mask_from_vector(vector_path, raster_src):
    """
    Rasteriza un archivo vectorial para crear una máscara booleana.
    True donde el vector existe, False en el resto.
    """
    if not vector_path or not os.path.exists(vector_path):
        print("Advertencia: No se proporcionó o no se encontró el archivo de máscara. No se aplicará la máscara.")
        return None

    with fiona.open(vector_path, "r") as vector_file:
        if vector_file.crs != raster_src.crs:
            print(f"¡ERROR CRÍTICO DE CRS! El CRS de la máscara ({vector_file.crs}) no coincide con el del ráster ({raster_src.crs}).")
            return None
        
        shapes = [feature["geometry"] for feature in vector_file]

    mask = rasterize(
        shapes,
        out_shape=raster_src.shape,
        transform=raster_src.transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    
    return mask.astype(bool)

# --- 3. ALGORITMO A* Y HELPERS (COMPILADOS CON NUMBA) ---

@njit
def heuristic_numba(r1, c1, r2, c2, dx, dy):
    return math.sqrt((r2 - r1)**2 + (c2 - c1)**2) * math.sqrt(dx * dy)

@njit
def a_star_numba_compiled(cost_array, nodata_value, start_pixel, end_pixel, dx, dy, weight, search_mask):
    height, width = cost_array.shape
    g_cost = np.full(cost_array.shape, np.inf, dtype=np.float64)
    came_from = np.full(cost_array.shape, -1, dtype=np.int16)
    open_set = np.zeros((1, 4), dtype=np.float64)
    h_initial = heuristic_numba(start_pixel[0], start_pixel[1], end_pixel[0], end_pixel[1], dx, dy)
    open_set[0] = [h_initial * weight, 0.0, start_pixel[0], start_pixel[1]]
    g_cost[start_pixel] = 0
    path_found = False
    
    while open_set.shape[0] > 0:
        min_idx = np.argmin(open_set[:, 0])
        f, g, r, c = open_set[min_idx]; current_pos = (int(r), int(c))
        if open_set.shape[0] == 1: open_set = np.zeros((0, 4), dtype=np.float64)
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
                cost_neighbor = cost_array[neighbor_pos]
                if cost_neighbor == nodata_value: continue
                dist_m = math.sqrt((dr * dy)**2 + (dc * dx)**2)
                avg_cost = (cost_current + cost_neighbor) / 2.0
                tentative_g_cost = g + (avg_cost * dist_m)
                
                if tentative_g_cost < g_cost[neighbor_pos]:
                    direction = (dr+1)*3 + (dc+1)
                    came_from[neighbor_pos] = direction; g_cost[neighbor_pos] = tentative_g_cost
                    h = heuristic_numba(neighbor_pos[0], neighbor_pos[1], end_pixel[0], end_pixel[1], dx, dy)
                    new_f_cost = tentative_g_cost + (h * weight)
                    new_entry = np.array([[new_f_cost, tentative_g_cost, neighbor_pos[0], neighbor_pos[1]]], dtype=np.float64)
                    if open_set.shape[0] == 0: open_set = new_entry
                    else: open_set = np.vstack((open_set, new_entry))
    
    return path_found, came_from, g_cost

@njit
def reconstruct_path_pixels_numba(came_from_array, start_pixel, end_pixel):
    path = np.zeros((came_from_array.size, 2), dtype=np.int32)
    current_pos_r, current_pos_c = end_pixel
    count = 0
    limit = came_from_array.size
    
    while (current_pos_r, current_pos_c) != start_pixel and count < limit:
        path[count] = np.array([current_pos_r, current_pos_c], dtype=np.int32)
        direction = came_from_array[current_pos_r, current_pos_c]
        if direction == -1: return None
        dc = (direction % 3) - 1; dr = (direction // 3) - 1
        current_pos_r -= dr; current_pos_c -= dc
        count += 1
        
    path[count] = np.array([start_pixel[0], start_pixel[1]], dtype=np.int32)
    return path[:count+1][::-1]

# --- 4. FUNCIONES PARA BÚSQUEDA JERÁRQUICA ---

def create_low_res_data(src, factor):
    low_res_shape = (src.height // factor, src.width // factor)
    low_res_data = src.read(1, out_shape=low_res_shape, resampling=Resampling.average)
    low_res_transform = src.transform * src.transform.scale(factor, factor)
    low_res_dx = src.res[0] * factor
    low_res_dy = src.res[1] * factor
    return low_res_data, low_res_transform, low_res_dx, low_res_dy

def create_search_corridor(path_low_res, high_res_shape, factor, buffer_pixels):
    corridor_mask = np.zeros(high_res_shape, dtype=bool)
    if path_low_res is None: return corridor_mask
    for r_low, c_low in path_low_res:
        r_high = int(r_low * factor + factor / 2)
        c_high = int(c_low * factor + factor / 2)
        rr, cc = disk((r_high, c_high), buffer_pixels, shape=high_res_shape)
        corridor_mask[rr, cc] = True
    return corridor_mask

def save_path_to_shapefile(pixel_path, transform, crs, output_path):
    world_coords = [transform * (p[1] + 0.5, p[0] + 0.5) for p in pixel_path]
    schema = {'geometry': 'LineString', 'properties': {'id': 'str'}}
    with fiona.open(output_path, 'w', 'ESRI Shapefile', schema, fiona.crs.from_string(str(crs))) as c:
        c.write({'geometry': mapping(LineString(world_coords)), 'properties': {'id': os.path.basename(output_path)}})

# --- 5. EJECUCIÓN DEL ANÁLISIS DE UNO A TODOS ---
if __name__ == '__main__':
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR = os.path.join(BASE_PATH, f"rutas_desde_punto_{ORIGIN_POINT_ID}_{timestamp}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Los resultados se guardarán en: {OUTPUT_DIR}")
        
        all_points, points_crs = load_all_points(ALL_POINTS_SHAPEFILE, ID_FIELD_NAME)

        if all_points:
            with rasterio.open(COST_RASTER_PATH) as src:
                if not (src.crs == points_crs):
                    print("¡ERROR CRÍTICO: El CRS de los puntos y el del ráster no coinciden!")
                else:
                    print("Cargando superficie de coste de alta resolución en RAM...")
                    cost_data_high_res = src.read(1)
                    
                    # --- MODIFICACIÓN: Cargar geometría de la máscara y crear máscara rasterizada ---
                    mask_polygon_geom = None
                    search_mask_hr_user = None
                    if MASK_SHAPEFILE_PATH:
                        print(f"\nCargando geometría de la máscara desde: {os.path.basename(MASK_SHAPEFILE_PATH)}...")
                        mask_polygon_geom = load_polygon_geometry(MASK_SHAPEFILE_PATH)
                        if mask_polygon_geom:
                            print(f"Creando máscara de búsqueda rasterizada...")
                            search_mask_hr_user = create_mask_from_vector(MASK_SHAPEFILE_PATH, src)

                    # --- NUEVA VERIFICACIÓN INICIAL DE TODOS LOS PUNTOS ---
                    if mask_polygon_geom and all_points:
                        print("\n--- VERIFICACIÓN DE PUNTOS CONTRA LA MÁSCARA ---")
                        for point_id, coords in all_points.items():
                            # Verificación vectorial
                            point_geom = Point(coords)
                            is_inside_vector = mask_polygon_geom.contains(point_geom)
                            vector_status = "DENTRO" if is_inside_vector else "FUERA"
                            
                            # Verificación raster
                            raster_status = "N/A"
                            if search_mask_hr_user is not None:
                                try:
                                    r, c = world_to_pixel(src.transform, coords[0], coords[1])
                                    if 0 <= r < search_mask_hr_user.shape[0] and 0 <= c < search_mask_hr_user.shape[1]:
                                        is_inside_raster = search_mask_hr_user[r, c]
                                        raster_status = "DENTRO" if is_inside_raster else "FUERA"
                                    else:
                                        raster_status = "FUERA (Píxel fuera de límites)"
                                except Exception:
                                    raster_status = "ERROR"

                            print(f"Punto ID {point_id}: Vectorial -> {vector_status}, Raster -> {raster_status}")
                        print("--------------------------------------------------\n")


                    origin_coords = all_points.get(ORIGIN_POINT_ID)
                    if not origin_coords:
                        print(f"Error: No se encontró el punto de origen con ID={ORIGIN_POINT_ID} en el shapefile.")
                    else:
                        # --- VERIFICACIÓN CRÍTICA: Comprobar si el punto de ORIGEN está en la máscara ---
                        if mask_polygon_geom and not mask_polygon_geom.contains(Point(origin_coords)):
                            print(f"¡ERROR CRÍTICO! El punto de origen ID={ORIGIN_POINT_ID} está FUERA del polígono de la máscara. Abortando proceso.")
                        else:
                            start_pixel_hr = world_to_pixel(src.transform, origin_coords[0], origin_coords[1])
                            print(f"\nOrigen Fijo: Punto ID {ORIGIN_POINT_ID} -> Píxel {start_pixel_hr}")
                            
                            for dest_id, dest_coords in tqdm(all_points.items(), desc="Calculando rutas"):
                                if dest_id == ORIGIN_POINT_ID:
                                    continue

                                # --- VERIFICACIÓN: Omitir si el punto de DESTINO está fuera de la máscara ---
                                if mask_polygon_geom and not mask_polygon_geom.contains(Point(dest_coords)):
                                    tqdm.write(f"  -> Omitiendo punto de destino ID={dest_id} por estar FUERA del polígono de la máscara.")
                                    continue
                                
                                end_pixel_hr = world_to_pixel(src.transform, dest_coords[0], dest_coords[1])
                                
                                path_found_lr = False
                                path_pixels_lr = None
                                successful_factor = None
                                
                                # --- Bucle para probar diferentes factores de downsampling ---
                                for factor in DOWNSAMPLING_FACTORS:
                                    # Crear datos de baja resolución para el factor actual
                                    cost_data_low_res, trans_low, dx_low, dy_low = create_low_res_data(src, factor)
                                    
                                    # Reducir máscara de usuario si existe
                                    if search_mask_hr_user is not None:
                                        search_mask_low_res_user = search_mask_hr_user[::factor, ::factor]
                                    else:
                                        search_mask_low_res_user = np.ones(cost_data_low_res.shape, dtype=bool)

                                    start_pixel_lr = (start_pixel_hr[0] // factor, start_pixel_hr[1] // factor)
                                    end_pixel_lr = (end_pixel_hr[0] // factor, end_pixel_hr[1] // factor)
                                    
                                    current_path_found, came_from_lr, _ = a_star_numba_compiled(
                                        cost_data_low_res, src.nodata, start_pixel_lr, end_pixel_lr, dx_low, abs(dy_low), HEURISTIC_WEIGHT, search_mask_low_res_user
                                    )
                                    
                                    if current_path_found:
                                        path_found_lr = True
                                        path_pixels_lr = reconstruct_path_pixels_numba(came_from_lr, start_pixel_lr, end_pixel_lr)
                                        successful_factor = factor
                                        break # Salir del bucle de factores, ya encontramos una ruta
                                
                                if not path_found_lr:
                                    tqdm.write(f"  ADVERTENCIA: No se encontró ruta de bajo nivel para {ORIGIN_POINT_ID}->{dest_id} con factores {DOWNSAMPLING_FACTORS}")
                                    continue
                                
                                search_mask_hr_corridor = create_search_corridor(path_pixels_lr, cost_data_high_res.shape, successful_factor, CORRIDOR_BUFFER_PIXELS)
                                
                                # --- MODIFICACIÓN: Combinar máscara de corredor con máscara de usuario ---
                                if search_mask_hr_user is not None:
                                    final_search_mask_hr = np.logical_and(search_mask_hr_corridor, search_mask_hr_user)
                                else:
                                    final_search_mask_hr = search_mask_hr_corridor

                                path_found_hr, came_from_hr, g_cost_hr = a_star_numba_compiled(
                                    cost_data_high_res, src.nodata, start_pixel_hr, end_pixel_hr, src.res[0], abs(src.res[1]), HEURISTIC_WEIGHT, final_search_mask_hr
                                )
                                
                                if path_found_hr:
                                    path_pixels_hr = reconstruct_path_pixels_numba(came_from_hr, start_pixel_hr, end_pixel_hr)
                                    
                                    # Guardar ruta de Fase 1
                                    output_shp_p1 = os.path.join(OUTPUT_DIR, f"ruta_fase1_desde_{ORIGIN_POINT_ID}_a_{dest_id}.shp")
                                    save_path_to_shapefile(path_pixels_lr, trans_low, src.crs, output_shp_p1)
                                    
                                    # Guardar ruta Final
                                    output_shp_final = os.path.join(OUTPUT_DIR, f"ruta_final_desde_{ORIGIN_POINT_ID}_a_{dest_id}.shp")
                                    save_path_to_shapefile(path_pixels_hr, src.transform, src.crs, output_shp_final)
                                else:
                                    tqdm.write(f"  ADVERTENCIA: No se encontró ruta de alto nivel para {ORIGIN_POINT_ID}->{dest_id}")

                        print("\n¡Proceso de Uno a Todos completado!")

    except Exception as e:
        print(f"Ocurrió un error inesperado en el proceso principal: {e}")