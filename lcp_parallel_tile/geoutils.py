# lcp_parallel_tile/geoutils.py

import os
import numpy as np
import rasterio
import fiona
import geopandas as gpd
from fiona.crs import CRS
from shapely.geometry import LineString, mapping, shape
from shapely.ops import unary_union
from skimage.draw import disk
from tqdm import tqdm
import glob
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rasterio.enums import Resampling
from rasterio.features import rasterize # <-- AÑADIDO
from . import tasks

# ... (otras funciones como create_low_res_data, etc. no cambian) ...
def create_low_res_data(cost_data_high_res, factor):
    return cost_data_high_res[::factor, ::factor]

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
    if pixel_path is None or not pixel_path.any(): return
    world_coords = [transform * (p[1] + 0.5, p[0] + 0.5) for p in pixel_path]
    schema = {'geometry': 'LineString', 'properties': {'id': 'str'}}
    with fiona.open(output_path, 'w', 'ESRI Shapefile', schema, crs=CRS.from_wkt(crs.to_wkt()) if crs else None) as c:
        c.write({'geometry': mapping(LineString(world_coords)), 'properties': {'id': os.path.basename(output_path)}})

def merge_results(session_dir, final_output_path, log):
    log.info("Fase final: Fusionando todos los segmentos de ruta generados...")
    search_pattern = os.path.join(session_dir, "ruta_*.shp")
    shapefile_list = glob.glob(search_pattern)
    if not shapefile_list:
        log.warning("No se encontraron archivos de ruta para fusionar.")
        return
    gdfs = [gpd.read_file(shp) for shp in tqdm(shapefile_list, desc="Cargando segmentos")]
    merged_gdf = gpd.pd.concat(gdfs, ignore_index=True)
    merged_gdf.to_file(final_output_path, driver='GPKG')
    log.info(f"¡Éxito! Red de flujo completa guardada en: {final_output_path}")

def load_polygon_geometry(shapefile_path, log):
    if not shapefile_path or not os.path.exists(shapefile_path):
        log.info("No se proporcionó una ruta de máscara o el archivo no existe. El análisis continuará sin máscara.")
        return None
    try:
        log.info(f"Cargando polígono de máscara desde: {os.path.basename(shapefile_path)}")
        with fiona.open(shapefile_path, 'r') as c:
            if len(c) == 0:
                log.warning(f"El archivo de máscara '{os.path.basename(shapefile_path)}' está vacío.")
                return None
            return unary_union([shape(f['geometry']) for f in c])
    except Exception as e:
        log.error(f"Error crítico al cargar el polígono de la máscara: {e}")
        return None

# --- [INICIO DE CORRECCIÓN 2] ---
def create_mask_from_vector(vector_path, raster_src, window=None):
    """
    Crea una máscara booleana a partir de un shapefile.
    Ahora acepta un argumento 'window' opcional para rasterizar solo una porción.
    """
    if not vector_path or not os.path.exists(vector_path):
        logging.warning("No se encontró archivo de máscara."); return None
    with fiona.open(vector_path, "r") as vf:
        if vf.crs != raster_src.crs:
            logging.critical(f"CRS de máscara ({vf.crs}) y ráster ({raster_src.crs}) no coinciden."); return None
        shapes = [f["geometry"] for f in vf]
    
    # Si se proporciona una ventana, ajustamos la forma y la transformación
    if window:
        transform = raster_src.window_transform(window)
        out_shape = (window.height, window.width)
    else:
        transform = raster_src.transform
        out_shape = raster_src.shape

    mask = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, all_touched=True, dtype=np.uint8)
    return mask.astype(bool)
# --- [FIN DE CORRECCIÓN 2] ---

def visualize_tiling_setup(config):
    # (El resto de esta función no necesita cambios)
    print("Generando previsualización de la segmentación del raster...")
    temp_log = logging.getLogger("TempLogger"); temp_log.setLevel(logging.CRITICAL)
    preview_tasks = tasks.prepare_tasks(config, temp_log)
    with rasterio.open(config.COST_RASTER_PATH) as src:
        preview_factor = max(1, src.width // 200, src.height // 200)
        bg_raster = src.read(1, out_shape=(src.height // preview_factor, src.width // preview_factor), resampling=Resampling.bilinear)
    all_nodes_r, all_nodes_c = [], []
    for task in preview_tasks:
        window = task['window']
        for r_local, c_local in task['border_nodes'].values():
            all_nodes_r.append(window.row_off + r_local)
            all_nodes_c.append(window.col_off + c_local)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(bg_raster, cmap='Greys_r', extent=[0, bg_raster.shape[1], bg_raster.shape[0], 0])
    for task in preview_tasks:
        win = task['window']
        rect = patches.Rectangle((win.col_off / preview_factor, win.row_off / preview_factor),
                                 win.width / preview_factor, win.height / preview_factor,
                                 linewidth=1, edgecolor='r', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
    ax.scatter(np.array(all_nodes_c) / preview_factor, np.array(all_nodes_r) / preview_factor, s=5, c='blue', alpha=0.6)
    ax.set_title(f"Previsualización de Tiling y Nodos\n({len(preview_tasks)} tiles, ~{len(all_nodes_r)} nodos)", fontsize=16)
    ax.set_xlabel("Coordenada X (píxeles de baja resolución)"); ax.set_ylabel("Coordenada Y (píxeles de baja resolución)")
    ax.set_aspect('equal'); plt.show()