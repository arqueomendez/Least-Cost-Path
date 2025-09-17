# lcp/processing.py

import numpy as np
import fiona  # <-- CORRECCIÓN: Se ha añadido la importación que faltaba
from rasterio.enums import Resampling
from rasterio.features import rasterize
from skimage.draw import disk

def world_to_pixel(transform, x, y):
    """Convierte coordenadas del mundo a píxel."""
    col, row = ~transform * (x, y)
    return int(row), int(col)

def create_low_res_data(src_dataset, factor):
    """Crea una versión de baja resolución del raster."""
    low_res_shape = (src_dataset.height // factor, src_dataset.width // factor)
    low_res_data = src_dataset.read(1, out_shape=low_res_shape, resampling=Resampling.average)
    low_res_transform = src_dataset.transform * src_dataset.transform.scale(factor, factor)
    dx_low = src_dataset.res[0] * factor
    dy_low = src_dataset.res[1] * factor
    return low_res_data, low_res_transform, dx_low, dy_low

def create_search_corridor(path_low_res, high_res_shape, factor, buffer_pixels):
    """
    Crea una máscara de corredor dibujando discos alrededor de la ruta de baja resolución.
    Fiel a la implementación original con skimage.
    """
    corridor_mask = np.zeros(high_res_shape, dtype=bool)
    if path_low_res is None or len(path_low_res) == 0:
        return corridor_mask
    
    for r_low, c_low in path_low_res:
        r_high = int(r_low * factor + factor / 2)
        c_high = int(c_low * factor + factor / 2)
        rr, cc = disk((r_high, c_high), buffer_pixels, shape=high_res_shape)
        corridor_mask[rr, cc] = True
        
    return corridor_mask

def create_mask_from_vector(vector_path, raster_src):
    """Crea una máscara booleana a partir de un shapefile."""
    print("Creando máscara booleana desde el polígono...")
    with fiona.open(vector_path, "r") as vf:
        if vf.crs != raster_src.crs:
            raise ValueError(f"El CRS de la máscara ({vf.crs}) y el raster ({raster_src.crs}) no coinciden.")
        shapes = [f["geometry"] for f in vf]
    
    mask = rasterize(shapes, out_shape=raster_src.shape, transform=raster_src.transform, fill=0, all_touched=True, dtype=np.uint8)
    print("Máscara creada.")
    return mask.astype(bool)