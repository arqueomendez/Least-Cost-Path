# lcp/processing.py

import numpy as np
import fiona
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from skimage.draw import disk
from shapely.geometry import shape


def world_to_pixel(transform, x, y):
    """
    Converts world coordinates to pixel (row, column) using the raster transform.
    """
    row, col = rasterio.transform.rowcol(transform, x, y)
    return int(row), int(col)


def create_search_corridor(path_low_res, high_res_shape, factor, buffer_pixels):
    """
    Crea una máscara de corredor dibujando discos alrededor de la ruta de baja resolución.
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
    """
    Crea una máscara booleana a partir de un vector, con limpieza automática de geometrías.
    """
    print("Creando máscara booleana desde el polígono (con limpieza de geometría)...")
    geometries_to_rasterize = []
    repaired_count = 0
    with fiona.open(vector_path, "r") as vf:
        if vf.crs != raster_src.crs:
            raise ValueError(
                f"Discrepancia de CRS. Raster: {raster_src.crs}, Vector: {vf.crs}"
            )
        for feature in vf:
            if not feature or not feature.get("geometry"):
                continue
            geom = shape(feature["geometry"])
            if geom.is_valid and not geom.is_empty:
                geometries_to_rasterize.append(geom)
            elif not geom.is_valid:
                repaired_geom = geom.buffer(0)
                if repaired_geom.is_valid and not repaired_geom.is_empty:
                    geometries_to_rasterize.append(repaired_geom)
                    repaired_count += 1
    if repaired_count > 0:
        print(
            f"  ADVERTENCIA: Se han reparado {repaired_count} geometrías inválidas en el archivo de máscara."
        )
    if not geometries_to_rasterize:
        raise ValueError("El archivo de máscara no contiene geometrías válidas.")
    mask = rasterize(
        geometries_to_rasterize,
        out_shape=raster_src.shape,
        transform=raster_src.transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    print("Máscara creada exitosamente.")
    return mask.astype(bool)
