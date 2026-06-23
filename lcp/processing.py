# lcp/processing.py

import numpy as np
import fiona
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from pyproj import CRS


def _get_geometry(feature):
    return feature.geometry if hasattr(feature, "geometry") else feature["geometry"]


def world_to_pixel(transform, x, y):
    """
    Converts world coordinates to pixel (row, column) using the raster transform.
    """
    row, col = rasterio.transform.rowcol(transform, x, y)
    return int(row), int(col)


def create_mask_from_vector(vector_path, raster_src):
    """
    Crea una máscara booleana a partir de un vector, con limpieza automática de
    geometrías. La comparación de CRS se hace con pyproj para evitar falsos
    positivos al comparar objetos CRS de clases distintas (fiona vs rasterio).
    """
    print("Creando máscara booleana desde el polígono (con limpieza de geometría)...")
    geometries_to_rasterize = []
    repaired_count = 0

    raster_crs = CRS.from_user_input(raster_src.crs)

    with fiona.open(vector_path, "r") as vf:
        vector_crs = CRS.from_user_input(vf.crs)
        if not raster_crs.equals(vector_crs):
            raise ValueError(
                f"Discrepancia de CRS. Raster: {raster_crs.to_string()}, "
                f"Vector: {vector_crs.to_string()}"
            )
        for feature in vf:
            geom_mapping = _get_geometry(feature)
            if not geom_mapping:
                continue
            geom = shape(geom_mapping)
            if geom.is_valid and not geom.is_empty:
                geometries_to_rasterize.append(geom)
            elif not geom.is_valid:
                repaired_geom = geom.buffer(0)
                if repaired_geom.is_valid and not repaired_geom.is_empty:
                    geometries_to_rasterize.append(repaired_geom)
                    repaired_count += 1

    if repaired_count > 0:
        print(
            f"  ADVERTENCIA: Se han reparado {repaired_count} geometrías inválidas "
            f"en el archivo de máscara."
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