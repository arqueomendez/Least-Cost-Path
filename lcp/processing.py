# lcp/processing.py

import fiona
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import rasterize
from shapely.geometry import shape


def _get_geometry(feature):
    return feature.geometry if hasattr(feature, "geometry") else feature["geometry"]


def edge_frame_mask(shape, buffer_px):
    """
    Devuelve una máscara booleana con ``True`` en el MARCO de píxeles a excluir:
    las ``buffer_px`` filas/columnas exteriores del ráster.

    Motivo: el modelo de coste puede no ser fiable cerca de los límites del
    ráster (extensiones mínimas, interpolaciones de borde, coberturas vecinas) y
    ser mucho más barato que el interior. Si el polígono de máscara cubre todo el
    ráster, ese marco queda navegable y las rutas lo usan como corredor
    perimetral. Con ``buffer_px=0`` la máscara es toda ``False`` (sin exclusión).
    """
    rows, cols = int(shape[0]), int(shape[1])
    b = max(0, int(buffer_px))
    frame = np.zeros((rows, cols), dtype=bool)
    if b == 0:
        return frame
    r = min(b, rows)
    c = min(b, cols)
    frame[:r, :] = True
    frame[rows - r :, :] = True
    frame[:, :c] = True
    frame[:, cols - c :] = True
    return frame


def world_to_pixel(transform, x, y):
    """
    Converts world coordinates to pixel (row, column) using the raster transform.
    """
    row, col = rasterio.transform.rowcol(transform, x, y)
    return int(row), int(col)


def create_mask_from_vector(vector_path, raster_src, all_touched=False):
    """
    Crea una máscara booleana a partir de un vector, con limpieza automática de
    geometrías. La comparación de CRS se hace con pyproj para evitar falsos
    positivos al comparar objetos CRS de clases distintas (fiona vs rasterio).

    ``all_touched`` decide qué píxeles del borde entran en la máscara:

    - ``False`` (por defecto, máscara ESTRICTA): solo los píxeles cuyo CENTRO cae
      dentro del polígono. Es lo correcto para buscar rutas: con ``True`` la
      máscara se expande a todo píxel que el polígono apenas roza, y ese anillo
      exterior conserva el valor del modelo de coste. Si el modelo es barato
      fuera del área de estudio, el anillo forma un camino continuo de bajo
      coste que rodea el polígono: las rutas se pegan al borde en vez de cruzar
      el interior.
    - ``True``: incluye además los píxeles tocados. Útil solo cuando hay puntos
      legítimos justo sobre el límite del polígono y se acepta el sesgo de borde.
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
        all_touched=bool(all_touched),
        dtype=np.uint8,
    )
    print("Máscara creada exitosamente.")
    return mask.astype(bool)
