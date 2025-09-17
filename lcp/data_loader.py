# lcp/data_loader.py

import rasterio
import fiona
import os
from shapely.geometry import shape
from shapely.ops import unary_union

def load_raster(path):
    """
    Carga un archivo raster y devuelve el objeto del dataset, que es
    compatible con el 'context manager' (la declaración 'with').
    """
    print(f"Cargando raster desde: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo raster no se encontró en la ruta: {path}")
    
    # Esta función devuelve el objeto 'dataset' que SÍ es compatible con 'with'
    dataset = rasterio.open(path)
    
    print("Raster cargado exitosamente.")
    return dataset # <-- CORRECCIÓN: Devolver solo el dataset

def load_points_as_dict(shapefile_path, id_field):
    """
    Carga puntos desde un shapefile a un diccionario {id: (x, y)}.
    """
    points = {}
    crs = None
    print(f"Cargando puntos desde: {shapefile_path}")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"El archivo de puntos no se encontró en la ruta: {shapefile_path}")
    
    with fiona.open(shapefile_path, 'r') as c:
        if len(c) == 0:
            raise ValueError(f"El archivo de puntos '{os.path.basename(shapefile_path)}' está vacío.")
        crs = c.crs
        for feature in c:
            point_id = int(feature['properties'][id_field])
            coords = feature['geometry']['coordinates']
            points[point_id] = coords
    print(f"Se cargaron {len(points)} puntos.")
    return points, crs

def load_mask_geometry(shapefile_path):
    """Carga y unifica la geometría de un polígono de máscara."""
    if not shapefile_path or not os.path.exists(shapefile_path):
        return None
    print("Cargando polígono de máscara...")
    with fiona.open(shapefile_path, 'r') as c:
        geoms = [shape(f['geometry']) for f in c]
    return unary_union(geoms)