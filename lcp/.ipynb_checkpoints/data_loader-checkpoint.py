# lcp/data_loader.py

import rasterio
import fiona
import os
from shapely.geometry import shape
from shapely.ops import unary_union

def load_raster(path):
    """
    Carga un archivo raster y devuelve el objeto del dataset.
    """
    print(f"Cargando raster desde: {os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo raster no se encontró en la ruta: {path}")
    return rasterio.open(path)

def load_points_as_dict(shapefile_path, id_field):
    """
    Carga puntos desde un shapefile o gpkg a un diccionario {id: (x, y)}.
    --- VERSIÓN FINAL CORREGIDA: Maneja 'fid' de GPKG sin error de 'seek' ---
    """
    points = {}
    crs = None
    print(f"Cargando puntos desde: {os.path.basename(shapefile_path)}")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"El archivo de puntos no se encontró en la ruta: {shapefile_path}")

    with fiona.open(shapefile_path, 'r') as c:
        if len(c) == 0:
            raise ValueError(f"El archivo de puntos '{os.path.basename(shapefile_path)}' está vacío.")
        crs = c.crs

        try:
            features_iterator = iter(c)
            first_feature = next(features_iterator)
        except StopIteration:
            return {}, crs

        use_feature_id = id_field not in first_feature['properties'] and id_field.lower() in ['fid', 'id']
        if use_feature_id:
            print(f"  ADVERTENCIA: Campo '{id_field}' no encontrado en propiedades. Se usará el 'feature.id' (fid del GeoPackage).")
        else:
            print(f"  Usando el campo de propiedad '{id_field}' como identificador.")

        def process_feature(feature):
            if use_feature_id:
                point_id = int(feature['id'])
            else:
                point_id = int(feature['properties'][id_field])
            coords = feature['geometry']['coordinates']
            points[point_id] = coords

        try:
            process_feature(first_feature)
        except KeyError:
            raise KeyError(f"El campo de ID especificado '{id_field}' no se encuentra en las propiedades. Revisa la configuración.")

        for feature in features_iterator:
            try:
                process_feature(feature)
            except KeyError:
                raise KeyError(f"El campo de ID especificado '{id_field}' no se encuentra en las propiedades. Revisa la configuración.")

    print(f"Se cargaron {len(points)} puntos.")
    return points, crs

def load_mask_geometry(shapefile_path):
    """Carga y unifica la geometría de un polígono de máscara."""
    if not shapefile_path or not os.path.exists(shapefile_path):
        return None
    with fiona.open(shapefile_path, 'r') as c:
        geoms = [shape(f['geometry']) for f in c]
    return unary_union(geoms)