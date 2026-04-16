# lcp/data_loader.py

import rasterio
import fiona
import os
from shapely.geometry import shape
from shapely.ops import unary_union


def load_raster(path):
    """
    Loads a raster file and returns the open dataset object.
    """
    print(f"Cargando raster desde: {os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo raster no se encontró en la ruta: {path}")
    return rasterio.open(path)


def load_points_as_dict(shapefile_path, id_field):
    """
    Loads points from a shapefile or GPKG into a dict {id: (x, y)}.
    Falls back to using the feature's fid when id_field is absent from properties.
    """
    points = {}
    crs = None
    print(f"Cargando puntos desde: {os.path.basename(shapefile_path)}")
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(
            f"El archivo de puntos no se encontró en la ruta: {shapefile_path}"
        )

    with fiona.open(shapefile_path, "r") as c:
        if len(c) == 0:
            raise ValueError(
                f"El archivo de puntos '{os.path.basename(shapefile_path)}' está vacío."
            )
        crs = c.crs

        features = list(c)
        if not features:
            return {}, crs

        use_feature_id = id_field not in features[0][
            "properties"
        ] and id_field.lower() in ["fid", "id"]
        if use_feature_id:
            print(
                f"  ADVERTENCIA: Campo '{id_field}' no encontrado en propiedades. "
                f"Se usará el 'feature.id' (fid del GeoPackage)."
            )
        else:
            print(f"  Usando el campo de propiedad '{id_field}' como identificador.")

        for feature in features:
            try:
                point_id = (
                    int(feature["id"])
                    if use_feature_id
                    else int(feature["properties"][id_field])
                )
            except KeyError:
                raise KeyError(
                    f"El campo de ID especificado '{id_field}' no se encuentra en las propiedades. "
                    f"Revisa la configuración."
                )
            points[point_id] = feature["geometry"]["coordinates"]

    print(f"Se cargaron {len(points)} puntos.")
    return points, crs


def load_mask_geometry(shapefile_path):
    """Loads and unifies the geometry of a mask polygon file."""
    if not shapefile_path or not os.path.exists(shapefile_path):
        return None
    with fiona.open(shapefile_path, "r") as c:
        geoms = [shape(f["geometry"]) for f in c]
    return unary_union(geoms)
