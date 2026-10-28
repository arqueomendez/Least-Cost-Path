# lcp/data_loader.py

import os

import fiona
import rasterio
from shapely.geometry import MultiPoint, Point, shape
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Helpers de compatibilidad fiona 1.9+ (acceso dict-style está deprecado y se
# elimina en fiona 2.0). Funcionan tanto con objetos Feature como con dicts.
# ---------------------------------------------------------------------------
def _get_properties(feature):
    return (
        feature.properties if hasattr(feature, "properties") else feature["properties"]
    )


def _get_id(feature):
    return feature.id if hasattr(feature, "id") else feature["id"]


def _get_geometry(feature):
    return feature.geometry if hasattr(feature, "geometry") else feature["geometry"]


def load_raster(path):
    """
    Loads a raster file and returns the open dataset object.
    """
    print(f"Cargando raster desde: {os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo raster no se encontró en la ruta: {path}")
    return rasterio.open(path)


def _split_point_geometry(geom, point_id):
    """
    Returns the list of ((id, (x, y))) nodes produced by one geometry.

    Point -> el propio id; MultiPoint de 1 parte -> el propio id; MultiPoint de
    k >= 2 partes -> ids derivados ``point_id * 1000 + j`` (j = 1..k), en el
    orden de la geometria. Solo se usan x, y (se ignora Z).
    """
    if isinstance(geom, Point):
        return [(point_id, (geom.x, geom.y))]

    if isinstance(geom, MultiPoint):
        parts = list(geom.geoms)
        if len(parts) <= 1:
            if not parts:
                raise ValueError(
                    f"La geometria del id {point_id} es un MultiPoint vacio; "
                    f"no aporta ningun nodo."
                )
            return [(point_id, (parts[0].x, parts[0].y))]
        if len(parts) > 999:
            raise ValueError(
                f"La feature con id {point_id} tiene {len(parts)} partes en su "
                f"MultiPoint; el maximo soportado es 999 (ids derivados "
                f"original*1000+parte)."
            )
        return [
            (point_id * 1000 + j, (part.x, part.y))
            for j, part in enumerate(parts, start=1)
        ]

    geom_type = geom.geom_type if geom is not None else "None"
    raise ValueError(
        f"Geometria no soportada para el id {point_id}: '{geom_type}'. "
        f"Solo se admiten geometrias Point y MultiPoint; convierte la capa "
        f"antes de cargarla."
    )


def load_points_as_dict(shapefile_path, id_field):
    """
    Loads points from a shapefile or GPKG into a dict {id: (x, y)}.
    Falls back to using the feature's fid when id_field is absent from properties.
    Compatible con fiona 1.9+ (acceso por atributo en lugar de dict-style).

    MultiPoint: una feature con k >= 2 coordenadas se explota en k nodos con
    ids derivados ``id_original * 1000 + j`` (j = 1..k, k <= 999).
    """
    points = {}
    duplicated = []
    collisions = []
    derived_ids = set()
    exploded_features = 0
    exploded_nodes = 0
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

        first_props = _get_properties(features[0])
        use_feature_id = id_field not in first_props and id_field.lower() in [
            "fid",
            "id",
        ]
        if use_feature_id:
            print(
                f"  ADVERTENCIA: Campo '{id_field}' no encontrado en propiedades. "
                f"Se usará el 'feature.id' (fid del GeoPackage)."
            )
        else:
            print(f"  Usando el campo de propiedad '{id_field}' como identificador.")

        for feature in features:
            props = _get_properties(feature)
            try:
                point_id = (
                    int(_get_id(feature)) if use_feature_id else int(props[id_field])
                )
            except KeyError:
                raise KeyError(
                    f"El campo de ID especificado '{id_field}' no se encuentra en las "
                    f"propiedades. Revisa la configuración."
                )

            # shape() normaliza Geometry de fiona 1.9 o dict-geojson antiguo.
            geom = shape(_get_geometry(feature))
            nodes = _split_point_geometry(geom, point_id)
            is_explosion = len(nodes) > 1
            if is_explosion:
                exploded_features += 1
                exploded_nodes += len(nodes)

            for node_id, coords in nodes:
                if node_id in points:
                    # Un id derivado nunca puede repetirse: rompe la unicidad
                    # exigida por npz (dest_ids int64), el GPKG y `did > origin_id`.
                    if is_explosion or node_id in derived_ids:
                        if node_id not in collisions:
                            collisions.append(node_id)
                        continue
                    if node_id not in duplicated:
                        duplicated.append(node_id)
                    continue
                points[node_id] = coords
                if is_explosion:
                    derived_ids.add(node_id)

        if duplicated:
            raise ValueError(
                f"IDs duplicados en '{os.path.basename(shapefile_path)}': "
                f"{sorted(duplicated)}. Cada punto debe tener un ID unico."
            )

        if collisions:
            raise ValueError(
                f"Colision de IDs derivados en "
                f"'{os.path.basename(shapefile_path)}': {sorted(collisions)}. "
                f"Los ids derivados (original*1000+parte) deben ser unicos: "
                f"renombra los ids originales que colisionen."
            )

        if exploded_features:
            print(
                f"  ADVERTENCIA: {exploded_features} feature(s) MultiPoint explotadas "
                f"en {exploded_nodes} puntos "
                f"(ids derivados = original*1000+parte)"
            )

    print(f"Se cargaron {len(points)} puntos.")
    return points, crs


def load_mask_geometry(shapefile_path):
    """Loads and unifies the geometry of a mask polygon file."""
    if not shapefile_path or not os.path.exists(shapefile_path):
        return None
    with fiona.open(shapefile_path, "r") as c:
        geoms = [shape(_get_geometry(f)) for f in c]
    if not geoms:
        return None
    return unary_union(geoms)
