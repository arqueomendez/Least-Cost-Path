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


def _normalize_value(value):
    """Normalizes a selector value for type-tolerant comparison (S2)."""
    if isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    try:
        num = float(text)
    except ValueError:
        return text.lower()
    # Una sola rama numerica para TODOS los tipos (int/float/str): los enteros
    # se conservan exactos (2, 2.0 y "2" -> "2"; ids de 7+ digitos intactos)
    # y los no enteros usan .17g, que no recorta digitos significativos.
    if num.is_integer():
        return str(int(num))
    return f"{num:.17g}"


def load_point_candidates(path, id_field="id"):
    """
    Loads every point feature as a candidate dict for preview/selection.

    Devuelve una lista de ``{id, fid, coords, properties}`` ordenada por id,
    donde ``id`` sigue la misma regla de explosion MultiPoint de
    ``load_points_as_dict`` (ids derivados ``original*1000+j``) y ``fid`` es
    el feature.id real del archivo. Pensado para el preview de la Celda 4 y
    para ``select_point``: nunca adivina ni filtra.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"El archivo de puntos no se encontró en la ruta: {path}"
        )
    candidates = []
    with fiona.open(path, "r") as c:
        if len(c) == 0:
            raise ValueError(
                f"El archivo de puntos '{os.path.basename(path)}' está vacío."
            )
        for feature in c:
            props = dict(_get_properties(feature))
            fid = int(_get_id(feature))
            geom = shape(_get_geometry(feature))
            # El id de cada nodo sigue la regla de explosion del loader
            # principal: Point/MultiPoint de 1 parte -> id del atributo
            # (o fid si el campo no existe, mismo fallback de compatibilidad);
            # MultiPoint de k partes -> ids derivados.
            use_fid = id_field not in props and id_field.lower() in ["fid", "id"]
            base_id = fid if use_fid else props.get(id_field)
            if base_id is None:
                raise KeyError(
                    f"El campo de ID especificado '{id_field}' no se encuentra "
                    "en las propiedades."
                )
            base_id = int(base_id)
            for node_id, coords in _split_point_geometry(geom, base_id):
                candidates.append(
                    {
                        "id": node_id,
                        "fid": fid,
                        "coords": coords,
                        "properties": props,
                    }
                )
    candidates.sort(key=lambda cand: cand["id"])
    return candidates


def select_point(candidates, field, value):
    """
    Selects ONE candidate by attribute value; strict, never falls back to fid.

    ``candidates`` es la lista de ``load_point_candidates``. La comparacion es
    tolerante a tipo (int/float/str normalizados, S2). Campo inexistente o
    valor no encontrado -> ValueError con las opciones disponibles; el fid
    solo se usa si el usuario pide explicitamente ``field='fid'``.
    """
    if not candidates:
        raise ValueError("La lista de candidatos de puntos esta vacia.")

    if field == "fid":
        matches = [c for c in candidates if c["fid"] == int(value)]
        if not matches:
            fids = sorted(c["fid"] for c in candidates)
            raise ValueError(
                f"No hay ningun punto con fid {value!r}. fids disponibles: {fids}"
            )
        return matches[0]

    available_fields = sorted({key for c in candidates for key in c["properties"]})
    if field not in available_fields:
        raise ValueError(
            f"El campo '{field}' no existe en la capa de puntos. "
            f"Campos disponibles: {available_fields}. "
            "ATENCION: el 'fid' no es un atributo; si queres seleccionar por "
            "fid usa explicitamente field='fid'."
        )

    target = _normalize_value(value)
    matches = [
        c for c in candidates if _normalize_value(c["properties"].get(field)) == target
    ]
    if not matches:
        available_values = sorted(
            {
                str(c["properties"].get(field))
                for c in candidates
                if c["properties"].get(field) is not None
            }
        )
        raise ValueError(
            f"Ningun punto tiene {field}={value!r}. "
            f"Valores disponibles de '{field}': {available_values}."
        )
    if len(matches) > 1:
        matched_ids = sorted(c["id"] for c in matches)
        raise ValueError(
            f"El valor {field}={value!r} corresponde a {len(matches)} puntos "
            f"(ids {matched_ids}): la seleccion debe ser unica. Refina el "
            "criterio (ej. otro campo) o usa el id exacto."
        )
    return matches[0]


def load_mask_geometry(shapefile_path):
    """Loads and unifies the geometry of a mask polygon file."""
    if not shapefile_path or not os.path.exists(shapefile_path):
        return None
    with fiona.open(shapefile_path, "r") as c:
        geoms = [shape(_get_geometry(f)) for f in c]
    if not geoms:
        return None
    return unary_union(geoms)
