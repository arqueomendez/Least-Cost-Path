# lcp/utils.py

from shapely.geometry import LineString


def build_route_record(
    pixel_path,
    transform,
    origin_id,
    dest_id,
    cost_total,
    origin_layer=None,
    dest_layer=None,
):
    """
    Converts a pixel path to a dict with geometry and attributes.
    Returns None if the path is invalid or too short.

    'length_m' es la longitud de la polilínea en unidades del CRS (asume un CRS
    métrico proyectado, p. ej. UTM; en un CRS geográfico saldría en grados) y
    'cost_total' es el coste acumulado mínimo que entrega el llamador.

    S6: 'origin_layer' y 'dest_layer' se agregan SOLO cuando no son None, de
    modo que sin ellos el registro conserva exactamente las seis claves
    históricas (esquema legacy intacto: mismos valores, mismo orden).
    """
    if pixel_path is None or len(pixel_path) < 2:
        return None

    world_coords = [transform * (p[1] + 0.5, p[0] + 0.5) for p in pixel_path]
    if len(world_coords) < 2:
        return None

    line = LineString(world_coords)
    record = {
        "geometry": line,
        "origin_id": int(origin_id),
        "dest_id": int(dest_id),
        "cost_total": float(cost_total),
        "length_m": float(line.length),
        "n_vertices": len(pixel_path),
    }
    if origin_layer is not None:
        record["origin_layer"] = origin_layer
    if dest_layer is not None:
        record["dest_layer"] = dest_layer
    return record
