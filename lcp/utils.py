# lcp/utils.py

from shapely.geometry import LineString


def build_route_record(pixel_path, transform, origin_id, dest_id, cost_total):
    """
    Converts a pixel path to a dict with geometry and attributes.
    Returns None if the path is invalid or too short.
    """
    if pixel_path is None or len(pixel_path) < 2:
        return None

    world_coords = [transform * (p[1] + 0.5, p[0] + 0.5) for p in pixel_path]
    if len(world_coords) < 2:
        return None

    line = LineString(world_coords)
    return {
        "geometry": line,
        "origin_id": int(origin_id),
        "dest_id": int(dest_id),
        "cost_total": float(cost_total),
        "length_m": float(line.length),
        "n_vertices": int(len(pixel_path)),
    }
