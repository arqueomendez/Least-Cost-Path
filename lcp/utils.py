# lcp/utils.py

import fiona
import os
from fiona.crs import CRS
from shapely.geometry import LineString, mapping


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


def save_path_to_shapefile(pixel_path, transform, crs, output_path, verbose=True):
    """
    Saves a pixel path to a shapefile.
    Kept for backwards-compatibility; new code should use build_route_record().
    """
    if pixel_path is None or len(pixel_path) == 0:
        if verbose:
            print(
                f"No se guardará {os.path.basename(output_path)}, la ruta está vacía o es inválida."
            )
        return

    world_coords = [transform * (p[1] + 0.5, p[0] + 0.5) for p in pixel_path]

    if len(world_coords) < 2:
        if verbose:
            print(
                f"No se guardará {os.path.basename(output_path)}, la ruta necesita al menos 2 puntos."
            )
        return

    schema = {"geometry": "LineString", "properties": {"id": "str"}}
    fiona_crs = CRS.from_wkt(crs.to_wkt()) if crs else None

    with fiona.open(output_path, "w", "ESRI Shapefile", schema, crs=fiona_crs) as c:
        c.write(
            {
                "geometry": mapping(LineString(world_coords)),
                "properties": {"id": os.path.basename(output_path)},
            }
        )
    if verbose:
        print(f"Ruta guardada exitosamente en: {os.path.basename(output_path)}")
