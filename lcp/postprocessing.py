# lcp/postprocessing.py

import lcp.pathfinder as pf
import lcp.utils as utils
import lcp.processing as proc


def extract_routes(
    traceback,
    costs_surface,
    origin_id,
    origin_coords,
    all_points,
    transform,
    verbose=False,
):
    """
    Reconstructs all routes from one origin to every other point using the
    traceback array from pf.cost_surface_calculator().

    Returns a list of route-record dicts (see utils.build_route_record).
    Skips the origin itself and any unreachable destinations silently.
    """
    records = []
    start_pixel = proc.world_to_pixel(transform, origin_coords[0], origin_coords[1])

    if verbose:
        print(
            f"  [postp] Reconstruyendo rutas desde origen {origin_id} (píxel {start_pixel})"
        )

    for dest_id, dest_coords in all_points.items():
        if dest_id <= origin_id:
            continue

        end_pixel = proc.world_to_pixel(transform, dest_coords[0], dest_coords[1])
        path_pixels = pf.reconstruct_path_from_traceback(
            traceback, start_pixel, end_pixel
        )

        if path_pixels is None:
            if verbose:
                print(f"    -> destino {dest_id}: inalcanzable, omitido.")
            continue

        end_row, end_col = end_pixel
        cost_total = float(costs_surface[end_row, end_col])

        record = utils.build_route_record(
            path_pixels, transform, origin_id, dest_id, cost_total
        )
        if record is not None:
            records.append(record)
            if verbose:
                print(
                    f"    -> destino {dest_id}: OK  (coste={cost_total:.4f}, vértices={record['n_vertices']})"
                )

    return records
