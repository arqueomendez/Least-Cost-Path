# lcp/postprocessing.py

import os


def reconstruct_routes_from_disk(args):
    """
    Worker de Fase 2: carga el traceback comprimido desde disco y reconstruye
    todas las rutas desde un origen hacia sus destinos (dest_id > origin_id).

    Retorna (origin_id, records) o (origin_id, []) ante error.
    """
    origin_id, origin_coords, dest_items, cost_scalars, transform, phase1_dir = args

    try:
        import numpy as np

        import lcp.pathfinder as pf
        import lcp.processing as proc
        import lcp.utils as utils

        npz_path = os.path.join(phase1_dir, f"traceback_{origin_id}.npz")

        # Cerrar el NpzFile tras leer (np.load es perezoso: deja el ZIP abierto y
        # acumularía descriptores -> "Too many open files" con miles de orígenes).
        with np.load(npz_path) as data:
            traceback = data["traceback"].copy()
            row_offset = int(data["row_offset"]) if "row_offset" in data.files else 0
            col_offset = int(data["col_offset"]) if "col_offset" in data.files else 0
            if "offsets" in data.files:
                offsets = data["offsets"].astype(np.int32)
            else:
                # Respaldo para .npz generados por una versión anterior.
                offsets = pf.DEFAULT_OFFSETS.copy()

        start_pixel = proc.world_to_pixel(transform, origin_coords[0], origin_coords[1])
        start_pixel_local = (start_pixel[0] - row_offset, start_pixel[1] - col_offset)

        records = []
        for dest_id, dest_coords in dest_items:
            end_pixel = proc.world_to_pixel(transform, dest_coords[0], dest_coords[1])
            end_pixel_local = (end_pixel[0] - row_offset, end_pixel[1] - col_offset)

            path_pixels_local = pf.reconstruct_path_from_traceback(
                traceback, start_pixel_local, end_pixel_local, offsets
            )

            if path_pixels_local is None:
                continue

            path_pixels = path_pixels_local + np.array(
                [[row_offset, col_offset]], dtype=np.int32
            )

            cost_total = cost_scalars.get(dest_id, float("nan"))
            record = utils.build_route_record(
                path_pixels, transform, origin_id, dest_id, cost_total
            )
            if record is not None:
                records.append(record)

        del traceback
        return (origin_id, records)

    except Exception as exc:
        print(f"ERROR Fase 2, origen {origin_id}: {exc}")
        import traceback as tb

        tb.print_exc()
        return (origin_id, [])