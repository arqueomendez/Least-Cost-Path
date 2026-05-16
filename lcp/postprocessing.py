# lcp/postprocessing.py

import os


def reconstruct_routes_from_disk(args):
    """
    Worker de Fase 2: carga el traceback comprimido desde disco y reconstruye
    todas las rutas desde un origen hacia sus destinos (dest_id > origin_id).

    Parámetros (empaquetados en args):
        origin_id       : identificador del punto de origen.
        origin_coords   : (x, y) en coordenadas mundo del origen.
        dest_items      : lista de (dest_id, dest_coords) con dest_id > origin_id.
        cost_scalars    : dict {dest_id: float} con el coste acumulado hasta cada destino.
        transform       : objeto rasterio.transform del raster de coste.
        phase1_dir      : directorio donde están los traceback_<origin_id>.npz.

    Retorna:
        (origin_id, records) donde records es una lista de dicts de ruta.
        En caso de error retorna (origin_id, []).
    """
    origin_id, origin_coords, dest_items, cost_scalars, transform, phase1_dir = args

    try:
        import numpy as np

        import lcp.pathfinder as pf
        import lcp.processing as proc
        import lcp.utils as utils

        npz_path = os.path.join(phase1_dir, f"traceback_{origin_id}.npz")
        data = np.load(npz_path)
        traceback = data["traceback"]
        # Offsets del recorte de bounding-box aplicado en Fase 1.
        # Compatibilidad hacia atrás: si el .npz fue generado sin recorte
        # (versión anterior del código) los offsets son 0.
        row_offset = int(data["row_offset"]) if "row_offset" in data.files else 0
        col_offset = int(data["col_offset"]) if "col_offset" in data.files else 0

        start_pixel = proc.world_to_pixel(transform, origin_coords[0], origin_coords[1])
        # Convertir a coordenadas locales del sub-raster.
        start_pixel_local = (start_pixel[0] - row_offset, start_pixel[1] - col_offset)

        records = []
        for dest_id, dest_coords in dest_items:
            end_pixel = proc.world_to_pixel(transform, dest_coords[0], dest_coords[1])
            end_pixel_local = (end_pixel[0] - row_offset, end_pixel[1] - col_offset)

            path_pixels_local = pf.reconstruct_path_from_traceback(
                traceback, start_pixel_local, end_pixel_local
            )

            if path_pixels_local is None:
                continue

            # Convertir de vuelta a coordenadas del raster completo antes de
            # calcular las coordenadas mundo con el transform original.
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
