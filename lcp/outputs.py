# lcp/outputs.py
"""Artefactos por raiz MCP: nombres canonicos (S5), manifiesto (S7) y
escritura atomica por archivo (S8).

Cada raiz MCP produce su propio juego de salidas (nunca se unifican). Este
modulo es puro de nombres + E/S de artefactos: no conoce MCP ni el raster.
"""

import csv
import os
import warnings

import geopandas as gpd

MANIFEST_NAME = "manifest_rutas_por_origen.csv"
MANIFEST_COLUMNS = (
    "root_id",
    "capa",
    "agrupacion",
    "esperados",
    "inalcanzables",
    "escritos",
    "estado",
    "gpkg",
    "mapa",
    "densificado",
    "heatmap",
)

# Orden de columnas de cada GPKG por raiz (S6). `origin_layer` / `dest_layer`
# solo aparecen si el registro las trae (esquema legacy intacto).
ROUTE_COLUMNS = (
    "origin_id",
    "dest_id",
    "origin_layer",
    "dest_layer",
    "cost_total",
    "length_m",
    "n_vertices",
)

_SUBDIRS = (
    "rutas_por_origen",
    "mapas_por_origen",
    "puntos_por_origen",
    "heatmaps_por_origen",
)


def session_paths(output_dir):
    """
    Creates the per-origin subdirectories under `output_dir` and returns the
    absolute paths of every artifact container plus the manifest file.

    Idempotente: volver a llamarla reutiliza los directorios existentes.
    """
    base = os.path.abspath(str(output_dir))
    paths = {}
    for name in _SUBDIRS:
        directory = os.path.join(base, name)
        os.makedirs(directory, exist_ok=True)
        paths[name] = directory
    paths["manifest"] = os.path.join(base, MANIFEST_NAME)
    return paths


def _fmt_number(value):
    """Renderiza un numero sin ceros finales cuando es integral (1.0 -> 1)."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)


def route_artifact(paths, root_id):
    """Ruta del GPKG de rutas de una raiz MCP."""
    return os.path.join(paths["rutas_por_origen"], f"rutas_origen_{root_id}.gpkg")


def figure_artifact(paths, root_id):
    """Ruta de la figura PNG de una raiz MCP."""
    return os.path.join(paths["mapas_por_origen"], f"mapa_origen_{root_id}.png")


def dense_artifact(paths, root_id):
    """Ruta del GPKG de puntos densificados de una raiz MCP."""
    return os.path.join(paths["puntos_por_origen"], f"puntos_origen_{root_id}.gpkg")


def heatmap_artifact(paths, root_id, res_m, bw_km, kernel):
    """Ruta del GeoTIFF del KDE de una raiz MCP."""
    res = _fmt_number(res_m)
    bw = _fmt_number(bw_km)
    return os.path.join(
        paths["heatmaps_por_origen"],
        f"heatmap_origen_{root_id}_{res}m_{bw}km_{kernel}.tif",
    )


def _ordered_route_columns(records):
    """Columnas presentes en los registros, en el orden de ROUTE_COLUMNS."""
    present = set()
    for record in records:
        present.update(record.keys())
    present.discard("geometry")
    ordered = [column for column in ROUTE_COLUMNS if column in present]
    ordered.extend(sorted(present - set(ROUTE_COLUMNS)))
    return [*ordered, "geometry"]


def _cell(value):
    """Normaliza un valor de manifiesto a celda CSV (None -> vacio)."""
    return "" if value is None else value


def write_origin_routes(records, paths, root_id, crs):
    """
    Writes the routes of one MCP root into its own GPKG (single layer named
    after the root) and returns `(path, n_written)`.

    Escritura atomica por archivo: se borra cualquier `.tmp` previo, se escribe
    siempre con `mode="w"` (nunca `mode="a"`, que duplicaria filas) y el archivo
    final solo aparece con `os.replace`. Sin registros no se crea nada.
    """
    path = route_artifact(paths, root_id)
    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if not records:
        return (path, 0)

    columns = _ordered_route_columns(records)
    gdf = gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=crs)
    layer = f"rutas_origen_{root_id}"

    # pyogrio avisa de la extension no conforme del `.tmp`; es esperado y local
    # a esta escritura (nunca un filtro global).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gdf.to_file(tmp_path, layer=layer, driver="GPKG", mode="w")

    os.replace(tmp_path, path)
    return (path, len(gdf))


def write_manifest(paths, rows):
    """
    Rewrites the per-root manifest atomically with the canonical header.
    Missing values are written as empty cells.
    """
    path = paths["manifest"]
    tmp_path = path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _cell(row.get(column)) for column in MANIFEST_COLUMNS}
            )

    os.replace(tmp_path, path)
    return path


def read_manifest(paths):
    """
    Reads the per-root manifest as a list of dicts (empty when it is missing).
    """
    path = paths["manifest"]
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {column: row.get(column, "") for column in MANIFEST_COLUMNS}
            for row in reader
        ]
