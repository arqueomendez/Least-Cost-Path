"""Harness para ejecutar subconjuntos de celdas del notebook con nbclient.

Permite testear la logica de las celdas (p. ej. la validacion de la Celda 4)
con fixtures propias, parcheando la Celda 3 de configuracion, sin correr el
pipeline completo.
"""

import re
from pathlib import Path

import fiona
import nbformat
import numpy as np
import rasterio
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]


def load_code_sources():
    nb = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def patch_cell3(src, replacements, tag="cell3"):
    """Aplica reemplazos exactos a la Celda 3; falla si un patron no existe."""
    for old, new in replacements.items():
        if old not in src:
            raise AssertionError(f"[{tag}] fragmento no encontrado: {old!r}")
        src = src.replace(old, new, 1)
    return src


def run_code(sources, timeout=240):
    """Ejecuta las celdas dadas; devuelve (texto_de_streams, error|None)."""
    cells = [nbformat.v4.new_code_cell(s) for s in sources]
    mini = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    client = NotebookClient(
        mini,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    err = None
    try:
        client.execute()
    except CellExecutionError as exc:
        err = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    texts = []
    for cell in mini.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                texts.append(out.get("text", ""))
    return "".join(texts), err


def write_synthetic_raster(path, nodata_cols_from=50):
    """Raster 100x100, EPSG:32719, 1 m: valores 1.0 y nodata (-9999) en las
    columnas >= nodata_cols_from (para probar puntos sobre pixeles invalidos)."""
    data = np.ones((100, 100), dtype=np.float32)
    data[:, nodata_cols_from:] = -9999.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32719",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def write_barrier_raster(path, barrier_col=50):
    """Raster 100x100, EPSG:32719, 1 m: solo la columna barrier_col es nodata
    (-9999), partiendo la grilla en dos regiones validas desconectadas."""
    data = np.ones((100, 100), dtype=np.float32)
    data[:, barrier_col] = -9999.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32719",
        transform=from_origin(0, 100, 1, 1),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def write_points(path, points, crs):
    """points: lista de (id, x, y)."""
    schema = {"geometry": "Point", "properties": {"id": "int"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as dst:
        for pid, x, y in points:
            dst.write(
                {
                    "geometry": {"type": "Point", "coordinates": (x, y)},
                    "properties": {"id": int(pid)},
                }
            )


def build_cells_1_to_4(tmp_path, raster=None, points=None, mask=None):
    """Celdas 1-4 con la Celda 3 parchada a fixtures (None = dejar el default)."""
    srcs = load_code_sources()
    reps = {
        "OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}_Refactored')": (
            f"OUTPUT_DIR = os.path.join({str(tmp_path)!r}, f'session_{{timestamp}}')"
        ),
    }
    if raster is not None:
        reps["COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')"] = (
            f"COST_RASTER_PATH = {str(raster)!r}"
        )
    if points is not None:
        reps["ALL_POINTS_SHAPEFILE = os.path.join(DATA_DIR, 'points.shp')"] = (
            f"ALL_POINTS_SHAPEFILE = {str(points)!r}"
        )
    if mask is None and points is not None:
        # fixtures de prueba sin máscara salvo que se pida otra
        reps["MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')"] = (
            "MASK_SHAPEFILE_PATH = None"
        )
    elif mask is not None:
        reps["MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')"] = (
            f"MASK_SHAPEFILE_PATH = {str(mask)!r}"
        )
    srcs[2] = patch_cell3(srcs[2], reps)
    return srcs[:4]
