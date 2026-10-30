"""Genera los mapas (PNG) de los cuatro escenarios sinteticos de corredor de
borde que ya cubren los tests.

Escenarios (identicos a ``tests/test_border_corridor.py``):

1. ``edge_frame_free``   -- raster 41x41 con un marco de 2 px 100x mas barato
   que el interior, sin exclusion.
2. ``edge_frame_masked`` -- el mismo raster con ``edge_frame_mask(shape, 2)``
   puesto a ``np.inf``.
3. ``mask_touched``      -- raster 21x21 y mascara vectorial con
   ``all_touched=True``: el anillo barato exterior es navegable.
4. ``mask_strict``       -- la misma geometria y los mismos extremos con la
   mascara ESTRICTA por defecto.

Uso (desde la raiz del repositorio)::

    uv run python scripts/synthetic_border_maps.py

Escribe solo imagenes, en ``output/synthetic_border_maps/`` (``output/`` esta
en ``.gitignore``). El script es determinista: sin RNG y sin timestamps.
"""

# ruff: noqa: E402  (los imports de lcp/ y tests/ van tras el bootstrap de sys.path)

import contextlib
import io
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entrada in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tests")):
    if _entrada not in sys.path:
        sys.path.insert(0, _entrada)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from shapely.geometry import box
from synthetic_surfaces import (
    CRS,
    StubRaster,
    build_transform,
    cheap_edge_frame,
    cheap_outside_polygon,
    path_total_cost,
    run_mcp,
    write_polygon,
)

import lcp.processing as proc

OUTPUT_DIR = os.path.join(_REPO_ROOT, "output", "synthetic_border_maps")

# --- Escenario 1 y 2: marco de borde del raster ---------------------------
EDGE_ROWS = 41
EDGE_COLS = 41
FRAME_PX = 2
INTERIOR_COST = 100.0
FRAME_COST = 1.0
EDGE_START = (3, 20)
EDGE_END = (37, 20)

# --- Escenario 3 y 4: anillo del poligono (all_touched vs estricta) --------
BOX_ROWS = 21
BOX_COLS = 21
STUDY_AREA = box(5.9, 4.9, 16.1, 15.1)
INSIDE_COST = 50.0
OUTSIDE_COST = 1.0
BOX_START = (7, 7)
BOX_END = (14, 14)


def _quiet(func, *args, **kwargs):
    """Llama a ``func`` silenciando sus prints (la consola final es ASCII puro)."""
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            return func(*args, **kwargs)
    except Exception as exc:  # se re-lanza con el detalle impreso por la funcion
        raise RuntimeError(f"{exc}\n{sink.getvalue().strip()}") from exc


def build_scenarios(scratch_dir):
    """Construye los cuatro escenarios con las constantes exactas de los tests."""
    edge_transform = build_transform(EDGE_ROWS, EDGE_COLS)
    edge_costs = cheap_edge_frame(
        EDGE_ROWS, EDGE_COLS, FRAME_PX, INTERIOR_COST, FRAME_COST
    )
    frame = proc.edge_frame_mask(edge_costs.shape, FRAME_PX)

    # El poligono se escribe en un directorio temporal fuera del worktree y se
    # borra al terminar: la mascara se construye con la funcion real de lcp.
    vector_path = os.path.join(scratch_dir, "area_estudio.shp")
    write_polygon(vector_path, STUDY_AREA)
    box_transform = build_transform(BOX_ROWS, BOX_COLS)
    stub = StubRaster(CRS, (BOX_ROWS, BOX_COLS), box_transform)
    box_costs = cheap_outside_polygon(
        BOX_ROWS, BOX_COLS, box_transform, STUDY_AREA, INSIDE_COST, OUTSIDE_COST
    )
    touched = _quiet(proc.create_mask_from_vector, vector_path, stub, all_touched=True)
    # Llamada por defecto a proposito: es la mascara estricta que usa el notebook.
    strict = _quiet(proc.create_mask_from_vector, vector_path, stub)
    ring = touched & ~strict
    if not ring.any():
        raise SystemExit(
            "ERROR: la mascara all_touched=True no anade ningun anillo sobre la "
            "estricta; el escenario 3 no es comparable."
        )

    return [
        {
            "escenario": "edge_frame_free",
            "titulo": "Marco de borde barato SIN exclusion",
            "costs": edge_costs,
            "surface": edge_costs,
            "transform": edge_transform,
            "start": EDGE_START,
            "end": EDGE_END,
            "border_mask": frame,
            "borde": "marco",
        },
        {
            "escenario": "edge_frame_masked",
            "titulo": "Marco de borde excluido (edge_frame_mask)",
            "costs": edge_costs,
            "surface": np.where(frame, np.inf, edge_costs),
            "transform": edge_transform,
            "start": EDGE_START,
            "end": EDGE_END,
            "border_mask": frame,
            "borde": "marco",
        },
        {
            "escenario": "mask_touched",
            "titulo": "Mascara all_touched=True (anillo barato navegable)",
            "costs": box_costs,
            "surface": np.where(touched, box_costs, np.inf),
            "transform": box_transform,
            "start": BOX_START,
            "end": BOX_END,
            "border_mask": ring,
            "borde": "anillo",
        },
        {
            "escenario": "mask_strict",
            "titulo": "Mascara estricta por defecto (sin anillo)",
            "costs": box_costs,
            "surface": np.where(strict, box_costs, np.inf),
            "transform": box_transform,
            "start": BOX_START,
            "end": BOX_END,
            "border_mask": ring,
            "borde": "anillo",
        },
    ]


def evaluate(scenario):
    """Ejecuta MCP sobre el escenario y mide la ruta resultante."""
    nombre = scenario["escenario"]
    surface = scenario["surface"]
    try:
        path, _ = run_mcp(surface, scenario["start"], scenario["end"])
    except AssertionError as exc:
        raise SystemExit(
            f"ERROR: el escenario '{nombre}' no produce ninguna ruta navegable "
            f"entre {scenario['start']} y {scenario['end']} ({exc}). Abortando."
        ) from exc
    if path is None or path.size < 2:
        raise SystemExit(
            f"ERROR: el escenario '{nombre}' no produce ninguna ruta navegable "
            f"entre {scenario['start']} y {scenario['end']}. Abortando."
        )
    border = scenario["border_mask"]
    return {
        "ruta": path,
        "coste_total": float(path_total_cost(surface, path)),
        "n_vertices": int(path.shape[0]),
        "vertices_borde": int(border[path[:, 0], path[:, 1]].sum()),
        "navegable_px": int(np.isfinite(surface).sum()),
    }


def _draw_panel(ax, scenario, result, fontsize_titulo, legend_y=-0.13):
    """Dibuja la superficie de coste, la ruta y los extremos en ``ax``."""
    surface = scenario["surface"]
    finito = np.isfinite(surface)
    valores = surface[finito]
    vmin = float(valores.min())
    vmax = float(valores.max())
    if vmax <= vmin:
        # Rango degenerado (p. ej. mask_strict: un unico coste navegable).
        vmax = vmin * 10.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    imagen = ax.imshow(
        np.ma.masked_where(~finito, surface),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )

    ruta = result["ruta"]
    ax.plot(
        ruta[:, 1],
        ruta[:, 0],
        color="red",
        linewidth=2.6,
        zorder=3,
        label="ruta de minimo coste",
    )
    ax.plot(
        ruta[:, 1],
        ruta[:, 0],
        linestyle="none",
        marker="o",
        color="red",
        markersize=3.0,
        zorder=4,
        label=f"vertices ({result['n_vertices']})",
    )
    ax.plot(
        scenario["start"][1],
        scenario["start"][0],
        linestyle="none",
        marker="o",
        color="green",
        markersize=10,
        markeredgecolor="black",
        zorder=5,
        label="origen",
    )
    ax.plot(
        scenario["end"][1],
        scenario["end"][0],
        linestyle="none",
        marker="o",
        color="blue",
        markersize=10,
        markeredgecolor="black",
        zorder=5,
        label="destino",
    )

    filas, cols = surface.shape
    ax.set_xticks(np.arange(-0.5, cols, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, filas, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.65)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", labelsize=fontsize_titulo - 2)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(filas - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("columna (pixel)", fontsize=fontsize_titulo - 1)
    ax.set_ylabel("fila (pixel)", fontsize=fontsize_titulo - 1)
    ax.set_title(
        f"{scenario['titulo']}\n"
        f"coste total = {result['coste_total']:.2f} | "
        f"vertices = {result['n_vertices']} | "
        f"vertices en el {scenario['borde']} = {result['vertices_borde']}",
        fontsize=fontsize_titulo,
    )

    manejadores, etiquetas = ax.get_legend_handles_labels()
    manejadores.append(
        Patch(facecolor="lightgrey", edgecolor="grey", label="no navegable (inf)")
    )
    etiquetas.append("no navegable (inf)")
    ax.legend(
        manejadores,
        etiquetas,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=2,
        fontsize=fontsize_titulo - 2,
        framealpha=0.9,
    )
    return imagen


def draw_single_figure(scenario, result, png_path):
    """Figura individual con colorbar etiquetada ``coste``."""
    fig, ax = plt.subplots(figsize=(7.6, 8.2))
    imagen = _draw_panel(ax, scenario, result, fontsize_titulo=11)
    barra = fig.colorbar(imagen, ax=ax, label="coste", shrink=0.82, pad=0.02)
    barra.ax.tick_params(labelsize=8)
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def draw_comparison(scenarios, resultados, png_path):
    """Panel 2x2 con los mismos cuatro graficos."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 16.5))
    for ax, scenario, result in zip(axes.ravel(), scenarios, resultados, strict=True):
        # La leyenda va bajo cada panel; hspace deja sitio a que no pise el
        # titulo del panel inferior (aspect='equal' encoge el area de dibujo).
        _draw_panel(ax, scenario, result, fontsize_titulo=10, legend_y=-0.19)
    fig.suptitle("Corredores de borde: antes y despues de cada guarda", fontsize=15)
    fig.subplots_adjust(
        left=0.055, right=0.99, top=0.945, bottom=0.055, wspace=0.18, hspace=0.55
    )
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_table(scenarios, resultados):
    """Tabla compacta ASCII: una fila por escenario."""
    cabecera = (
        f"{'escenario':<20}{'vertices':>10}{'coste_total':>14}"
        f"{'vertices_borde':>16}{'navegable_px':>15}"
    )
    print(cabecera)
    print("-" * len(cabecera))
    for scenario, result in zip(scenarios, resultados, strict=True):
        print(
            f"{scenario['escenario']:<20}"
            f"{result['n_vertices']:>10d}"
            f"{result['coste_total']:>14.2f}"
            f"{result['vertices_borde']:>16d}"
            f"{result['navegable_px']:>15d}"
        )
    print()


def main():
    plt.switch_backend("Agg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scratch_dir = tempfile.mkdtemp(prefix="lcp_synthetic_border_")
    try:
        scenarios = build_scenarios(scratch_dir)
        resultados = [evaluate(scenario) for scenario in scenarios]

        print("Escenarios sinteticos de corredor de borde (raster -> ruta MCP)")
        print()
        print_table(scenarios, resultados)

        archivos = []
        for scenario, result in zip(scenarios, resultados, strict=True):
            ruta_png = os.path.join(OUTPUT_DIR, f"{scenario['escenario']}.png")
            draw_single_figure(scenario, result, ruta_png)
            archivos.append(ruta_png)
        ruta_comparacion = os.path.join(OUTPUT_DIR, "comparacion.png")
        draw_comparison(scenarios, resultados, ruta_comparacion)
        archivos.append(ruta_comparacion)

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    print("Archivos generados:")
    for archivo in archivos:
        print(f"  {os.path.abspath(archivo)}")


if __name__ == "__main__":
    main()
