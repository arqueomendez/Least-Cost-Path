"""Mide cuanto pesa de verdad el anillo que anade ``all_touched=True``.

El script es determinista (sin RNG) y no escribe archivos: imprime dos matrices
por consola (ASCII puro, apto para la consola cp1252 de Windows).

Bloque 1 -- tamano del poligono. Un raster fijo de 600 x 600 px a 30.30 m/px y
un cuadrado de estudio centrado, barrido en lados de 20, 60, 180 y 400 px. Todos
los lados van desplazados 0.37 px respecto de la rejilla de pixeles para que el
borde caiga SIEMPRE dentro de un pixel con el centro fuera: sin ese desplazamiento
las dos mascaras coincidirian en muchos tamanos y la comparacion seria vacua.

Bloque 2 -- razon de coste. Con el cuadrado de 180 px, barrido de la razon
coste interior / coste exterior, para ver el umbral a partir del cual la ruta de
la mascara ``all_touched`` se mete realmente en el anillo.

Uso (desde la raiz del repositorio)::

    uv run python scripts/mask_all_touched_eval.py
"""

# ruff: noqa: E402  (los imports de lcp/ y tests/ van tras el bootstrap de sys.path)

import contextlib
import io
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entrada in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tests")):
    if _entrada not in sys.path:
        sys.path.insert(0, _entrada)

import numpy as np
from shapely.geometry import box
from synthetic_surfaces import (
    CRS,
    StubRaster,
    build_transform,
    cheap_outside_polygon,
    path_total_cost,
    run_mcp,
    write_polygon,
)

import lcp.processing as proc

RASTER_PX = 600
PIXEL_M = 30.30
CENTRE_PX = RASTER_PX / 2.0
OFFSET_PX = 0.37
INTERIOR_COST = 50.0
EXTERIOR_COST = 1.0
SIZES_PX = (20, 60, 180, 400)
SWEEP_SIZE_PX = 180
RATIO_SWEEP = (1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 50.0)


def _quiet(func, *args, **kwargs):
    """Llama a ``func`` silenciando sus prints (la salida final es una tabla)."""
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            return func(*args, **kwargs)
    except Exception as exc:  # se re-lanza con el detalle que imprimio la funcion
        raise RuntimeError(f"{exc}\n{sink.getvalue().strip()}") from exc


def study_polygon(side_px):
    """Cuadrado centrado de ``side_px`` de lado, con los bordes desplazados.

    El desplazamiento de 0.37 px deja cada borde dentro de un pixel y con el
    centro del pixel fuera del poligono, de modo que las cuatro caras generan
    anillo tanto con la mascara estricta como con la tocada.
    """
    half = side_px / 2.0
    low = (CENTRE_PX - half - OFFSET_PX) * PIXEL_M
    high = (CENTRE_PX + half + OFFSET_PX) * PIXEL_M
    return box(low, low, high, high)


def boundary_pixels(mask):
    """Pixeles de ``mask`` con al menos un vecino de 4-adyacencia fuera de ella."""
    border = np.zeros_like(mask)
    border[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    border[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    border[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    border[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    return int(border.sum())


def inside_row_pair(strict, row=None):
    """Dos pixeles de la misma fila, uno junto a cada borde de ``strict``."""
    rows_with_mask = np.where(strict.any(axis=1))[0]
    if row is None:
        row = int((rows_with_mask.min() + rows_with_mask.max()) // 2)
    columns = np.where(strict[row])[0]
    return (row, int(columns.min())), (row, int(columns.max()))


def measure(strict, touched, transform, interior_cost, geom):
    """Devuelve una fila de metricas para una mascara y una razon de coste."""
    ring = touched & ~strict
    costs = cheap_outside_polygon(
        strict.shape[0],
        strict.shape[1],
        transform,
        geom,
        interior_cost,
        EXTERIOR_COST,
    )
    start, end = inside_row_pair(strict)
    strict_surface = np.where(strict, costs, np.inf)
    touched_surface = np.where(touched, costs, np.inf)
    strict_path, _ = run_mcp(strict_surface, start, end)
    touched_path, _ = run_mcp(touched_surface, start, end)
    strict_cost = path_total_cost(strict_surface, strict_path)
    touched_cost = path_total_cost(touched_surface, touched_path)
    ring_vertices = int(ring[touched_path[:, 0], touched_path[:, 1]].sum())
    return {
        "navegable_estricta": int(strict.sum()),
        "navegable_tocada": int(touched.sum()),
        "anillo_px": int(ring.sum()),
        "anillo_pct": 100.0 * float(ring.sum()) / float(strict.sum()),
        "perimetro_px": boundary_pixels(strict),
        "coste_estricto": strict_cost,
        "coste_tocado": touched_cost,
        "ratio": touched_cost / strict_cost,
        "en_anillo": "si" if ring_vertices > 0 else "no",
    }


HEADER = (
    f"{'lado':>9}{'navegable':>11}{'navegable':>11}{'anillo':>9}{'anillo':>9}"
    f"{'perimetro':>11}{'coste':>11}{'coste':>10}{'ratio':>8}{'anillo':>8}"
)
SUBHEADER = (
    f"{'px':>9}{'estricta':>11}{'tocada':>11}{'px':>9}{'%area':>9}"
    f"{'px':>11}{'estricto':>11}{'tocado':>10}{'':>8}{'en ruta':>8}"
)


def print_row(label, row):
    print(
        f"{label:>9}"
        f"{row['navegable_estricta']:>11d}"
        f"{row['navegable_tocada']:>11d}"
        f"{row['anillo_px']:>9d}"
        f"{row['anillo_pct']:>9.2f}"
        f"{row['perimetro_px']:>11d}"
        f"{row['coste_estricto']:>11.1f}"
        f"{row['coste_tocado']:>10.1f}"
        f"{row['ratio']:>8.3f}"
        f"{row['en_anillo']:>8}"
    )


def main():
    transform = build_transform(RASTER_PX, RASTER_PX, PIXEL_M)
    stub = StubRaster(CRS, (RASTER_PX, RASTER_PX), transform)

    scratch_dir = tempfile.mkdtemp(prefix="lcp_mask_all_touched_")
    try:
        print(
            f"Raster {RASTER_PX}x{RASTER_PX} px a {PIXEL_M} m/px "
            f"({RASTER_PX * PIXEL_M:.0f} m de lado), cuadrado centrado con "
            f"bordes desplazados {OFFSET_PX} px."
        )
        print(
            "coste exterior = "
            f"{EXTERIOR_COST}, coste interior = {INTERIOR_COST}; anillo% = "
            "anillo_px / navegable_estricta; extremos = dos pixeles de la fila "
            "central junto a los bordes de la mascara estricta."
        )
        print()
        print("BLOQUE 1 - tamano del poligono (anillo constante, area variable)")
        print(HEADER)
        print(SUBHEADER)
        size_rows = []
        for side_px in SIZES_PX:
            geom = study_polygon(side_px)
            vector_path = os.path.join(scratch_dir, f"cuadrado_{side_px}.shp")
            write_polygon(vector_path, geom)
            strict = _quiet(proc.create_mask_from_vector, vector_path, stub)
            touched = _quiet(
                proc.create_mask_from_vector, vector_path, stub, all_touched=True
            )
            row = measure(strict, touched, transform, INTERIOR_COST, geom)
            size_rows.append((side_px, row))
            print_row(f"{side_px}", row)

        print()
        print(
            "BLOQUE 2 - razon de coste interior/exterior con el cuadrado de "
            f"{SWEEP_SIZE_PX} px"
        )
        print(HEADER)
        print(SUBHEADER)
        geom = study_polygon(SWEEP_SIZE_PX)
        vector_path = os.path.join(scratch_dir, "cuadrado_barrido.shp")
        write_polygon(vector_path, geom)
        strict = _quiet(proc.create_mask_from_vector, vector_path, stub)
        touched = _quiet(
            proc.create_mask_from_vector, vector_path, stub, all_touched=True
        )
        sweep_rows = []
        for interior_cost in RATIO_SWEEP:
            row = measure(strict, touched, transform, interior_cost, geom)
            sweep_rows.append((interior_cost, row))
            print_row(f"{interior_cost:g}", row)

        first = next((c for c, r in sweep_rows if r["en_anillo"] == "si"), None)
        before = max((c for c, r in sweep_rows if r["en_anillo"] == "no"), default=None)
        smallest = size_rows[0][1]
        largest = size_rows[-1][1]
        print()
        print("CONCLUSIONES (solo lo que sostienen los numeros de arriba)")
        print(
            f"1. El anillo es siempre de 1 px, asi que su peso relativo cae con "
            f"el tamano: anillo% {smallest['anillo_pct']:.2f} a "
            f"{size_rows[0][0]} px frente a {largest['anillo_pct']:.2f} a "
            f"{size_rows[-1][0]} px (anillo_px {smallest['anillo_px']} frente a "
            f"{largest['anillo_px']}, perimetro px {smallest['perimetro_px']} "
            f"frente a {largest['perimetro_px']})."
        )
        print(
            f"2. El ahorro de coste no desaparece al crecer el poligono: la razon "
            f"coste tocado/estricto va de {smallest['ratio']:.3f} a "
            f"{largest['ratio']:.3f}, es decir la ruta por el anillo barato sigue "
            f"siendo mucho mas barata incluso con {size_rows[-1][0]} px de lado "
            f"({largest['coste_tocado']:.1f} frente a "
            f"{largest['coste_estricto']:.1f})."
        )
        print(
            f"3. El anillo solo es atractivo por encima de una razon de coste "
            f"interior/exterior: con {SWEEP_SIZE_PX} px la ruta tocada entra en "
            f"el anillo a partir de {first:g} y no entra con {before:g} ni por "
            f"debajo, y por eso las dos mascaras dan la misma ruta en el regimen "
            f"de coste casi uniforme."
        )
    finally:
        for nombre in os.listdir(scratch_dir):
            ruta = os.path.join(scratch_dir, nombre)
            try:
                os.remove(ruta)
            except OSError:
                pass
        try:
            os.rmdir(scratch_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
