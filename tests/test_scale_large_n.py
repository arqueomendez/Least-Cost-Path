"""Pins estructurales de escala para N grandes (S1-S3 de scale-large-n)."""

import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]


def test_s1_phase2_streams_and_writes_incrementally():
    # S1: Fase 2 se consume como generador y escribe por lotes a un GPKG
    # temporal (os.replace al final); NADA se acumula en all_route_records.
    s = SRC[5]
    assert 'return_as="generator_unordered"' in s
    assert "all_route_records" not in s
    assert "gpkg_tmp" in s
    assert "os.replace" in s
    assert "n_rutas_escritas" in s
    # tras el exito, la Celda 7 debe leer desde disco (sin gdf viejo en memoria)
    assert "routes_gdf = None" in s
