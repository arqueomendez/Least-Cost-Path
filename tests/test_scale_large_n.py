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


def test_s2_densify_streams_chunks():
    # S2: el densificado escribe por lote (generator_unordered) y NO
    # concatena xs/ys/idx globales en el padre (~1.6 GB para 81M pts).
    s = SRC[7]
    assert 'return_as="generator_unordered"' in s
    assert "np.concatenate([r[0] for r in results])" not in s
    assert "all_idx" not in s
    assert "escritos" in s


def test_s3_tqdm_progress_on_both_phases():
    # S3: tqdm.auto en Fase 1 y Fase 2 (progreso por resultado consumido),
    # con joblib verbose=0 en ambas para no duplicar el ruido.
    s = SRC[5]
    assert "from tqdm.auto import tqdm" in s
    assert s.count("tqdm(") >= 2  # una barra por fase (fase1 pbar + fase2)
    assert "verbose=5" not in s
