"""Pins estructurales de los specs del notebook (verificador independiente).

S2-S8: la fuente del notebook es la especificacion ejecutable; estos tests
fijan las reglas observables sin ejecutar el pipeline completo.
"""

import ast
import pathlib

import nbformat
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]


def _cell(idx):
    return SRC[idx]


def test_s2_phase1_hard_fail():
    s = _cell(5)
    assert "failed_origins" in s
    assert "raise RuntimeError" in s
    assert s.index("raise RuntimeError") < s.index("reconstruct_routes_from_disk")
    assert "pipeline_ok" in s and "and pipeline_ok" in s
    # sin codigo muerto: una sola impresion de traceback, antes del raise del except
    assert s.count("tb.print_exc()") == 1
    exc_block = s[s.index("except Exception as e:") :]
    assert exc_block.index("tb.print_exc()") < exc_block.index("raise")


def test_s3_inf_pairs_reported():
    assert "np.isfinite" in _cell(5)
    assert "pares_inf" in _cell(5)


def test_s4_dynamic_puntos_name():
    s = _cell(8)
    assert "globals().get('SPACING_M', 25.0)" in s
    assert "puntos_rutas_25m" not in s


def test_s5_s6_notebook_hygiene():
    assert "ID_FIELD_NAME = 'id'" in _cell(2)
    for idx in (5, 6, 7, 8):
        assert "dir()" not in _cell(idx)


def test_s7_crs_none_and_densify_guard():
    s = _cell(7)
    assert "None if _crs_rutas is None" in s
    tree = ast.parse(s)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "densify_chunk"
    )
    ns = {"np": np}
    exec(ast.get_source_segment(s, node), ns)
    fn = ns["densify_chunk"]
    idx = np.array([0], dtype=np.int32)
    # ruta degenerada de 1 vertice -> conserva el unico punto
    xs, ys, got_idx = fn(
        idx, [np.array([[100.0, 200.0]])], np.array([0.0]), np.array([1]), 25.0
    )
    assert list(xs) == [100.0] and list(ys) == [200.0]
    assert list(got_idx) == [0]
    # lote de rutas sin coordenadas -> arrays vacios, sin lanzar (S14b)
    xs, ys, got_idx = fn(idx, [np.zeros((0, 2))], np.array([0.0]), np.array([1]), 25.0)
    assert xs.size == 0 and ys.size == 0 and got_idx.size == 0


def test_s8_export_has_no_drift_markers():
    py = (ROOT / "LCP_MCP_Geometric.py").read_text(encoding="utf-8")
    assert "Proyecto Vannia" not in py
    assert "HEATMAP_BANDWIDTH_M = 1000" in py
    assert "_PUNTOS_NAME" in py
    assert "pipeline_ok" in py


def test_max_workers_phase1_override():
    # S-extra: parametro MAX_WORKERS_PHASE1 en la Celda 3 y aplicado en la
    # Celda 6 (remedio explicito para OOM de Fase 1 en rams compartidas,
    # verificado en dos maquinas con el raster de 171 Mpx).
    assert "MAX_WORKERS_PHASE1" in SRC[2]
    s = SRC[5]
    assert "MAX_WORKERS_PHASE1" in s
    i = s.index("MAX_WORKERS_PHASE1")
    assert "min(n_jobs_p1" in s[i : i + 400]


def test_mask_all_touched_param_wired():
    # S-extra (sesgo de borde): la mascara es ESTRICTA por defecto y el parametro
    # llega hasta create_mask_from_vector; sin esto la mascara se expande un anillo
    # fuera del poligono y ese anillo barato se vuelve corredor perimetral.
    assert "MASK_ALL_TOUCHED" in SRC[2]
    s4 = SRC[3]
    assert "all_touched=MASK_ALL_TOUCHED" in s4


def test_edge_buffer_param_wired():
    # S-extra (marco barato del raster): EDGE_BUFFER_PX excluye el marco exterior
    # del raster cuando el modelo de coste no es fiable en los limites.
    assert "EDGE_BUFFER_PX" in SRC[2]
    s4 = SRC[3]
    assert "EDGE_BUFFER_PX" in s4
    i = s4.index("EDGE_BUFFER_PX")
    assert "main_search_mask" in s4[i : i + 900]
