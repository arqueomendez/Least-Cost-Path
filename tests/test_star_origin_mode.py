"""Pins estructurales del modo estrella (specs S1-S5 de star-origin-mode).

Se evalúan sobre la fuente del notebook sin ejecutarlo; el comportamiento
funcional se verifica con corridas completas (T5).
"""

import pathlib

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]


def test_s1_mode_parameter_in_config_cell():
    # Celda 3: panel de parametros con enum y sentinel
    s = SRC[2]
    assert 'ANALYSIS_MODE = "all_pairs"' in s
    assert "ORIGIN_ID = None" in s
    assert '"single_origin"' in s


def test_s3_validation_in_cell4():
    # Celda 4: modo desconocido, single sin ID e ID inexistente -> error temprano
    s = SRC[3]
    assert "ANALYSIS_MODE invalido" in s
    assert "Modos validos" in s
    assert "requiere definir ORIGIN_ID" in s
    assert "no existe" in s
    assert "IDs disponibles" in s


def test_s2_star_branch_in_cell6():
    # Celda 6: un task con destinos = todos excepto O; all_pairs conserva la dedup
    s = SRC[5]
    assert 'ANALYSIS_MODE == "single_origin"' in s
    assert "if did != ORIGIN_ID" in s
    assert "if did > origin_id" in s
    assert "n_expected = len(all_points) - 1" in s


def test_s4_mode_echo_and_counts():
    s = SRC[5]
    assert "MODO:" in s
    assert "rutas esperadas:" in s
    assert "!= esperadas" in s  # aviso al final si reales != esperadas


def test_s6_parameters_tag_for_papermill():
    # La Celda 3 debe estar etiquetada 'parameters' para que papermill -p
    # inyecte los overrides DESPUES de los defaults (si no, se pisan).
    cells = [c for c in NB.cells if c.cell_type == "code"]
    tags = cells[2].metadata.get("tags", [])
    assert "parameters" in tags


def test_s5_highlight_selected_origin():
    # Celdas 5 (idx 4) y 7 (idx 6): resaltado sin interactividad
    for idx in (4, 6):
        s = SRC[idx]
        assert 'ANALYSIS_MODE == "single_origin"' in s
        assert 'marker="*"' in s
        assert '"all_points_for_validation" in globals()' in s
