"""Pins estructurales del modo estrella (specs S1-S5 de star-origin-mode).

Se evalúan sobre la fuente del notebook sin ejecutarlo; el comportamiento
funcional se verifica con corridas completas (T5).
"""

import pathlib
import re

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


def _strip_comments(src):
    """Quita lineas de comentario y comentarios a fin de linea (2+ espacios)."""
    out = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line.split("  #")[0])
    return "\n".join(out)


def _cell6_branches():
    cell = _strip_comments(SRC[5])
    i = cell.index('if ANALYSIS_MODE == "single_origin":')
    j = cell.index("    else:", i)
    k = cell.index("n_jobs_p1 = _calculate_n_jobs_for_phase", j)
    return cell[i:j], cell[j:k]


def test_s2_star_branch_has_no_upper_id_filter():
    # S2: la rama estrella NO filtra por did > origin_id (los destinos con id
    # menor tambien son objetivos legitimos) y crea UN solo task.
    # Cierra la hueca de test_s2_star_branch_in_cell6: ahi "did > origin_id"
    # se satisface con el texto del comentario de la propia rama.
    star, _ = _cell6_branches()
    assert "did != ORIGIN_ID" in star
    assert "did > origin_id" not in star
    assert star.count("tasks_p1.append") == 1


def test_s2_all_pairs_branch_retains_dedup_filter():
    # S2/S1: la rama all_pairs conserva la dedup del triangulo superior.
    _, allp = _cell6_branches()
    assert "did > origin_id" in allp
    assert "did != ORIGIN_ID" not in allp


def test_s4_expected_counts_both_modes():
    # S4: esperadas = N-1 (estrella) y N(N-1)/2 (all_pairs).
    cell = _strip_comments(SRC[5])
    star, allp = _cell6_branches()
    assert "n_expected = len(all_points) - 1" in star
    assert "len(all_points) * (len(all_points) - 1) // 2" in allp
    assert "MODO:" in cell and "rutas esperadas:" in cell
    assert "Rutas escritas:" in cell and "!= esperadas" in cell


def test_s5_highlight_guarded_by_analysis_mode():
    # S5: resaltado no interactivo en Celdas 5 y 7, protegido por guard.
    for idx in (4, 6):
        assert '"ANALYSIS_MODE" in globals()' in SRC[idx]
        assert 'ANALYSIS_MODE == "single_origin"' in SRC[idx]
        assert 'marker="*"' in SRC[idx]


def test_s6_papermill_dependency_declared():
    # S6: papermill como dev dep y pin en el export de requirements.
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"papermill>=2.7.0"' in pyproject
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^papermill==2\.7\.0", reqs, re.M)
