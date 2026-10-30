"""Tests del backlog de calidad: estimación de workers de la Celda 2 (S6).

S7-S9 (validación N>=2, capa de inalcanzables, RESUME) se agregan aqui
cuando se implementen.
"""

import ast
import sys

import nbformat

import lcp.resources as res

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]


def _n_jobs_fn():
    tree = ast.parse(SRC[1])
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_calculate_n_jobs_for_phase"
    )
    ns = {}
    exec(ast.get_source_segment(SRC[1], node), ns)
    return ns["_calculate_n_jobs_for_phase"]


def _stub(monkeypatch, available, cpu_fisicos=8, cpu_logicos=None):
    """Inyecta el hardware en el seam nuevo: lcp.resources.detect_hardware."""
    hw = {
        "ram_total": max(available, 1),
        "ram_disponible": available,
        "cpu_fisicos": cpu_fisicos,
        "cpu_logicos": cpu_logicos or cpu_fisicos * 2,
        "disco_libre": 10**12,
    }
    monkeypatch.setattr(res, "detect_hardware", lambda *a, **k: dict(hw))


def test_n_jobs_phase1_limited_by_cpu(monkeypatch):
    fn = _n_jobs_fn()
    # raster chico: la RAM no limita, manda el tope de nucleos FISICOS
    _stub(monkeypatch, 64 * 1024**3, cpu_fisicos=16)
    assert fn((1000, 1000), "float64", 1, verbose=False) == 16


def test_n_jobs_min_one_on_low_ram(monkeypatch):
    fn = _n_jobs_fn()
    # 10 MB disponibles y worker de ~0,1 GB: no entra ninguno -> 1, nunca 0
    _stub(monkeypatch, 10 * 1024**2, cpu_fisicos=16)
    assert fn((1000, 1000), "float64", 1, verbose=False) == 1


def test_n_jobs_windows_cap_61(monkeypatch):
    fn = _n_jobs_fn()
    _stub(monkeypatch, 512 * 1024**3, cpu_fisicos=100)
    monkeypatch.setattr(sys, "platform", "win32")
    # limite de WaitForMultipleObjects en Windows
    assert fn((1000, 1000), "float64", 1, verbose=False) == 61


def test_n_jobs_phase2_uses_the_traceback_model(monkeypatch):
    fn = _n_jobs_fn()
    # Fase 2: 2 B/px + 30 MB por worker (0,03 GB en un raster de 1 Mpx), asi que
    # con 8 GB la RAM admite 205 workers y el limite pasa a ser la CPU
    # (2 x nucleos logicos = 2 x 32 = 64)
    _stub(monkeypatch, 8 * 1024**3, cpu_fisicos=16)
    esperado = 61 if sys.platform == "win32" else 64  # tope de Windows
    assert fn((1000, 1000), "float64", 2, verbose=False) == esperado


def test_n_jobs_phase1_uses_the_measured_model(monkeypatch):
    # El modelo medido es 76 B/px + 30 MB por worker (0,10 GB en un raster de
    # 1 Mpx). Con 550 MB disponibles: 0,8 x 550 MB / 0,10 GB = 4 workers.
    # El estimador viejo (raster x 4.5) habria dicho 12 y desbordado la RAM.
    fn = _n_jobs_fn()
    _stub(monkeypatch, 550 * 1024**2, cpu_fisicos=64)
    assert fn((1000, 1000), "float64", 1, verbose=False) == 4


def test_n_jobs_real_raster_suggests_two_on_40gb(monkeypatch):
    # El caso que fallaba: raster de 171,5 Mpx = 12,17 GB por worker. Con 40 GB
    # disponibles caben 2; el estimador viejo elegia 6 y provocaba
    # ArrayMemoryError en skimage._offset_edge_map.
    fn = _n_jobs_fn()
    _stub(monkeypatch, 40 * 1024**3, cpu_fisicos=8)
    assert fn((14355, 11945), "float64", 1, n_tasks=60, verbose=False) == 2


def test_s7_single_origin_requires_two_points():
    # S7: el modo estrella con N=1 (sin destinos) debe fallar temprano en la
    # Celda 4, no generar un task degenerado en la Fase 1.
    s = SRC[3]
    assert "al menos 2 puntos" in s


def test_s8_unreachable_layer_written():
    # S8: la Celda 6 materializa los destinos con coste infinito en una segunda
    # capa aditiva, sin pisar el GPKG principal.
    s = SRC[5]
    assert 'layer="destinos_inalcanzables"' in s
    assert "pares_inf" in s
    # idempotente: el to_file de LA CAPA DE DESTINOS usa mode="w" (reemplazo)
    # — el fix del REJECT L15; nunca append. (mode="a" solo es legitimo en el
    # GPKG temporal de rutas de S1, fuera de este bloque.)
    i = s.index("inf_gdf.to_file")
    j = s.index("mode=", i)
    bloque = s[i : s.index(")", j)]
    assert 'mode="w"' in bloque
    assert 'mode="a"' not in bloque


def test_s9_resume_wiring():
    # S9: parametro RESUME en la Celda 3, huella + reutilizacion en la Celda 6
    # y escalares persistidos por el worker de Fase 1.
    assert "RESUME = True" in SRC[2]
    s6 = SRC[5]
    assert "fingerprint" in s6
    assert "phase1_meta.json" in s6
    assert "reusable" in s6
    assert "RESUME: reutilizando" in s6
    pf_src = (ROOT / "lcp" / "pathfinder.py").read_text(encoding="utf-8")
    # kwargs de np.savez_compressed (sin comillas: son argumentos de palabra clave)
    assert "dest_ids=np.array(" in pf_src
    assert "cost_scalars=np.array(" in pf_src


def test_s10_markdown_before_every_code_cell():
    # S10: cada celda de codigo tiene su celda markdown explicativa en espanol
    # justo antes (incluye la de modos antes de la Celda 3).
    types = [c.cell_type for c in NB.cells]
    assert types.count("markdown") == 9
    assert types.count("code") == 9
    for i, t in enumerate(types):
        if t == "code":
            assert i > 0 and types[i - 1] == "markdown", (
                f"code en posicion {i} sin markdown previo"
            )
    joined = "\n".join(c.source for c in NB.cells if c.cell_type == "markdown")
    for token in (
        "Celda 1",
        "Celda 2",
        "Celda 4",
        "Celda 6",
        "RESUME",
        "_compute_heatmap",
    ):
        assert token in joined, token


def test_s11_nbstripout_declared():
    # S11: politica de outputs — nbstripout como dependencia dev (el filtro
    # local de git se instala con `nbstripout --install`; config no trackeada).
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"nbstripout' in pyproject
