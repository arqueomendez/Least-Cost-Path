"""Tests del backlog de calidad: estimación de workers de la Celda 2 (S6).

S7-S9 (validación N>=2, capa de inalcanzables, RESUME) se agregan aqui
cuando se implementen.
"""

import ast
import sys
from types import SimpleNamespace

import nbformat
import psutil

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


def _stub(monkeypatch, available, cpu):
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: cpu)


def test_n_jobs_phase1_limited_by_cpu(monkeypatch):
    fn = _n_jobs_fn()
    _stub(monkeypatch, 8 * 1024**3, 16)
    # 8 GB: el presupuesto por worker (raster 8 MB x 5.5) no limita -> manda la CPU
    assert fn((1000, 1000), "float64", 1) == 16


def test_n_jobs_min_one_on_low_ram(monkeypatch):
    fn = _n_jobs_fn()
    _stub(monkeypatch, 10 * 1024**2, 16)
    # 10 MB disponibles: int(0.8*10MB/44MB) = 0 -> debe quedar 1 worker
    assert fn((1000, 1000), "float64", 1) == 1


def test_n_jobs_windows_cap_61(monkeypatch):
    fn = _n_jobs_fn()
    _stub(monkeypatch, 64 * 1024**3, 100)
    monkeypatch.setattr(sys, "platform", "win32")
    # limite de WaitForMultipleObjects en Windows
    assert fn((1000, 1000), "float64", 1) == 61


def test_n_jobs_phase2_uses_80mb_budget(monkeypatch):
    fn = _n_jobs_fn()
    _stub(monkeypatch, 8 * 1024**3, 16)
    # 80 MB por worker: int(0.8*8GB/80MB)=81 -> limitado a cpu*2 = 32
    assert fn((1000, 1000), "float64", 2) == 32


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
    assert 'mode="a" if os.path.exists' in s


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
