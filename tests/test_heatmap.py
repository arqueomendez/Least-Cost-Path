"""Regresion KDE/FFT de la Celda 9 (S3): pico, georreferenciacion y masa."""

import ast
import pathlib

import nbformat
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]


def _heatmap_fn():
    src = SRC[8]
    tree = ast.parse(src)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_compute_heatmap"
    )
    ns = {"np": np}
    exec("from scipy.fft import rfft2, irfft2", ns)
    exec(ast.get_source_segment(src, node), ns)
    return ns["_compute_heatmap"]


@pytest.mark.parametrize("kernel", ["quartic", "gaussian", "triangular", "uniform"])
def test_s3_single_point_peak_at_its_own_cell(kernel):
    # bounds simetricos -> grilla 20x20; punto en mundo (0,0) -> celda (10,10).
    # Un crop FFT mal desplazado (bug historico be6161f) desplaza el maximo.
    # Se usa igualdad aproximada: kernels con meseta (uniform) tienen empates
    # flotantes entre celdas vecinas.
    fn = _heatmap_fn()
    hm, _, _ = fn(np.array([[0.0, 0.0]]), -1000, -1000, 1000, 1000, 100, 500, kernel)
    assert hm.shape == (20, 20)
    assert hm[10, 10] == pytest.approx(float(hm.max()), rel=1e-9)


def test_s3_peak_argmax_strict_quartic():
    # quartic estrictamente decreciente -> argmax unico en la celda del punto
    fn = _heatmap_fn()
    hm, _, _ = fn(np.array([[0.0, 0.0]]), -1000, -1000, 1000, 1000, 100, 500, "quartic")
    peak = np.unravel_index(int(np.argmax(hm)), hm.shape)
    assert peak == (10, 10)


@pytest.mark.parametrize("kernel", ["quartic", "gaussian", "triangular", "uniform"])
def test_s3_mass_conservation(kernel):
    # El kernel se normaliza por su AREA (kernel_area = suma * ps^2), asi que la
    # masa discreta del KDE es 1/ps^2 (densidad por unidad de area; consistente
    # con el label 'rutas / km^2' del plot). Regresiona si desaparece la
    # normalizacion o si el recorte "same" pierde masa del kernel.
    fn = _heatmap_fn()
    hm, px, py = fn(np.array([[0.0, 0.0]]), -1000, -1000, 1000, 1000, 100, 500, kernel)
    ps = (px + py) / 2.0
    assert float(hm.sum()) == pytest.approx(1.0 / ps**2, rel=1e-9)


def test_s3_pixel_size_derived_from_bounds_not_res():
    # span 1111 con res 30 -> width=38 -> pixel_size=1111/38=29.236... < res.
    # Fijar pixel_size != res protege la georreferenciacion (bug dbd1595).
    fn = _heatmap_fn()
    hm, px, py = fn(np.zeros((1, 2)), -333, -333, 778, 778, 30, 300, "quartic")
    assert px == pytest.approx(1111 / 38)
    assert py == pytest.approx(1111 / 38)
    assert px < 30
    assert hm.shape == (38, 38)


def test_s3_peak_with_asymmetric_pixels_and_axes():
    # Grilla no cuadrada (pixel x != res): el pico debe caer en la celda que
    # el binning deduce del MISMO pixel_size (ejes fila/columna sin intercambiar).
    fn = _heatmap_fn()
    hm, px, py = fn(np.array([[0.0, 0.0]]), -333, -333, 778, 778, 30, 300, "quartic")
    col = int((0.0 - (-333)) / px)  # binning de columnas
    row = int((778.0 - 0.0) / py)  # binning de filas
    peak = np.unravel_index(int(np.argmax(hm)), hm.shape)
    assert peak == (row, col)


def test_s3_unknown_kernel_raises():
    fn = _heatmap_fn()
    with pytest.raises(ValueError, match="no soportado"):
        fn(np.zeros((1, 2)), -100, -100, 100, 100, 10, 50, "epanechnikov")
