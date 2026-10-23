import numpy as np
from skimage.graph import MCP_Geometric

import lcp.pathfinder as pf


def _mcp(shape, start, end):
    mcp = MCP_Geometric(np.ones(shape, dtype=np.float64), fully_connected=True)
    _, traceback = mcp.find_costs([start], [end])
    return traceback, np.asarray(mcp.offsets, dtype=np.int32)


def test_default_offsets_match_skimage():
    # Canary: si skimage reordenara offsets, DEFAULT_OFFSETS quedaria invalido
    # para .npz antiguos (el warm-up del notebook avisa, este test falla).
    _, offsets = _mcp((10, 10), (5, 5), (2, 2))
    assert np.array_equal(offsets, pf.DEFAULT_OFFSETS)


def test_reconstruct_diagonal():
    traceback, offsets = _mcp((7, 7), (0, 0), (6, 6))
    path = pf.reconstruct_path_from_traceback(traceback, (0, 0), (6, 6), offsets)
    assert path is not None
    assert path.shape == (7, 2)
    assert list(path[0]) == [0, 0]
    assert list(path[-1]) == [6, 6]
    steps = np.abs(np.diff(path, axis=0))
    assert steps.max() == 1  # cada paso es vecino 8-conectado


def test_reconstruct_corrupt_traceback_returns_none():
    traceback, offsets = _mcp((7, 7), (0, 0), (6, 6))
    traceback = traceback.copy()
    traceback[6, 6] = -1  # codigo invalido en el pixel final
    assert (
        pf.reconstruct_path_from_traceback(traceback, (0, 0), (6, 6), offsets) is None
    )


def test_reconstruct_never_exceeds_grid():
    traceback, offsets = _mcp((7, 7), (0, 0), (6, 6))
    path = pf.reconstruct_path_from_traceback(traceback, (0, 0), (6, 6), offsets)
    assert path is not None
    # cota de seguridad H*W: el buffer jamas alcanza el tamano del raster
    assert path.shape[0] <= 7 * 7
