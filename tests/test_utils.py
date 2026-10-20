import numpy as np
import pytest
from rasterio.transform import Affine

import lcp.utils as utils

TRANSFORM = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


def test_build_route_record_valid():
    path = np.array([[0, 0], [0, 1]], dtype=np.int32)
    rec = utils.build_route_record(path, TRANSFORM, 1, 2, 5.5)
    assert rec is not None
    assert rec["origin_id"] == 1
    assert rec["dest_id"] == 2
    assert rec["cost_total"] == pytest.approx(5.5)
    # centros de pixel: (100,200) + (0.5,0.5)*10 -> (105,195) y (115,195)
    assert rec["length_m"] == pytest.approx(10.0)
    assert rec["n_vertices"] == 2
    assert rec["geometry"].geom_type == "LineString"
    assert next(iter(rec["geometry"].coords)) == (105.0, 195.0)


def test_build_route_record_none_path():
    assert utils.build_route_record(None, TRANSFORM, 1, 2, 1.0) is None


def test_build_route_record_too_short():
    path = np.array([[3, 4]], dtype=np.int32)
    assert utils.build_route_record(path, TRANSFORM, 1, 2, 1.0) is None
