from rasterio.transform import Affine

import lcp.processing as proc

TRANSFORM = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


def test_world_to_pixel_known_points():
    # (115,185): col=(115-100)/10=1.5 -> 1 ; row=(185-200)/-10=1.5 -> 1
    assert proc.world_to_pixel(TRANSFORM, 115.0, 185.0) == (1, 1)
    # esquina superior-izquierda del raster
    assert proc.world_to_pixel(TRANSFORM, 100.0, 200.0) == (0, 0)


def test_world_to_pixel_returns_ints():
    row, col = proc.world_to_pixel(TRANSFORM, 105.0, 195.0)
    assert isinstance(row, int)
    assert isinstance(col, int)
    assert (row, col) == (0, 0)
