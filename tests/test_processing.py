import fiona
import pytest
from rasterio.transform import Affine
from shapely.geometry import Polygon, box, mapping

import lcp.processing as proc

TRANSFORM = Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0)


class _StubRaster:
    def __init__(self, crs, shape, transform):
        self.crs = crs
        self.shape = shape
        self.transform = transform


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


def test_create_mask_crs_mismatch_raises(tmp_path):
    path = tmp_path / "poligono_4326.shp"
    schema = {"geometry": "Polygon", "properties": {}}
    with fiona.open(
        path, "w", driver="ESRI Shapefile", schema=schema, crs="EPSG:4326"
    ) as dst:
        dst.write({"geometry": mapping(box(0, 0, 1, 1)), "properties": {}})
    src = _StubRaster("EPSG:32719", (8, 8), Affine(1.0, 0, 0, 0, -1.0, 8))
    with pytest.raises(ValueError, match="Discrepancia de CRS"):
        proc.create_mask_from_vector(str(path), src)


def test_create_mask_repairs_invalid_geometry(tmp_path):
    # bowtie: poligono auto-intersectado (is_valid=False) -> buffer(0) lo repara
    path = tmp_path / "bowtie.shp"
    schema = {"geometry": "Polygon", "properties": {}}
    bowtie = Polygon([(0, 0), (4, 4), (4, 0), (0, 4)])
    assert not bowtie.is_valid
    with fiona.open(
        path, "w", driver="ESRI Shapefile", schema=schema, crs="EPSG:32719"
    ) as dst:
        dst.write({"geometry": mapping(bowtie), "properties": {}})
    src = _StubRaster("EPSG:32719", (8, 8), Affine(1.0, 0, 0, 0, -1.0, 8))
    mask = proc.create_mask_from_vector(str(path), src)
    assert mask.dtype == bool
    assert mask.shape == (8, 8)
    assert mask.any()
