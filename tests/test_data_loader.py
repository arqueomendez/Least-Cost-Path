from pathlib import Path

import fiona
import pytest
from shapely.geometry import Point, mapping

import lcp.data_loader as dl

SAMPLE_POINTS = Path(__file__).resolve().parents[1] / "data" / "points.shp"


def test_load_points_from_sample_shapefile():
    points, crs = dl.load_points_as_dict(str(SAMPLE_POINTS), "id")
    assert len(points) == 14
    assert set(points) == set(range(1, 15))
    assert crs is not None
    x, y = points[1]
    assert isinstance(x, float) and isinstance(y, float)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        dl.load_points_as_dict(str(SAMPLE_POINTS) + ".nope", "id")


def test_missing_id_field_raises():
    with pytest.raises(KeyError):
        dl.load_points_as_dict(str(SAMPLE_POINTS), "campo_inexistente")


def test_duplicate_ids_raise(tmp_path):
    path = tmp_path / "dup.gpkg"
    schema = {"geometry": "Point", "properties": {"id": "int"}}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32719") as dst:
        dst.write({"geometry": mapping(Point(1.0, 2.0)), "properties": {"id": 7}})
        dst.write({"geometry": mapping(Point(3.0, 4.0)), "properties": {"id": 7}})
    with pytest.raises(ValueError, match="7"):
        dl.load_points_as_dict(str(path), "id")
