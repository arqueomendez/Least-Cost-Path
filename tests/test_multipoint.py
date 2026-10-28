"""Tests de multipoint-support (S1 y S3).

S1: explosion de MultiPoint con IDs derivados (orig_id * 1000 + parte).
S3: pins estructurales de las Celdas 5 y 7 (grafican desde el dict cargado,
sin acceder a `.x` sobre la geometria cruda releida del archivo).
"""

import pathlib

import fiona
import nbformat
import pytest
from shapely.geometry import LineString, MultiPoint, Point, mapping

import lcp.data_loader as dl

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]


def _write_gpkg(path, geometry_type, features):
    """Escribe un GPKG con (geometria, id) en el orden dado."""
    schema = {"geometry": geometry_type, "properties": {"id": "int"}}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:32719") as dst:
        for geom, pid in features:
            dst.write({"geometry": mapping(geom), "properties": {"id": pid}})


# ---------------------------------------------------------------------------
# S1: semantica de explosion con IDs derivados
# ---------------------------------------------------------------------------
def test_multipoint_explodes_with_derived_ids(tmp_path):
    path = tmp_path / "mp.gpkg"
    _write_gpkg(
        path,
        "MultiPoint",
        [
            (MultiPoint([(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)]), 7),
            (MultiPoint([(700.0, 800.0), (900.0, 1000.0)]), 8),
        ],
    )

    points, crs = dl.load_points_as_dict(str(path), "id")

    assert crs is not None
    assert len(points) == 5
    assert set(points) == {7001, 7002, 7003, 8001, 8002}
    assert all(isinstance(pid, int) for pid in points)
    assert points[7001] == (100.0, 200.0)
    assert points[7002] == (300.0, 400.0)
    assert points[7003] == (500.0, 600.0)
    assert points[8001] == (700.0, 800.0)
    assert points[8002] == (900.0, 1000.0)


def test_multipoint_single_part_keeps_original_id(tmp_path):
    path = tmp_path / "mp1.gpkg"
    _write_gpkg(path, "MultiPoint", [(MultiPoint([(10.0, 20.0)]), 5)])

    points, _ = dl.load_points_as_dict(str(path), "id")

    assert set(points) == {5}
    assert points[5] == (10.0, 20.0)


def test_derived_id_collision_with_later_feature_raises(tmp_path):
    # id 7 explota en 7001/7002 y luego aparece una feature con id 7001
    path = tmp_path / "colision.gpkg"
    _write_gpkg(
        path,
        "MultiPoint",
        [
            (MultiPoint([(1.0, 1.0), (2.0, 2.0)]), 7),
            (MultiPoint([(3.0, 3.0)]), 7001),
        ],
    )

    with pytest.raises(ValueError) as exc:
        dl.load_points_as_dict(str(path), "id")

    assert "7001" in str(exc.value)


def test_derived_id_collision_with_earlier_feature_raises(tmp_path):
    # el id 7001 ya fue emitido antes de que id 7 explote en 7001/7002
    path = tmp_path / "colision2.gpkg"
    _write_gpkg(
        path,
        "MultiPoint",
        [
            (MultiPoint([(3.0, 3.0)]), 7001),
            (MultiPoint([(1.0, 1.0), (2.0, 2.0)]), 7),
        ],
    )

    with pytest.raises(ValueError) as exc:
        dl.load_points_as_dict(str(path), "id")

    assert "7001" in str(exc.value)


def test_plain_points_unchanged(tmp_path):
    # Regresion S1: un Point simple conserva su id original.
    path = tmp_path / "pts.gpkg"
    _write_gpkg(
        path,
        "Point",
        [
            (Point(1.0, 2.0), 7),
            (Point(3.0, 4.0), 3),
        ],
    )

    points, _ = dl.load_points_as_dict(str(path), "id")

    assert sorted(points) == [3, 7]
    assert points[7] == (1.0, 2.0)
    assert points[3] == (3.0, 4.0)


def test_non_point_geometry_raises(tmp_path):
    path = tmp_path / "lineas.gpkg"
    _write_gpkg(path, "LineString", [(LineString([(0.0, 0.0), (1.0, 1.0)]), 1)])

    with pytest.raises(ValueError) as exc:
        dl.load_points_as_dict(str(path), "id")

    assert "LineString" in str(exc.value)


def test_multipoint_summary_is_printed(tmp_path, capsys):
    path = tmp_path / "mp.gpkg"
    _write_gpkg(
        path,
        "MultiPoint",
        [
            (MultiPoint([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]), 7),
            (MultiPoint([(4.0, 4.0)]), 8),
        ],
    )

    dl.load_points_as_dict(str(path), "id")
    out = capsys.readouterr().out

    assert "ADVERTENCIA: 1 feature(s) MultiPoint explotadas en 3 puntos" in out
    assert "original*1000+parte" in out


# ---------------------------------------------------------------------------
# S3: pins estructurales de las Celdas 5 y 7
# ---------------------------------------------------------------------------
def test_cell5_plots_from_loaded_dict_not_raw_geometry():
    s = SRC[4]
    assert "point.x" not in s
    assert "point.y" not in s
    assert "all_points_for_validation" in s
    assert "rasterio.transform.rowcol" in s


def test_cell7_does_not_plot_points_gdf():
    s = SRC[6]
    assert "points_gdf.plot" not in s
    assert "all_points_for_validation" in s
    assert "ax.scatter" in s
