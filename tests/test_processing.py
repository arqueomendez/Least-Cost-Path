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


def _write_polygon(path, geom):
    schema = {"geometry": "Polygon", "properties": {}}
    with fiona.open(
        path, "w", driver="ESRI Shapefile", schema=schema, crs="EPSG:32719"
    ) as dst:
        dst.write({"geometry": mapping(geom), "properties": {}})


STRICT_SRC = _StubRaster("EPSG:32719", (8, 8), Affine(1.0, 0, 0, 0, -1.0, 8))


def test_create_mask_is_strict_by_default(tmp_path):
    # Defecto corregido: con all_touched=True la mascara se EXPANDE a todo pixel
    # que el poligono roza, aunque su centro quede fuera. Ese anillo exterior
    # conserva el coste del modelo y, si es barato, forma un camino continuo por
    # el borde del poligono (sintoma reportado: rutas pegadas al borde).
    # raster 8x8: el pixel (r,c) cubre x in [c, c+1], y in [8-r-1, 8-r]
    path = tmp_path / "area.shp"
    _write_polygon(path, box(0.2, 0.2, 4.2, 5.8))

    strict = proc.create_mask_from_vector(str(path), STRICT_SRC)
    touched = proc.create_mask_from_vector(str(path), STRICT_SRC, all_touched=True)

    # centros dentro: x in {0.5..3.5} (4 columnas) x y in {0.5..5.5} (6 filas)
    assert int(strict.sum()) == 24
    # la columna 4 solo esta TOCADA (centro 4.5 > 4.2): +6 pixeles
    assert int(touched.sum()) == 30
    # la mascara estricta es subconjunto de la tocada
    assert int((strict & ~touched).sum()) == 0
    # el pixel de borde NO entra en la mascara estricta
    assert not strict[6, 4]


def test_create_mask_all_touched_false_explicit(tmp_path):
    path = tmp_path / "area.shp"
    _write_polygon(path, box(0.2, 0.2, 4.2, 5.8))
    mask = proc.create_mask_from_vector(str(path), STRICT_SRC, all_touched=False)
    assert int(mask.sum()) == 24
    assert mask[6, 3]  # centro dentro
    assert not mask[6, 4]  # solo rozado


def test_edge_frame_mask_zero_is_empty():
    # buffer 0 => no excluye nada (comportamiento por defecto intacto)
    frame = proc.edge_frame_mask((10, 12), 0)
    assert frame.shape == (10, 12)
    assert not frame.any()


def test_edge_frame_mask_excludes_rows_and_cols():
    frame = proc.edge_frame_mask((10, 12), 2)
    assert frame[:2, :].all() and frame[-2:, :].all()
    assert frame[:, :2].all() and frame[:, -2:].all()
    # interior intacto: 6 filas x 8 columnas
    assert int((~frame).sum()) == 6 * 8


def test_edge_frame_mask_covers_small_raster():
    # buffer mayor que el raster: se excluye todo sin desbordar
    frame = proc.edge_frame_mask((4, 5), 99)
    assert frame.all()
