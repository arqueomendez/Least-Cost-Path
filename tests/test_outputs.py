"""Artefactos por raiz MCP (S5), manifiesto por raiz (S7) y escritura atomica (S8)."""

import csv
import os
import pathlib
import warnings

import fiona
import geopandas as gpd
import numpy as np
import rasterio.transform

import lcp.outputs as outputs
import lcp.utils as utils

TRANSFORM = rasterio.transform.from_origin(500000.0, 8000000.0, 10.0, 10.0)
CRS = "EPSG:32719"


def _records(n, root_id=7, first_dest=100):
    """N registros de ruta con la forma exacta de `build_route_record`."""
    records = []
    for i in range(n):
        pixel_path = np.array([[0, 0], [0, 1 + i], [0, 2 + i]], dtype=np.int32)
        rec = utils.build_route_record(
            pixel_path, TRANSFORM, root_id, first_dest + i, 1.5 + i
        )
        assert rec is not None
        records.append(rec)
    return records


def _junk_gpkg(path):
    """GPKG valido de 1 fila, usado como `.tmp` huerfano previo."""
    row = _records(1)[0]
    gdf = gpd.GeoDataFrame([row], geometry="geometry", crs=CRS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gdf.to_file(path, layer="basura", driver="GPKG", mode="w")


def test_session_paths_creates_subdirs_and_manifest(tmp_path):
    output_dir = tmp_path / "session_x"

    paths = outputs.session_paths(str(output_dir))

    for key in (
        "rutas_por_origen",
        "mapas_por_origen",
        "puntos_por_origen",
        "heatmaps_por_origen",
    ):
        assert pathlib.Path(paths[key]).is_dir()
        assert pathlib.Path(paths[key]).parent == output_dir
    assert paths["manifest"] == str(output_dir / outputs.MANIFEST_NAME)
    assert os.path.isabs(paths["manifest"])

    # idempotente: llamar dos veces no falla y devuelve las mismas rutas
    assert outputs.session_paths(str(output_dir)) == paths


def test_artifact_names_follow_s5(tmp_path):
    paths = outputs.session_paths(str(tmp_path))

    assert outputs.route_artifact(paths, 7) == str(
        tmp_path / "rutas_por_origen" / "rutas_origen_7.gpkg"
    )
    assert outputs.figure_artifact(paths, 7) == str(
        tmp_path / "mapas_por_origen" / "mapa_origen_7.png"
    )
    assert outputs.dense_artifact(paths, 7) == str(
        tmp_path / "puntos_por_origen" / "puntos_origen_7.gpkg"
    )
    # res_m/bw_km integrales sin ceros finales; kernel verbatim
    assert outputs.heatmap_artifact(paths, 3, 1.0, 2.5, "quartic") == str(
        tmp_path / "heatmaps_por_origen" / "heatmap_origen_3_1m_2.5km_quartic.tif"
    )


def test_write_origin_routes_writes_single_layer(tmp_path):
    paths = outputs.session_paths(str(tmp_path))
    records = _records(3)

    path, n_written = outputs.write_origin_routes(records, paths, 7, CRS)

    assert path == outputs.route_artifact(paths, 7)
    assert n_written == len(records) == 3
    assert pathlib.Path(path).is_file()
    assert not pathlib.Path(path + ".tmp").exists()
    assert fiona.listlayers(path) == ["rutas_origen_7"]

    gdf = gpd.read_file(path, layer="rutas_origen_7")
    assert len(gdf) == 3
    assert list(gdf.columns) == [
        "origin_id",
        "dest_id",
        "cost_total",
        "length_m",
        "n_vertices",
        "geometry",
    ]
    assert sorted(gdf["dest_id"].tolist()) == [100, 101, 102]
    assert set(gdf["origin_id"].tolist()) == {7}
    assert gdf.crs.to_epsg() == 32719
    assert gdf["length_m"].min() > 0.0


def test_write_origin_routes_ignores_stale_tmp_and_is_idempotent(tmp_path):
    paths = outputs.session_paths(str(tmp_path))
    final = pathlib.Path(outputs.route_artifact(paths, 7))
    _junk_gpkg(str(final) + ".tmp")
    assert pathlib.Path(str(final) + ".tmp").is_file()

    records = _records(3)
    path, n_written = outputs.write_origin_routes(records, paths, 7, CRS)

    assert n_written == 3
    assert not pathlib.Path(path + ".tmp").exists()
    gdf = gpd.read_file(path, layer="rutas_origen_7")
    assert len(gdf) == 3
    assert sorted(gdf["dest_id"].tolist()) == [100, 101, 102]

    # reejecutar la misma raiz no duplica filas
    _, n_again = outputs.write_origin_routes(records, paths, 7, CRS)
    assert n_again == 3
    assert len(gpd.read_file(path, layer="rutas_origen_7")) == 3


def test_write_origin_routes_empty_records_creates_nothing(tmp_path):
    paths = outputs.session_paths(str(tmp_path))

    path, n_written = outputs.write_origin_routes([], paths, 7, CRS)

    assert n_written == 0
    assert path == outputs.route_artifact(paths, 7)
    assert not pathlib.Path(path).exists()
    assert not pathlib.Path(path + ".tmp").exists()


def test_manifest_roundtrip_subset_rewrite_and_no_tmp(tmp_path):
    paths = outputs.session_paths(str(tmp_path))
    manifest = pathlib.Path(paths["manifest"])

    rows = [
        {"root_id": 7, "capa": "sitios", "agrupacion": "origin", "esperados": 6},
        {
            "root_id": 8,
            "capa": "fuentes",
            "agrupacion": "dest",
            "esperados": 6,
            "inalcanzables": 1,
            "escritos": 5,
            "estado": "incompleto",
            "gpkg": "rutas_origen_8.gpkg",
            "mapa": "mapa_origen_8.png",
            "densificado": "puntos_origen_8.gpkg",
            "heatmap": "heatmap_origen_8_1m_2.5km_quartic.tif",
        },
    ]

    outputs.write_manifest(paths, rows)

    assert not pathlib.Path(str(manifest) + ".tmp").exists()
    with open(manifest, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == outputs.MANIFEST_COLUMNS

    loaded = outputs.read_manifest(paths)
    assert len(loaded) == 2
    assert loaded[0]["root_id"] == "7"
    assert loaded[0]["estado"] == ""  # clave ausente -> celda vacia
    assert loaded[1]["estado"] == "incompleto"
    assert loaded[1]["heatmap"] == "heatmap_origen_8_1m_2.5km_quartic.tif"

    # reescribir con menos filas elimina las anteriores
    outputs.write_manifest(paths, [{"root_id": 9, "estado": "ok"}])
    loaded = outputs.read_manifest(paths)
    assert len(loaded) == 1
    assert loaded[0]["root_id"] == "9"
    assert loaded[0]["capa"] == ""
    assert not pathlib.Path(str(manifest) + ".tmp").exists()


def test_manifest_absent_and_empty(tmp_path):
    paths = outputs.session_paths(str(tmp_path))

    assert outputs.read_manifest(paths) == []

    # sin filas igual se escribe la cabecera
    outputs.write_manifest(paths, [])
    assert outputs.read_manifest(paths) == []
    with open(paths["manifest"], newline="", encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == list(outputs.MANIFEST_COLUMNS)
