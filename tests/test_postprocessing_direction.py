"""Tests de direccion de Fase 2 (S6, S9 del modo cross_layers).

Cubren:
  - el esquema de seis claves de ``build_route_record`` y las dos columnas de
    capa opcionales (S6),
  - la reconstruccion directa (``reconstruct_routes_from_disk``) y la invertida
    (``reconstruct_inverted_routes_from_disk``) sobre un raster sintetico,
  - el descarte de registros con coste no finito (S9).

Todos los fixtures se crean dentro de ``tmp_path``: sin datos reales ni red.
"""

import math
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import lcp.pathfinder as pf
import lcp.postprocessing as postp
import lcp.utils as utils

SHAPE = (30, 40)
PIXEL_M = 10.0
TRANSFORM = from_origin(400000.0, 8000000.0, PIXEL_M, PIXEL_M)


def _px_to_world(transform, pixel):
    """Centro del pixel (row, col) en coordenadas del mundo (CRS metrico)."""
    row, col = pixel
    return transform * (col + 0.5, row + 0.5)


def _write_raster(tmp_path, cost_array, name="cost.tif"):
    """Escribe el raster de coste (GeoTIFF, EPSG:32719) y lo devuelve como array."""
    path = tmp_path / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=cost_array.shape[0],
        width=cost_array.shape[1],
        count=1,
        dtype="float64",
        crs="EPSG:32719",
        transform=TRANSFORM,
    ) as dst:
        dst.write(cost_array, 1)
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32719
        array = src.read(1).astype(np.float64)
    return array


def _shared_data(tmp_path, array, name="raster.dat"):
    """memmap del raster + dict con las claves que espera el worker de Fase 1."""
    memmap_path = str(tmp_path / name)
    mm = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=array.shape)
    mm[:] = array
    mm.flush()
    del mm
    return {
        "memmap_path": memmap_path,
        "raster_dtype": np.float64,
        "raster_shape": array.shape,
        "compress_npz": False,
    }


def _phase1_dir(tmp_path, name="phase1"):
    path = str(tmp_path / name)
    os.makedirs(path, exist_ok=True)
    return path


def _uniform_raster():
    return np.ones(SHAPE, dtype=np.float64)


def _barrier_raster():
    """Coste unitario con una columna inf que separa dos regiones validas."""
    cost = np.ones(SHAPE, dtype=np.float64)
    cost[:, 20] = np.inf
    return cost


# ---------------------------------------------------------------------------
# Caso 1: reconstruccion directa (origen -> destinos)
# ---------------------------------------------------------------------------


def test_forward_reconstructs_all_reachable_records(tmp_path):
    array = _write_raster(tmp_path, _uniform_raster())
    shared = _shared_data(tmp_path, array)
    phase1 = _phase1_dir(tmp_path)

    origin_pixel = (5, 5)
    dest_pixels = {2: (20, 30), 3: (25, 10)}

    oid, costs = pf.compute_and_save_cost_surface(
        (1, origin_pixel, dest_pixels, shared, phase1)
    )
    assert oid == 1
    assert set(costs) == {2, 3}

    origin_coords = _px_to_world(TRANSFORM, origin_pixel)
    dest_items = [(did, _px_to_world(TRANSFORM, px)) for did, px in dest_pixels.items()]
    returned_id, records = postp.reconstruct_routes_from_disk(
        (1, origin_coords, dest_items, costs, TRANSFORM, phase1)
    )

    assert returned_id == 1
    assert len(records) == 2
    assert {r["dest_id"] for r in records} == {2, 3}
    for record in records:
        assert record["origin_id"] == 1
        assert math.isfinite(record["cost_total"])
        assert record["cost_total"] > 0.0
        assert record["length_m"] > 0.0
        assert record["n_vertices"] >= 2


# ---------------------------------------------------------------------------
# Caso 2: columnas de capa (septimo elemento opcional)
# ---------------------------------------------------------------------------


def test_layer_columns_added_only_when_layers_given(tmp_path):
    array = _write_raster(tmp_path, _uniform_raster())
    shared = _shared_data(tmp_path, array)
    phase1 = _phase1_dir(tmp_path)

    origin_pixel = (5, 5)
    dest_pixels = {2: (20, 30)}
    _oid, costs = pf.compute_and_save_cost_surface(
        (1, origin_pixel, dest_pixels, shared, phase1)
    )
    origin_coords = _px_to_world(TRANSFORM, origin_pixel)
    dest_items = [(did, _px_to_world(TRANSFORM, px)) for did, px in dest_pixels.items()]

    _rid, con_capas = postp.reconstruct_routes_from_disk(
        (1, origin_coords, dest_items, costs, TRANSFORM, phase1, ("sitios", "fuentes"))
    )
    assert len(con_capas) == 1
    assert con_capas[0]["origin_layer"] == "sitios"
    assert con_capas[0]["dest_layer"] == "fuentes"

    _rid, sin_capas = postp.reconstruct_routes_from_disk(
        (1, origin_coords, dest_items, costs, TRANSFORM, phase1)
    )
    assert len(sin_capas) == 1
    assert "origin_layer" not in sin_capas[0]
    assert "dest_layer" not in sin_capas[0]


# ---------------------------------------------------------------------------
# Caso 3: compatibilidad hacia atras del 6-tuple
# ---------------------------------------------------------------------------


def test_six_tuple_call_keeps_legacy_schema(tmp_path):
    array = _write_raster(tmp_path, _uniform_raster())
    shared = _shared_data(tmp_path, array)
    phase1 = _phase1_dir(tmp_path)

    origin_pixel = (5, 5)
    dest_pixels = {7: (12, 18)}
    _oid, costs = pf.compute_and_save_cost_surface(
        (4, origin_pixel, dest_pixels, shared, phase1)
    )

    args = (
        4,
        _px_to_world(TRANSFORM, origin_pixel),
        [(7, _px_to_world(TRANSFORM, (12, 18)))],
        costs,
        TRANSFORM,
        phase1,
    )
    assert len(args) == 6
    returned_id, records = postp.reconstruct_routes_from_disk(args)

    assert returned_id == 4
    assert len(records) == 1
    record = records[0]
    assert set(record) == {
        "geometry",
        "origin_id",
        "dest_id",
        "cost_total",
        "length_m",
        "n_vertices",
    }
    assert record["origin_id"] == 4
    assert record["dest_id"] == 7
    assert record["cost_total"] == pytest.approx(costs[7])
    # El coste minimo es geometrico en pixeles: length_m = coste * tamano_pixel.
    assert record["length_m"] == pytest.approx(record["cost_total"] * PIXEL_M)
    assert record["n_vertices"] == 14  # 13 pasos de diagonal + el origen


# ---------------------------------------------------------------------------
# Caso 4: equivalencia de la inversion sobre el mismo raster
# ---------------------------------------------------------------------------


def test_inverted_routes_match_forward_endpoints_and_cost(tmp_path):
    array = _write_raster(tmp_path, _uniform_raster())
    shared = _shared_data(tmp_path, array)

    a_pixel = (5, 5)  # sitio (capa de origenes)
    b_pixel = (20, 30)  # fuente (capa de destinos)
    a_coords = _px_to_world(TRANSFORM, a_pixel)
    b_coords = _px_to_world(TRANSFORM, b_pixel)

    # Directo: MCP desde A hacia B.
    phase1_forward = _phase1_dir(tmp_path, "phase1_forward")
    _oid, costs_forward = pf.compute_and_save_cost_surface(
        (1, a_pixel, {2: b_pixel}, shared, phase1_forward)
    )
    _fid, forward_records = postp.reconstruct_routes_from_disk(
        (1, a_coords, [(2, b_coords)], costs_forward, TRANSFORM, phase1_forward)
    )
    assert len(forward_records) == 1
    forward = forward_records[0]
    assert (forward["origin_id"], forward["dest_id"]) == (1, 2)

    # Invertido: MCP desde B (raiz) hacia los sitios objetivo.
    phase1_inverted = _phase1_dir(tmp_path, "phase1_inverted")
    _oid, costs_inverted = pf.compute_and_save_cost_surface(
        (2, b_pixel, {1: a_pixel}, shared, phase1_inverted)
    )
    assert costs_inverted[1] == pytest.approx(costs_forward[2])

    returned_id, inverted_records = postp.reconstruct_inverted_routes_from_disk(
        (2, b_coords, [(1, a_coords)], costs_inverted, TRANSFORM, phase1_inverted)
    )

    assert returned_id == 2
    assert len(inverted_records) == 1
    inverted = inverted_records[0]
    # Semantica sitio -> fuente, aunque el MCP se calculo desde la fuente.
    assert inverted["origin_id"] == 1
    assert inverted["dest_id"] == 2
    assert math.isfinite(inverted["cost_total"])
    assert inverted["cost_total"] == pytest.approx(forward["cost_total"])
    assert inverted["length_m"] > 0.0
    assert inverted["n_vertices"] >= 2

    forward_coords = list(forward["geometry"].coords)
    inverted_coords = list(inverted["geometry"].coords)
    # Mismos extremos (los vertices intermedios pueden diferir por empates).
    assert inverted_coords[0] == forward_coords[0]
    assert inverted_coords[-1] == forward_coords[-1]


# ---------------------------------------------------------------------------
# Caso 5: descarte de costes no finitos (S9)
# ---------------------------------------------------------------------------


def test_non_finite_cost_records_are_discarded(tmp_path):
    array = _write_raster(tmp_path, _barrier_raster())
    shared = _shared_data(tmp_path, array)
    phase1 = _phase1_dir(tmp_path)

    # El origen queda al este de la barrera y el destino 3 al oeste de ella, en
    # la diagonal NW que vuelve al origen: inalcanzable y con el traceback en
    # cero (clase de ruta falsa del hallazgo F3).
    origin_pixel = (21, 23)
    dest_pixels = {2: (21, 24), 3: (0, 2)}

    _oid, costs = pf.compute_and_save_cost_surface(
        (1, origin_pixel, dest_pixels, shared, phase1)
    )
    assert math.isfinite(costs[2])
    assert not math.isfinite(costs[3])  # precondicion del defecto

    origin_coords = _px_to_world(TRANSFORM, origin_pixel)
    dest_items = [(did, _px_to_world(TRANSFORM, px)) for did, px in dest_pixels.items()]
    _rid, records = postp.reconstruct_routes_from_disk(
        (1, origin_coords, dest_items, costs, TRANSFORM, phase1)
    )

    assert {r["dest_id"] for r in records} == {2}
    assert all(math.isfinite(r["cost_total"]) for r in records)

    # Pin directo del descarte: aunque el traceback permita reconstruir la
    # geometria, un coste no finito (inf o nan) declara la ruta invalida.
    reachable_items = [
        (2, _px_to_world(TRANSFORM, (21, 24))),
        (4, _px_to_world(TRANSFORM, (21, 25))),
        (6, _px_to_world(TRANSFORM, (22, 26))),
    ]
    _rid, filtrados = postp.reconstruct_routes_from_disk(
        (
            1,
            origin_coords,
            reachable_items,
            {2: costs[2], 4: float("inf"), 6: float("nan")},
            TRANSFORM,
            phase1,
        )
    )
    assert {r["dest_id"] for r in filtrados} == {2}
    assert all(math.isfinite(r["cost_total"]) for r in filtrados)


# ---------------------------------------------------------------------------
# Caso 6: contrato de build_route_record
# ---------------------------------------------------------------------------


def test_build_route_record_schema_and_short_paths():
    assert (
        utils.build_route_record(
            np.array([[3, 4]], dtype=np.int32), TRANSFORM, 1, 2, 1.0
        )
        is None
    )
    assert utils.build_route_record(None, TRANSFORM, 1, 2, 1.0) is None

    path = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int32)

    legacy = utils.build_route_record(path, TRANSFORM, 1, 2, 5.5)
    assert set(legacy) == {
        "geometry",
        "origin_id",
        "dest_id",
        "cost_total",
        "length_m",
        "n_vertices",
    }
    assert legacy["length_m"] == pytest.approx(legacy["geometry"].length)
    assert legacy["cost_total"] == pytest.approx(5.5)

    with_layers = utils.build_route_record(
        path, TRANSFORM, 1, 2, 5.5, origin_layer="sitios", dest_layer="fuentes"
    )
    assert len(with_layers) == 8
    assert with_layers["origin_layer"] == "sitios"
    assert with_layers["dest_layer"] == "fuentes"
    # El resto del registro no cambia al agregar las capas.
    for key in ("origin_id", "dest_id", "cost_total", "length_m", "n_vertices"):
        assert with_layers[key] == legacy[key]
    assert with_layers["geometry"].equals_exact(legacy["geometry"], 0.0)

    only_origin = utils.build_route_record(
        path, TRANSFORM, 1, 2, 5.5, origin_layer="sitios"
    )
    assert "origin_layer" in only_origin
    assert "dest_layer" not in only_origin
