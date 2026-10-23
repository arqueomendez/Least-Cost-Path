import os

import numpy as np
import pytest
from rasterio.transform import Affine

import lcp.pathfinder as pf
import lcp.postprocessing as postp

SHAPE = (20, 20)
TRANSFORM = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)  # 1 m, north-up


def _px_to_world(pixel):
    return (pixel[1] + 0.5, -(pixel[0] + 0.5))


def test_phase1_phase2_roundtrip(tmp_path):
    memmap_path = str(tmp_path / "raster.dat")
    mm = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=SHAPE)
    mm[:] = 1.0
    mm.flush()
    del mm

    phase1_dir = str(tmp_path / "phase1")
    os.makedirs(phase1_dir)

    origin_id = 1
    origin_pixel = (2, 2)
    dest_pixels = {5: (10, 10), 9: (15, 4)}
    shared = {
        "memmap_path": memmap_path,
        "raster_dtype": np.float64,
        "raster_shape": SHAPE,
    }

    # --- Fase 1: MCP + traceback a disco ---
    oid, costs = pf.compute_and_save_cost_surface(
        (origin_id, origin_pixel, dest_pixels, shared, phase1_dir)
    )
    assert oid == origin_id
    assert set(costs) == {5, 9}
    assert os.path.exists(os.path.join(phase1_dir, f"traceback_{origin_id}.npz"))
    # coste geometrico unitario: (2,2)->(10,10) es una diagonal de 8 pasos
    assert costs[5] == pytest.approx(8 * np.sqrt(2))
    # (2,2)->(15,4): 2 diagonales + 11 ortogonales
    assert costs[9] == pytest.approx(11 + 2 * np.sqrt(2))

    # --- Fase 2: reconstruccion desde disco ---
    origin_coords = _px_to_world(origin_pixel)
    dest_items = [(did, _px_to_world(px)) for did, px in dest_pixels.items()]
    oid2, records = postp.reconstruct_routes_from_disk(
        (origin_id, origin_coords, dest_items, costs, TRANSFORM, phase1_dir)
    )
    assert oid2 == origin_id
    assert len(records) == 2

    rec5 = next(r for r in records if r["dest_id"] == 5)
    assert rec5["origin_id"] == 1
    assert rec5["n_vertices"] == 9  # 8 pasos + origen
    assert rec5["cost_total"] == pytest.approx(8 * np.sqrt(2))
    assert rec5["length_m"] == pytest.approx(8 * np.sqrt(2))
    assert rec5["geometry"].geom_type == "LineString"

    rec9 = next(r for r in records if r["dest_id"] == 9)
    assert rec9["n_vertices"] == 14  # 13 pasos + origen
    assert rec9["cost_total"] == pytest.approx(11 + 2 * np.sqrt(2))
