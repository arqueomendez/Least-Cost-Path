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


def test_phase1_npz_stores_cost_scalars(tmp_path):
    """S9: el npz de Fase 1 persiste los escalares por destino para poder
    reanudar sin recomputar el MCP."""
    memmap_path = str(tmp_path / "raster.dat")
    mm = np.memmap(memmap_path, dtype=np.float64, mode="w+", shape=SHAPE)
    mm[:] = 1.0
    mm.flush()
    del mm
    phase1_dir = str(tmp_path / "phase1")
    os.makedirs(phase1_dir)
    shared = {
        "memmap_path": memmap_path,
        "raster_dtype": np.float64,
        "raster_shape": SHAPE,
    }
    _, costs = pf.compute_and_save_cost_surface(
        (1, (2, 2), {5: (10, 10)}, shared, phase1_dir)
    )
    with np.load(os.path.join(phase1_dir, "traceback_1.npz")) as data:
        assert "dest_ids" in data.files
        assert "cost_scalars" in data.files
        stored = {
            int(d): float(c) for d, c in zip(data["dest_ids"], data["cost_scalars"])
        }
    assert set(stored) == {5}
    assert stored[5] == pytest.approx(costs[5])


def test_phase1_npz_compression_policy(tmp_path):
    """S2: compress_npz=False -> np.savez plano (ZIP_STORED, rapido, temporal);
    True/ausente -> ZIP_DEFLATED (npz persistente chico para reanudar).
    np.load lee ambos y la Fase 2 funciona sobre un npz plano."""
    import zipfile

    def run(compress_value):
        p = tmp_path / f"r_{compress_value}.dat"
        mm = np.memmap(p, dtype=np.float64, mode="w+", shape=SHAPE)
        mm[:] = 1.0
        mm.flush()
        del mm
        d = str(tmp_path / f"phase1_{compress_value}")
        os.makedirs(d, exist_ok=True)
        shared = {
            "memmap_path": str(p),
            "raster_dtype": np.float64,
            "raster_shape": SHAPE,
        }
        if compress_value is not None:
            shared["compress_npz"] = compress_value
        pf.compute_and_save_cost_surface((1, (2, 2), {5: (10, 10)}, shared, d))
        npz = os.path.join(d, "traceback_1.npz")
        with zipfile.ZipFile(npz) as z:
            tipos = {i.filename: i.compress_type for i in z.infolist()}
        return npz, tipos, d

    npz_plain, tipos_plain, d_plain = run(False)
    npz_default, tipos_default, _ = run(None)  # clave ausente -> comportamiento actual
    _npz_true, tipos_true, _ = run(True)

    # politica de almacenamiento
    assert all(t == zipfile.ZIP_STORED for t in tipos_plain.values()), tipos_plain
    assert all(t == zipfile.ZIP_DEFLATED for t in tipos_default.values()), tipos_default
    assert all(t == zipfile.ZIP_DEFLATED for t in tipos_true.values()), tipos_true

    # ambos formatos cargan contenido identico
    with np.load(npz_plain) as a, np.load(npz_default) as b:
        assert np.array_equal(a["traceback"], b["traceback"])
        assert np.array_equal(a["offsets"], b["offsets"])
        assert np.array_equal(a["dest_ids"], b["dest_ids"])
        assert np.array_equal(a["cost_scalars"], b["cost_scalars"])

    # Fase 2 sobre npz PLANO (reanudacion sin comprimir)
    with np.load(npz_plain) as data:
        costs = {
            int(d): float(c) for d, c in zip(data["dest_ids"], data["cost_scalars"])
        }
    oid, records = postp.reconstruct_routes_from_disk(
        (
            1,
            _px_to_world((2, 2)),
            [(5, _px_to_world((10, 10)))],
            costs,
            TRANSFORM,
            d_plain,
        )
    )
    assert oid == 1
    assert len(records) == 1
    assert records[0]["dest_id"] == 5


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
