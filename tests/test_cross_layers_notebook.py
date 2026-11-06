"""S1/S3/S4/S5/S7/S8/S9 del modo cross_layers en el notebook.

Pins estructurales sobre la fuente de las celdas 3, 4 y 6 (sin ejecutarlas) y
corridas end-to-end de las celdas 1..6 con dos capas de puntos propias.
Cells 7..9 no se tocan (las cubre otro spec).
"""

import pathlib

import geopandas as gpd
import nbformat
import notebook_harness as nh
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import lcp.outputs as outputs

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]

OUTPUT_DIR_FRAGMENT = (
    "OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}_Refactored')"
)
MASK_FRAGMENT = "MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')"

# Capas de fixtures: ids repetidos (1, 2) en AMBAS capas a proposito, para
# probar que nunca se fusionan y que cada una aporta sus propias coordenadas.
ORIGINS = [(1, 10.5, 89.5), (2, 20.5, 89.5)]
DESTS = [(1, 60.5, 40.5), (2, 70.5, 30.5)]


def _strip_comments(src):
    """Quita lineas de comentario y comentarios a fin de linea (2+ espacios)."""
    out = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line.split("  #")[0])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 1. Pins estructurales
# ---------------------------------------------------------------------------
def test_s1_new_modules_imported_in_cell1():
    # Los modulos nuevos se importan con el resto de `lcp.*` en la Celda 1.
    code = _strip_comments(SRC[0])
    assert "import lcp.planner as planner" in code
    assert "import lcp.outputs as outputs" in code


def test_s1_cell3_has_cross_layer_parameters():
    # S1: parametros nuevos en la Celda 3 y enum documentado con el modo nuevo.
    code = _strip_comments(SRC[2])
    for token in (
        "ORIGIN_POINTS_PATH = None",
        "DEST_POINTS_PATH = None",
        'ORIGIN_ID_FIELD = "id"',
        'DEST_ID_FIELD = "id"',
        'INVERT_DIRECTION = "auto"',
    ):
        assert token in code, token
    assert "cross_layers" in SRC[2], "el enum de ANALYSIS_MODE debe documentarlo"


def test_s3_s4_cell4_validates_invert_and_same_pixel():
    # S3/S4: politica de inversion validada e ids de la misma capa/pixel.
    code = _strip_comments(SRC[3])
    assert '"all_pairs", "single_origin", "cross_layers"' in code
    assert "planner.INVERT_POLICIES" in code
    assert "INVERT_DIRECTION" in code
    for token in ("definir ORIGIN_POINTS_PATH", "definir DEST_POINTS_PATH"):
        assert token in code, token
    # ambas capas se cargan por separado (nunca un dict fusionado)
    assert 'dl.load_points_as_dict(layer["path"], layer["field"])' in code
    assert "planner.identical_pixel_pairs" in code
    assert "MISMO pixel" in code


def test_s4_cell4_keeps_legacy_single_layer_validation():
    # Modos legacy: misma validacion de una sola capa (CRS/extend/pixeles) y los
    # mismos mensajes historicos.
    code = _strip_comments(SRC[3])
    assert "El CRS de {layer['subject']} no coincide con el del RÁSTER." in code
    assert "La extensión de {layer['subject']} no está completamente " in code
    assert "IDs de muestra: " in code
    assert 'label": "PUNTOS"' in code


def test_s5_s7_cell6_cross_branch_writes_per_root():
    # S5/S7/S8: rama cross_layers con plan, un GPKG por raiz y manifiesto.
    code = _strip_comments(SRC[5])
    assert "planner.build_pair_plan(" in code
    assert '"cross_layers"' in code
    assert "outputs.write_origin_routes(" in code
    assert "outputs.write_manifest(" in code
    assert "outputs.session_paths(" in code
    assert "outputs.route_artifact(" in code
    # inversion: worker distinto segun la direccion resuelta
    assert "postp.reconstruct_inverted_routes_from_disk" in code
    # estados del manifiesto (S7)
    for estado in ("ok", "incompleto", "sin_rutas", "error"):
        assert f'"{estado}"' in code, estado


def test_s5_cell6_keeps_legacy_unified_writer():
    # El camino legacy (all_pairs / single_origin) no cambia: sigue escribiendo
    # el GPKG unificado y su capa aditiva de inalcanzables.
    code = _strip_comments(SRC[5])
    assert "if not _IS_CROSS:" in code
    assert '"red_completa_unificada.gpkg"' in code
    assert 'layer="rutas_unificadas"' in code
    assert 'layer="destinos_inalcanzables"' in code


def test_s9_cell6_fingerprint_includes_both_layers_and_mask():
    # S9: la huella de RESUME identifica la capa de destinos y la mascara.
    code = _strip_comments(SRC[5])
    assert '_fp_data["dest_layer"]' in code
    assert '_fp_data["origin_layer"]' in code
    assert '_fp_data["mask"]' in code
    assert '_fp_data["inverted"]' in code
    assert "phase1_meta.json" in code
    assert "fingerprint" in code


def test_parameters_tag_for_papermill():
    # La Celda 3 sigue etiquetada 'parameters' para los overrides de papermill.
    cells = [c for c in NB.cells if c.cell_type == "code"]
    assert "parameters" in cells[2].metadata.get("tags", [])


# ---------------------------------------------------------------------------
# 2. Corrida end-to-end (fixtures propias)
# ---------------------------------------------------------------------------
def _write_flat_raster(path):
    """Raster 100x100 EPSG:32719, coste 1.0 y sin nodata."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="float32",
        crs="EPSG:32719",
        transform=from_origin(0, 100, 1, 1),
    ) as dst:
        dst.write(np.ones((100, 100), dtype="float32"), 1)


def _fixtures(tmp_path):
    raster = tmp_path / "cost.tif"
    _write_flat_raster(raster)
    origins = tmp_path / "Sitios.shp"
    dests = tmp_path / "Fuentes.shp"
    nh.write_points(origins, ORIGINS, crs="EPSG:32719")
    nh.write_points(dests, DESTS, crs="EPSG:32719")
    return raster, origins, dests


def _patched_sources(
    tmp_path, raster, origins, dests, session_name, invert=None, **overrides
):
    """Celdas del notebook con la Celda 3 parchada a los fixtures de dos capas."""
    repl = {
        OUTPUT_DIR_FRAGMENT: (
            f"OUTPUT_DIR = os.path.join({str(tmp_path)!r}, {session_name!r})"
        ),
        "COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')": (
            f"COST_RASTER_PATH = {str(raster)!r}"
        ),
        'ANALYSIS_MODE = "all_pairs"': 'ANALYSIS_MODE = "cross_layers"',
        "ORIGIN_POINTS_PATH = None": (
            f"ORIGIN_POINTS_PATH = {str(origins)!r}"
            if origins
            else "ORIGIN_POINTS_PATH = None"
        ),
        "DEST_POINTS_PATH = None": (
            f"DEST_POINTS_PATH = {str(dests)!r}" if dests else "DEST_POINTS_PATH = None"
        ),
        MASK_FRAGMENT: "MASK_SHAPEFILE_PATH = None",
        "MAX_WORKERS_PHASE1 = None": "MAX_WORKERS_PHASE1 = 1",
    }
    if invert is not None:
        repl['INVERT_DIRECTION = "auto"'] = f'INVERT_DIRECTION = "{invert}"'
    repl.update(overrides)
    srcs = nh.load_code_sources()
    srcs[2] = nh.patch_cell3(srcs[2], repl)
    return srcs


def _run_cells_1_to_6(tmp_path, raster, origins, dests, invert, session_name):
    srcs = _patched_sources(
        tmp_path, raster, origins, dests, session_name, invert=invert
    )
    # Celdas 1..6 sin la Celda 5 (grafico diagnostico: no aporta al calculo).
    return nh.run_code([*srcs[:4], srcs[5]], timeout=600)


def _routes(session_dir):
    """{nombre_archivo: GeoDataFrame} de los GPKG por raiz."""
    files = sorted((session_dir / "rutas_por_origen").glob("rutas_origen_*.gpkg"))
    out = {}
    for path in files:
        gdf = gpd.read_file(path)
        out[path.name] = gdf
    return out


def _endpoints(geom):
    coords = list(geom.coords)
    return coords[0], coords[-1]


def _manifest(session_dir):
    return outputs.read_manifest(
        {"manifest": str(session_dir / "manifest_rutas_por_origen.csv")}
    )


def test_e2e_cross_layers_direct_writes_one_gpkg_per_origin(tmp_path):
    raster, origins, dests = _fixtures(tmp_path)
    out, err = _run_cells_1_to_6(
        tmp_path, raster, origins, dests, "never", "session_direct"
    )
    assert err is None, err
    assert "MODO: cross_layers (2 x 2)" in out
    assert "invertido: False" in out

    session = tmp_path / "session_direct"
    gpkgs = _routes(session)
    assert sorted(gpkgs) == ["rutas_origen_1.gpkg", "rutas_origen_2.gpkg"]

    cost_by_pair = {}
    for name, gdf in gpkgs.items():
        assert len(gdf) == 2, f"{name}: {len(gdf)} filas"
        root_id = int(name[len("rutas_origen_") : -len(".gpkg")])
        assert set(gdf["origin_id"]) == {root_id}, "la raiz debe ser el sitio"
        assert set(gdf["dest_id"]) == {1, 2}
        for _, row in gdf.iterrows():
            assert np.isfinite(row["length_m"]) and row["length_m"] > 0
            assert np.isfinite(row["cost_total"])
            assert row["origin_layer"] == "Sitios"
            assert row["dest_layer"] == "Fuentes"
            assert row["n_vertices"] >= 2
            start, end = _endpoints(row.geometry)
            # Los extremos prueban que las capas NO se fusionan: cada id aporta
            # sus propias coordenadas (mismo id, puntos distintos).
            ox, oy = next((x, y) for pid, x, y in ORIGINS if pid == row["origin_id"])
            dx, dy = next((x, y) for pid, x, y in DESTS if pid == row["dest_id"])
            assert abs(start[0] - ox) < 1.5 and abs(start[1] - oy) < 1.5
            assert abs(end[0] - dx) < 1.5 and abs(end[1] - dy) < 1.5
            cost_by_pair[(row["origin_id"], row["dest_id"])] = row["cost_total"]

    assert len(cost_by_pair) == 4

    rows = _manifest(session)
    assert len(rows) == 2
    assert {row["root_id"] for row in rows} == {"1", "2"}
    for row in rows:
        assert row["estado"] == "ok"
        assert row["capa"] == "origin"
        assert row["agrupacion"] == "root"
        assert row["esperados"] == "2"
        assert row["escritos"] == "2"
        assert row["inalcanzables"] == "0"
        assert row["gpkg"].endswith(".gpkg")
        # mapa/densificado/heatmap los llenan las Celdas 7..9
        assert row["mapa"] == "" and row["densificado"] == "" and row["heatmap"] == ""

    # El modo nuevo NO produce el GPKG unificado.
    assert not list(session.glob("red_completa_unificada.gpkg"))


def test_e2e_cross_layers_inverted_mirrors_costs(tmp_path):
    raster, origins, dests = _fixtures(tmp_path)
    _out_d, err_d = _run_cells_1_to_6(
        tmp_path, raster, origins, dests, "never", "session_direct"
    )
    assert err_d is None, err_d
    out_i, err_i = _run_cells_1_to_6(
        tmp_path, raster, origins, dests, "always", "session_inverted"
    )
    assert err_i is None, err_i
    assert "MODO: cross_layers (2 x 2)" in out_i
    assert "invertido: True" in out_i

    direct = {}
    for gdf in _routes(tmp_path / "session_direct").values():
        for _, row in gdf.iterrows():
            direct[(int(row["origin_id"]), int(row["dest_id"]))] = row["cost_total"]

    session = tmp_path / "session_inverted"
    gpkgs = _routes(session)
    # una corrida MCP por FUENTE (2 raices) con su propio GPKG
    assert sorted(gpkgs) == ["rutas_origen_1.gpkg", "rutas_origen_2.gpkg"]

    inverted = {}
    for name, gdf in gpkgs.items():
        root_id = int(name[len("rutas_origen_") : -len(".gpkg")])
        assert len(gdf) == 2, f"{name}: {len(gdf)} filas"
        assert set(gdf["dest_id"]) == {root_id}, "la raiz invertida es la fuente"
        assert set(gdf["origin_id"]) == {1, 2}, "los origenes son los sitios"
        for _, row in gdf.iterrows():
            assert np.isfinite(row["length_m"]) and row["length_m"] > 0
            assert row["origin_layer"] == "Sitios"
            assert row["dest_layer"] == "Fuentes"
            inverted[(int(row["origin_id"]), int(row["dest_id"]))] = row["cost_total"]

    assert sorted(inverted) == sorted(direct)
    for pair, cost in inverted.items():
        # Simetria del MCP: el mismo par cuesta lo mismo en ambas direcciones.
        assert cost == pytest.approx(direct[pair], rel=1e-9), pair

    rows = _manifest(session)
    assert len(rows) == 2
    assert {row["capa"] for row in rows} == {"dest"}
    assert {row["estado"] for row in rows} == {"ok"}
    assert not list(session.glob("red_completa_unificada.gpkg"))


def test_cell4_rejects_same_pixel_site_and_source(tmp_path):
    raster = tmp_path / "cost.tif"
    _write_flat_raster(raster)
    origins = tmp_path / "Sitios.shp"
    dests = tmp_path / "Fuentes.shp"
    # Sitio 1 y fuente 1 comparten pixel: ruta de longitud cero, no representable.
    nh.write_points(origins, [(1, 10.5, 89.5), (2, 20.5, 89.5)], crs="EPSG:32719")
    nh.write_points(dests, [(1, 10.5, 89.5), (2, 70.5, 30.5)], crs="EPSG:32719")

    srcs = _patched_sources(
        tmp_path, raster, origins, dests, "session_same_pixel", invert="never"
    )
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None, "la validacion debia detener la ejecucion"
    assert "MISMO pixel" in err
    assert "(sitio 1, fuente 1)" in err


def test_cell4_rejects_unknown_invert_policy(tmp_path):
    raster, origins, dests = _fixtures(tmp_path)
    srcs = _patched_sources(
        tmp_path,
        raster,
        origins,
        dests,
        "session_bad_invert",
        **{'INVERT_DIRECTION = "auto"': 'INVERT_DIRECTION = "sometimes"'},
    )
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None, "una politica invalida debia detener la ejecucion"
    assert "INVERT_DIRECTION invalido" in err
    assert "auto, never, always" in err


def test_e2e_cross_layers_requires_both_layer_paths(tmp_path):
    raster, origins, _dests = _fixtures(tmp_path)
    srcs = _patched_sources(
        tmp_path, raster, origins, None, "session_missing", invert="never"
    )
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None, "cross_layers sin DEST_POINTS_PATH debia fallar"
    assert "requiere definir DEST_POINTS_PATH" in err
