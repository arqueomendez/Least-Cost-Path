"""Pins estructurales del modo od_pair en LCP_MCP_Geometric.ipynb (S1, S4, S6, S7).

Lee la fuente del notebook sin ejecutarlo: mismo enfoque que
test_notebook_hardening.py y test_star_origin_mode.py.
"""

import json
import pathlib

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "LCP_MCP_Geometric.ipynb"

# Celdas por indice absoluto (20 celdas tras insertar la celda markdown de
# od_pair): 7 = Celda 3 (parametros), 9 = Celda 4 (validacion), 13 = Celda 6
# (ejecucion dos fases). Los e2e usan indices de CODIGO (load_code_sources).
CELL_PARAMS = 7
CELL_VALIDATION = 9
CELL_EXEC = 13


def _source(index):
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    return "".join(nb["cells"][index]["source"])


# ---------------------------------------------------------------------------
# Celda 3: parametros de seleccion (S2)
# ---------------------------------------------------------------------------


def test_cell3_declares_selection_parameters():
    src = _source(CELL_PARAMS)
    for name in (
        "ORIGIN_SELECT_FIELD",
        "ORIGIN_SELECT_VALUE",
        "DEST_SELECT_FIELD",
        "DEST_SELECT_VALUE",
    ):
        assert name in src, f"Celda 3 no declara {name}"


def test_cell3_mode_comment_mentions_od_pair():
    src = _source(CELL_PARAMS)
    assert '"od_pair"' in src


# ---------------------------------------------------------------------------
# Celda 4: preview y seleccion estricta (S3, S4)
# ---------------------------------------------------------------------------


def test_cell4_valid_modes_includes_od_pair():
    src = _source(CELL_VALIDATION)
    assert '"od_pair"' in src


def test_cell4_uses_strict_selector():
    src = _source(CELL_VALIDATION)
    assert "dl.load_point_candidates(" in src
    assert "dl.select_point(" in src


def test_cell4_preview_stops_when_selection_missing():
    # Con valores sin definir imprime la tabla de candidatos y detiene: nunca
    # elige automaticamente (S4).
    src = _source(CELL_VALIDATION)
    assert "ORIGIN_SELECT_VALUE is None or DEST_SELECT_VALUE is None" in src


def test_cell4_prints_candidate_table_with_id_fid_and_pixel():
    src = _source(CELL_VALIDATION)
    assert "fid" in src


def test_cell4_echoes_selection_with_fid():
    # S6: eco de la seleccion (id, fid, campo/valor) cuando los valores existen.
    src = _source(CELL_VALIDATION)
    assert "OD_ORIGIN_ID" in src
    assert "OD_DEST_ID" in src


# ---------------------------------------------------------------------------
# Celda 6: plan, un solo task y huella (S1, S6)
# ---------------------------------------------------------------------------


def test_cell6_builds_od_pair_plan():
    src = _source(CELL_EXEC)
    assert 'planner.build_pair_plan(\n            "od_pair"' in src or (
        "build_pair_plan" in src and '"od_pair"' in src
    )


def test_cell6_fingerprint_includes_od_ids():
    # RESUME: cambiar la seleccion debe invalidar los tracebacks (S9).
    src = _source(CELL_EXEC)
    assert "OD_ORIGIN_ID" in src
    assert "OD_DEST_ID" in src


def test_cell6_phase1_single_task_for_od_pair():
    src = _source(CELL_EXEC)
    # Un solo task de Fase 1 con la raiz = origen elegido; con el kernel
    # corridor_astar (KERNEL opcional) el task MCP se omite para od_pair.
    # R2-003/R3-odpair-task-assert (revision): el pin debe exigir el root
    # exacto OD_ORIGIN_ID, no cualquier append de la celda.
    assert "if not _KERNEL_CORRIDOR:" in src
    assert "tasks_p1.append(\n                (OD_ORIGIN_ID," in src


def test_cell6_mode_description_mentions_od_pair():
    src = _source(CELL_EXEC)
    assert "od_pair" in src


# ---------------------------------------------------------------------------
# Corridas end-to-end (S8): celdas 1..4 + ejecucion con fixtures propias
# ---------------------------------------------------------------------------

import fiona  # noqa: E402

import lcp.data_loader as dl  # noqa: E402


def _tmp_dir():
    import tempfile

    return pathlib.Path(tempfile.mkdtemp())


import geopandas as gpd  # noqa: E402
import notebook_harness as nh  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Point, mapping  # noqa: E402

# Capa de una sola capa: el origen tiene id MAYOR que el destino (sin dedup).
# Campo extra "nombre" para probar seleccion por atributo no-id.
OD_POINTS = [
    (1, 10.5, 89.5, "alpha"),
    (2, 20.5, 89.5, "beta"),
    (3, 30.5, 40.5, "gamma"),
]


def _write_od_layer(path):
    schema = {"geometry": "Point", "properties": {"id": "int", "nombre": "str"}}
    with fiona.open(
        path, "w", driver="ESRI Shapefile", schema=schema, crs="EPSG:32719"
    ) as dst:
        for pid, x, y, nombre in OD_POINTS:
            dst.write(
                {
                    "geometry": mapping(Point(x, y)),
                    "properties": {"id": int(pid), "nombre": str(nombre)},
                }
            )


def _od_sources(tmp_path, session_name, **overrides):
    raster = tmp_path / "cost.tif"
    nh.write_synthetic_raster(str(raster))
    points = tmp_path / "points.shp"
    _write_od_layer(str(points))
    repl = {
        "OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}_Refactored')": (
            f"OUTPUT_DIR = os.path.join({str(tmp_path)!r}, {session_name!r})"
        ),
        "COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')": (
            f"COST_RASTER_PATH = {str(raster)!r}"
        ),
        "ALL_POINTS_SHAPEFILE = os.path.join(DATA_DIR, 'points.shp')": (
            f"ALL_POINTS_SHAPEFILE = {str(points)!r}"
        ),
        'ANALYSIS_MODE = "all_pairs"': 'ANALYSIS_MODE = "od_pair"',
        "MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')": (
            "MASK_SHAPEFILE_PATH = None"
        ),
        "MAX_WORKERS_PHASE1 = None": "MAX_WORKERS_PHASE1 = 1",
    }
    repl.update(overrides)
    srcs = nh.load_code_sources()
    srcs[2] = nh.patch_cell3(srcs[2], repl)
    return srcs


def test_e2e_od_pair_writes_single_route_no_dedup(tmp_path):
    # Seleccion por campo NO-id ("nombre"), valor como texto; origen id 3 >
    # destino id 1: el par pedido se calcula tal cual (sin triangulo superior).
    srcs = _od_sources(
        tmp_path,
        "od_e2e",
        **{
            'ORIGIN_SELECT_FIELD = "id"': 'ORIGIN_SELECT_FIELD = "nombre"',
            "ORIGIN_SELECT_VALUE = None": 'ORIGIN_SELECT_VALUE = "gamma"',
            'DEST_SELECT_FIELD = "id"': 'DEST_SELECT_FIELD = "id"',
            "DEST_SELECT_VALUE = None": 'DEST_SELECT_VALUE = "1"',
        },
    )
    _out, err = nh.run_code([*srcs[:4], srcs[5]], timeout=600)
    assert err is None, err
    session = tmp_path / "od_e2e"
    gpkg = session / "red_completa_unificada.gpkg"
    assert gpkg.exists()
    gdf = gpd.read_file(gpkg, layer="rutas_unificadas")
    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert int(row["origin_id"]) == 3
    assert int(row["dest_id"]) == 1
    assert np.isfinite(row["cost_total"])
    assert np.isfinite(row["length_m"])
    assert row.geometry.geom_type == "LineString"


def test_e2e_corridor_astar_writes_single_route_and_skips_mcp(tmp_path):
    # R3-corridor-notebook-branch-uncovered (revision): cobertura e2e del
    # branch de Celda 6 con KERNEL="corridor_astar": la ruta se escribe en
    # el mismo GPKG unificado y el resumen reporta el kernel, no MCP.
    srcs = _od_sources(
        tmp_path,
        "corridor_e2e",
        **{
            'KERNEL = "mcp_geometric"': 'KERNEL = "corridor_astar"',
            "ORIGIN_SELECT_VALUE = None": 'ORIGIN_SELECT_VALUE = "3"',
            "DEST_SELECT_VALUE = None": 'DEST_SELECT_VALUE = "1"',
        },
    )
    _out, err = nh.run_code([*srcs[:4], srcs[5]], timeout=600)
    assert err is None, err
    session = tmp_path / "corridor_e2e"
    gpkg = session / "red_completa_unificada.gpkg"
    assert gpkg.exists()
    gdf = gpd.read_file(gpkg, layer="rutas_unificadas")
    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert int(row["origin_id"]) == 3
    assert int(row["dest_id"]) == 1
    assert np.isfinite(row["cost_total"])
    assert row.geometry.geom_type == "LineString"
    # El branch corridor no debe ejecutar Fase 1 MCP ni reportar tracebacks.
    assert "Tracebacks guardados" not in (_out or "")
    assert "corridor_astar" in (_out or "")


def test_e2e_od_pair_missing_selection_shows_preview_and_stops(tmp_path):
    # S4: valores sin definir -> tabla de candidatos y detencion; nunca elige.
    srcs = _od_sources(tmp_path, "od_preview")
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None
    assert "PREVIEW" in err
    assert "ORIGIN_SELECT_VALUE" in err
    # La tabla de candidatos muestra id, fid y coordenadas.
    assert "fid" in err


def test_e2e_od_pair_unknown_field_lists_available_fields(tmp_path):
    # S3: campo inexistente -> error con los campos disponibles; jamas cae al fid.
    srcs = _od_sources(
        tmp_path,
        "od_badfield",
        **{
            'ORIGIN_SELECT_FIELD = "id"': 'ORIGIN_SELECT_FIELD = "campo_fantasma"',
            "ORIGIN_SELECT_VALUE = None": "ORIGIN_SELECT_VALUE = 1",
            "DEST_SELECT_VALUE = None": "DEST_SELECT_VALUE = 2",
        },
    )
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None
    assert "campo_fantasma" in err
    assert "Campos disponibles" in err
    assert "'nombre'" in err


def test_e2e_od_pair_unknown_value_lists_available_values(tmp_path):
    srcs = _od_sources(
        tmp_path,
        "od_badvalue",
        **{
            "ORIGIN_SELECT_VALUE = None": "ORIGIN_SELECT_VALUE = 999",
            "DEST_SELECT_VALUE = None": "DEST_SELECT_VALUE = 2",
        },
    )
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None
    assert "999" in err
    assert "Valores disponibles" in err


def test_e2e_od_pair_origin_equals_dest_rejected(tmp_path):
    srcs = _od_sources(
        tmp_path,
        "od_same",
        **{
            "ORIGIN_SELECT_VALUE = None": "ORIGIN_SELECT_VALUE = 2",
            'DEST_SELECT_FIELD = "id"': 'DEST_SELECT_FIELD = "nombre"',
            "DEST_SELECT_VALUE = None": 'DEST_SELECT_VALUE = "beta"',
        },
    )
    _out, err = nh.run_code(srcs[:4], timeout=300)
    assert err is not None
    assert "mismo punto" in err


def test_e2e_od_pair_unreachable_destination_reported_not_silent(tmp_path):
    # R3-5: destino al otro lado de una barrera de nodata = inalcanzable.
    # La corrida no escribe rutas: la capa rutas_unificadas queda vacia o sin
    # crear, el par aparece en destinos_inalcanzables con coste infinito y el
    # aviso se imprime (n_expected=1, escritas=0) en vez de perderse callado.
    raster = tmp_path / "cost_barrier.tif"
    nh.write_barrier_raster(str(raster), barrier_col=50)
    points = tmp_path / "points_barrier.shp"
    schema = {"geometry": "Point", "properties": {"id": "int", "nombre": "str"}}
    with fiona.open(
        points, "w", driver="ESRI Shapefile", schema=schema, crs="EPSG:32719"
    ) as dst:
        dst.write(
            {
                "geometry": mapping(Point(10.5, 89.5)),
                "properties": {"id": 1, "nombre": "oeste"},
            }
        )
        dst.write(
            {
                "geometry": mapping(Point(60.5, 40.5)),
                "properties": {"id": 3, "nombre": "este"},
            }
        )
    repl = {
        "OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}_Refactored')": (
            f"OUTPUT_DIR = os.path.join({str(tmp_path)!r}, 'od_unreachable')"
        ),
        "COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')": (
            f"COST_RASTER_PATH = {str(raster)!r}"
        ),
        "ALL_POINTS_SHAPEFILE = os.path.join(DATA_DIR, 'points.shp')": (
            f"ALL_POINTS_SHAPEFILE = {str(points)!r}"
        ),
        'ANALYSIS_MODE = "all_pairs"': 'ANALYSIS_MODE = "od_pair"',
        "MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')": (
            "MASK_SHAPEFILE_PATH = None"
        ),
        "MAX_WORKERS_PHASE1 = None": "MAX_WORKERS_PHASE1 = 1",
        'ORIGIN_SELECT_FIELD = "id"': 'ORIGIN_SELECT_FIELD = "nombre"',
        "ORIGIN_SELECT_VALUE = None": 'ORIGIN_SELECT_VALUE = "oeste"',
        'DEST_SELECT_FIELD = "id"': 'DEST_SELECT_FIELD = "nombre"',
        "DEST_SELECT_VALUE = None": 'DEST_SELECT_VALUE = "este"',
    }
    srcs = nh.load_code_sources()
    srcs[2] = nh.patch_cell3(srcs[2], repl)
    _out, err = nh.run_code([*srcs[:4], srcs[5]], timeout=600)
    assert err is None, err
    gpkg = tmp_path / "od_unreachable" / "red_completa_unificada.gpkg"
    assert gpkg.exists()
    layers = set(fiona.listlayers(str(gpkg)))
    if "rutas_unificadas" in layers:
        gdf = gpd.read_file(gpkg, layer="rutas_unificadas")
        assert len(gdf) == 0
    unreachable = gpd.read_file(gpkg, layer="destinos_inalcanzables")
    assert len(unreachable) == 1
    row = unreachable.iloc[0]
    assert int(row["origin_id"]) == 1
    assert int(row["dest_id"]) == 3
    assert not np.isfinite(row["cost_total"])
    # El aviso de coste infinito se imprime: nada se pierde en silencio.
    assert "infinito" in _out


def test_cell6_dest_pixels_all_covers_single_origin():
    # R3-dest-pixels-scope/R4-dest-pixels-mode-gate: single_origin TAMBIEN
    # referencia dest_pixels_all; la compuerta compartida debe cubrirlo (o
    # single_origin fallaria con NameError en corridas validas al baseline).
    src = _source(CELL_EXEC)
    assert 'ANALYSIS_MODE != "cross_layers"' in src


def test_load_point_candidates_warns_fid_fallback_once():
    # R4-fid-fallback-warning-per-feature: el aviso de compatibilidad se
    # imprime UNA vez por carga, no una por feature.
    import io
    from contextlib import redirect_stdout

    schema = {"geometry": "Point", "properties": {"nombre": "str"}}
    path = _tmp_dir() / "no_id_field.shp"
    with fiona.open(
        str(path), "w", driver="ESRI Shapefile", schema=schema, crs="EPSG:32719"
    ) as dst:
        for x, nombre in ((1.0, "a"), (2.0, "b"), (3.0, "c")):
            dst.write(
                {
                    "geometry": mapping(Point(x, 0.0)),
                    "properties": {"nombre": nombre},
                }
            )
    buf = io.StringIO()
    with redirect_stdout(buf):
        cands = dl.load_point_candidates(str(path), "id")
    assert len(cands) == 3
    assert buf.getvalue().count("ADVERTENCIA") == 1
