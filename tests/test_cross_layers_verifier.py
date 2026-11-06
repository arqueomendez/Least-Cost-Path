"""Sondas independientes del modo cross_layers (S2, S3, S5, S6, S7, S8, S9, S11).

Sondas del verificador externo, adoptadas como tests de regresion. A diferencia
de los pins de las Celdas 3..9, estas sondas ejercitan los modulos `lcp.*` por su
API publica y leen la documentacion generada, de modo que un cambio de
comportamiento (no solo del texto de una celda) las rompe.

La sonda del nombre del GeoTIFF por raiz estaba escrita para FALLAR a proposito
(B2: el artefacto se nombraba con metros donde la API espera km); aqui afirma ya
el comportamiento corregido.
"""

import json
import os
import pathlib

import geopandas as gpd
import nbformat
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString

import lcp.outputs as outputs
import lcp.planner as planner
import lcp.postprocessing as postp
import lcp.utils as utils

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "LCP_MCP_Geometric.ipynb", as_version=4)
SRC = [c.source for c in NB.cells if c.cell_type == "code"]

ORIGINS = {1: (10.5, 89.5), 2: (20.5, 89.5), 3: (30.5, 89.5)}
DESTS = {1: (60.5, 40.5), 2: (70.5, 30.5)}

# Transform del raster sintetico: 1 m por pixel, origen al noroeste.
TRANSFORM = from_origin(0.0, 100.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# S2 — planificador puro de pares por modo
# ---------------------------------------------------------------------------
def test_s2_build_pair_plan_per_mode_counts():
    points = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)}

    all_pairs = planner.build_pair_plan("all_pairs", points)
    assert all_pairs["mode"] == "all_pairs"
    assert all_pairs["n_expected"] == 3  # 3 x 2 / 2
    # triangulo superior: cada raiz solo apunta a ids mayores
    assert all_pairs["roots"][1] == {2: points[2], 3: points[3]}
    assert 1 not in all_pairs["roots"][2]

    single = planner.build_pair_plan("single_origin", points, origin_id=2)
    assert single["n_expected"] == 2
    assert set(single["roots"]) == {2}
    assert set(single["roots"][2]) == {1, 3}

    cross = planner.build_pair_plan("cross_layers", ORIGINS, DESTS)
    assert cross["mode"] == "cross_layers"
    assert cross["n_expected"] == len(ORIGINS) * len(DESTS)
    assert set(cross["roots"]) == set(ORIGINS)
    for root_id, targets in cross["roots"].items():
        assert set(targets) == set(DESTS), root_id
        assert targets[1] == DESTS[1]


def test_s2_build_pair_plan_errors_name_valid_modes_and_ids():
    with pytest.raises(ValueError) as bad_mode:
        planner.build_pair_plan("crosses", ORIGINS, DESTS)
    for mode in planner.MODES:
        assert mode in str(bad_mode.value)

    with pytest.raises(ValueError) as missing_origin:
        planner.build_pair_plan("single_origin", ORIGINS)
    assert "origin_id" in str(missing_origin.value).lower()

    with pytest.raises(ValueError) as unavailable_origin:
        planner.build_pair_plan("single_origin", ORIGINS, origin_id=99)
    assert "99" in str(unavailable_origin.value)
    assert "1" in str(unavailable_origin.value)

    with pytest.raises(ValueError) as empty_layer:
        planner.build_pair_plan("cross_layers", ORIGINS, {})
    assert str(empty_layer.value)


# ---------------------------------------------------------------------------
# S3 — inversion de direccion
# ---------------------------------------------------------------------------
def test_s3_inversion_policies_and_mirrored_semantics():
    # auto: 3 origenes > 2 destinos -> invertido (un MCP por fuente)
    auto = planner.build_pair_plan("cross_layers", ORIGINS, DESTS, invert="auto")
    assert auto["inverted"] is True
    assert set(auto["roots"]) == set(DESTS)
    for root_id, targets in auto["roots"].items():
        assert set(targets) == set(ORIGINS), root_id

    # never: MCP por sitio aunque haya mas origenes que destinos
    never = planner.build_pair_plan("cross_layers", ORIGINS, DESTS, invert="never")
    assert never["inverted"] is False
    assert set(never["roots"]) == set(ORIGINS)

    # always: invierte incluso cuando hay mas destinos que origenes
    always = planner.build_pair_plan("cross_layers", DESTS, ORIGINS, invert="always")
    assert always["inverted"] is True
    assert set(always["roots"]) == set(ORIGINS)

    # la semantica M x N no depende de la direccion resuelta
    assert auto["n_expected"] == never["n_expected"] == len(ORIGINS) * len(DESTS)

    # auto con mas destinos que origenes no invierte
    not_inverted = planner.build_pair_plan(
        "cross_layers", DESTS, ORIGINS, invert="auto"
    )
    assert not_inverted["inverted"] is False

    with pytest.raises(ValueError) as bad_policy:
        planner.build_pair_plan("cross_layers", ORIGINS, DESTS, invert="sometimes")
    for policy in planner.INVERT_POLICIES:
        assert policy in str(bad_policy.value)


# ---------------------------------------------------------------------------
# S5 — artefactos por raiz, nunca unificados
# ---------------------------------------------------------------------------
def test_s5_artifact_names_are_per_root(tmp_path):
    paths = outputs.session_paths(tmp_path)
    for subdir in (
        "rutas_por_origen",
        "mapas_por_origen",
        "puntos_por_origen",
        "heatmaps_por_origen",
    ):
        assert os.path.isdir(paths[subdir]), subdir

    assert pathlib.Path(outputs.route_artifact(paths, 4)).name == "rutas_origen_4.gpkg"
    assert pathlib.Path(outputs.figure_artifact(paths, 4)).name == "mapa_origen_4.png"
    assert pathlib.Path(outputs.dense_artifact(paths, 4)).name == "puntos_origen_4.gpkg"

    names = [
        pathlib.Path(outputs.route_artifact(paths, 4)),
        pathlib.Path(outputs.figure_artifact(paths, 4)),
        pathlib.Path(outputs.dense_artifact(paths, 4)),
        pathlib.Path(outputs.heatmap_artifact(paths, 4, 30, 1, "quartic")),
    ]
    # ninguna salida unificada del camino legacy, y cada una en su subdirectorio
    for path in names:
        assert path.name.startswith(
            ("rutas_origen_", "mapa_origen_", "puntos_origen_", "heatmap_origen_")
        )
        assert "unificada" not in path.name
        assert path.parent.parent == tmp_path
    assert len({path.parent.name for path in names}) == 4


# ---------------------------------------------------------------------------
# S6 — columnas de cada GPKG (length_m y cost_total incluidas)
# ---------------------------------------------------------------------------
def test_s6_route_record_columns_with_and_without_layers():
    pixels = np.array([[10, 10], [10, 11], [11, 12]], dtype=np.int32)

    legacy = utils.build_route_record(pixels, TRANSFORM, 1, 2, 2.5)
    assert set(legacy) == {
        "geometry",
        "origin_id",
        "dest_id",
        "cost_total",
        "length_m",
        "n_vertices",
    }
    assert legacy["length_m"] > 0

    tagged = utils.build_route_record(
        pixels, TRANSFORM, 1, 2, 2.5, origin_layer="Sitios", dest_layer="Fuentes"
    )
    assert tagged["origin_layer"] == "Sitios"
    assert tagged["dest_layer"] == "Fuentes"
    assert tagged["cost_total"] == pytest.approx(legacy["cost_total"])
    assert tagged["length_m"] == pytest.approx(legacy["length_m"])


def test_s6_written_gpkg_has_length_and_cost_columns(tmp_path):
    paths = outputs.session_paths(tmp_path)
    records = [
        utils.build_route_record(
            np.array([[1, 1], [1, 2]], dtype=np.int32),
            TRANSFORM,
            1,
            2,
            3.5,
            "Sitios",
            "Fuentes",
        )
    ]
    path, written = outputs.write_origin_routes(records, paths, 1, "EPSG:32719")
    assert written == 1
    gdf = gpd.read_file(path)
    assert list(gdf.columns) == [*outputs.ROUTE_COLUMNS, "geometry"]
    assert gdf["cost_total"].iloc[0] == pytest.approx(3.5)
    assert gdf["length_m"].iloc[0] > 0
    assert gdf["origin_layer"].iloc[0] == "Sitios"
    assert gdf["dest_layer"].iloc[0] == "Fuentes"
    assert isinstance(gdf.geometry.iloc[0], LineString)

    # el esquema legacy (sin capas) queda intacto: sin columnas de capa
    legacy_records = [
        utils.build_route_record(
            np.array([[1, 1], [1, 2]], dtype=np.int32), TRANSFORM, 1, 2, 3.5
        )
    ]
    legacy_path, _n = outputs.write_origin_routes(
        legacy_records, paths, 2, "EPSG:32719"
    )
    legacy_columns = list(gpd.read_file(legacy_path).columns)
    assert "origin_layer" not in legacy_columns
    assert "dest_layer" not in legacy_columns
    assert legacy_columns[:4] == ["origin_id", "dest_id", "cost_total", "length_m"]


# ---------------------------------------------------------------------------
# S7 — manifiesto por raiz
# ---------------------------------------------------------------------------
def test_s7_manifest_roundtrip_declares_every_column(tmp_path):
    paths = outputs.session_paths(tmp_path)
    rows = [
        {
            "root_id": 1,
            "capa": "origin",
            "agrupacion": "root",
            "esperados": 2,
            "inalcanzables": 0,
            "escritos": 2,
            "estado": "ok",
            "gpkg": "rutas_origen_1.gpkg",
            "mapa": None,
            "densificado": None,
            "heatmap": None,
        }
    ]
    outputs.write_manifest(paths, rows)
    loaded = outputs.read_manifest(paths)
    assert len(loaded) == 1
    assert set(loaded[0]) == set(outputs.MANIFEST_COLUMNS)
    assert loaded[0]["estado"] == "ok"
    # los artefactos aun no producidos quedan como celda vacia, no como 'None'
    assert loaded[0]["mapa"] == ""
    # el encabezado declarado es el contrato
    header = pathlib.Path(paths["manifest"]).read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(outputs.MANIFEST_COLUMNS)
    assert loaded[0]["estado"] in {"ok", "incompleto", "sin_rutas", "error"}
    # Las Celdas 7..9 releen, anotan y reescriben estas filas: deben ser escalares
    # JSON-seguros, no objetos ni arrays (json.dumps falla con tipos numpy).
    assert json.loads(json.dumps(loaded[0])) == loaded[0]


# ---------------------------------------------------------------------------
# S8 — escritura atomica por archivo
# ---------------------------------------------------------------------------
def test_s8_atomic_write_drops_stale_tmp_and_never_duplicates(tmp_path):
    paths = outputs.session_paths(tmp_path)
    records = [
        utils.build_route_record(
            np.array([[1, 1], [1, 2]], dtype=np.int32), TRANSFORM, 1, 2, 1.0
        )
    ]

    # un .tmp huerfano de una corrida cortada nunca se reutiliza
    stale = pathlib.Path(outputs.route_artifact(paths, 1) + ".tmp")
    stale.write_bytes(b"basura de una corrida cortada")

    path, first = outputs.write_origin_routes(records, paths, 1, "EPSG:32719")
    assert first == 1
    assert not stale.exists()

    # reescribir la misma raiz reemplaza (mode="w"), no agrega filas
    path, second = outputs.write_origin_routes(records, paths, 1, "EPSG:32719")
    assert second == 1
    assert len(gpd.read_file(path)) == 1
    assert not list(pathlib.Path(path).parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# S9 — Fase 2 descarta los costes no finitos
# ---------------------------------------------------------------------------
def test_s9_phase2_discards_non_finite_costs(tmp_path):
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    # traceback en cero (celdas que el MCP no finalizo) + costes no finitos
    np.savez(
        phase1 / "traceback_1.npz",
        traceback=np.zeros((100, 100), dtype=np.int16),
        dest_ids=np.array([2, 3], dtype=np.int64),
        cost_scalars=np.array([np.inf, np.nan]),
    )
    root_coords = ORIGINS[1]
    dest_items = [(2, DESTS[1]), (3, DESTS[2])]
    costs = {2: float("inf"), 3: float("nan")}

    root_id, records = postp.reconstruct_routes_from_disk(
        (1, root_coords, dest_items, costs, TRANSFORM, str(phase1))
    )
    assert root_id == 1
    assert records == []

    # sin coste registrado tambien se descarta (nan por defecto) y un padre
    # que no existe no tumba el pool: el worker devuelve lista vacia
    _root_id, missing = postp.reconstruct_routes_from_disk(
        (1, root_coords, dest_items, {}, TRANSFORM, str(phase1))
    )
    assert missing == []
    _root_id, absent = postp.reconstruct_routes_from_disk(
        (9, root_coords, dest_items, {}, TRANSFORM, str(tmp_path / "vacio"))
    )
    assert absent == []


# ---------------------------------------------------------------------------
# S11 — documentacion y export regenerado
# ---------------------------------------------------------------------------
def test_s11_docs_document_the_cross_layers_mode():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for text in (readme, agents):
        assert "cross_layers" in text
        assert "INVERT_DIRECTION" in text
        assert "manifest_rutas_por_origen.csv" in text
    assert "heatmap_origen_" in readme
    # README documenta los nombres con km (30m / 1km), no en metros
    assert "1km" in readme

    # celda markdown del notebook dedicada al modo nuevo
    markdown = [c.source for c in NB.cells if c.cell_type == "markdown"]
    assert any("cross_layers" in md and "M x N" in md for md in markdown)


def test_s11_export_matches_the_notebook_code_cells():
    export = (ROOT / "LCP_MCP_Geometric.py").read_text(encoding="utf-8")
    # El export de nbconvert reproduce cada celda de codigo tal cual, en orden
    # (las celdas markdown van como comentarios, los code cells sin cambios).
    cursor = 0
    for index, source in enumerate(SRC):
        found = export.find(source.strip(), cursor)
        assert found >= 0, f"celda de codigo {index} ausente del export (drift)"
        cursor = found + 1

    # y contiene los marcadores de las correcciones de esta ronda
    assert "_estado_raiz(" in export
    assert "edge_buffer_px" in export
    assert "HEATMAP_BANDWIDTH_M / 1000" in export


def test_b2_per_root_heatmap_name_uses_km(tmp_path):
    # Sonda original del verificador (fallaba a proposito): la API recibe km, asi
    # que un radio de 1 km se nombra `1km` y nunca `1000km` ni una fraccion.
    paths = outputs.session_paths(tmp_path)
    name = pathlib.Path(
        outputs.heatmap_artifact(paths, 1, 30, 1000 / 1000, "quartic")
    ).name
    assert name == "heatmap_origen_1_30m_1km_quartic.tif"
    assert "1000" not in name

    # la Celda 9 por raiz llama con la conversion, igual que el camino legacy
    cell9 = SRC[8]
    assert "HEATMAP_BANDWIDTH_M / 1000" in cell9
    assert "int(HEATMAP_BANDWIDTH_M/1000)" in cell9
