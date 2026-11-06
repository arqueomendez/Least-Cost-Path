"""S10 (T6) del modo cross_layers: Celdas 7, 8 y 9 por raiz + docs del modo.

Pins estructurales sobre la fuente de las celdas de codigo 6, 7 y 8 (Celdas 7,
8 y 9 del notebook), mas una corrida end-to-end de las nueve celdas con
fixtures propias de dos capas. El camino legacy (`all_pairs` / `single_origin`)
se pinea aparte para comprobar que sigue apuntando a los archivos unificados.
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

CELL7 = 6  # Celda 7 (mapas) entre las celdas de codigo
CELL8 = 7  # Celda 8 (puntos densificados)
CELL9 = 8  # Celda 9 (KDE)

MARKER7 = "if _IS_CROSS_MAPS:"
MARKER8 = "if _IS_CROSS_DENSE:"
MARKER9 = "if _IS_CROSS_KDE:"
MARKER5 = "if _IS_CROSS_DIAG:"

CELL5 = 4  # Celda 5 (visualizacion diagnostica) entre las celdas de codigo

# Ids repetidos (1, 2) en AMBAS capas a proposito: cada capa aporta sus propias
# coordenadas y nunca se fusionan (mismo fixture que el e2e de las Celdas 1..6).
ORIGINS = [(1, 10.5, 89.5), (2, 20.5, 89.5)]
DESTS = [(1, 60.5, 40.5), (2, 70.5, 30.5)]

HEATMAP_RES_M = 7
HEATMAP_BANDWIDTH_M = 21


def _strip_comments(src):
    """Quita lineas de comentario y comentarios a fin de linea (2+ espacios)."""
    out = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(line.split("  #")[0])
    return out


def _split_branch(src, marker):
    """Separa la rama cross_layers del `else:` legacy de una celda.

    Devuelve (lineas_sin_comentarios, inicio_rama, linea_del_else).
    """
    lines = _strip_comments(src)
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(marker))
    indent = len(lines[start]) - len(lines[start].lstrip())
    else_prefix = " " * indent + "else:"
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].rstrip() == else_prefix
    )
    return lines, start, end


def _branch_text(src, marker):
    """Texto de la rama cross_layers y texto del camino legacy de una celda."""
    lines, start, end = _split_branch(src, marker)
    return "\n".join(lines[start:end]), "\n".join(lines[end:])


# ---------------------------------------------------------------------------
# 1. Pins estructurales de las Celdas 7, 8 y 9
# ---------------------------------------------------------------------------
def test_cell5_cross_branch_plots_both_layers_from_their_own_paths():
    # Regresion del NameError: en cross_layers la Celda 5 no puede depender de
    # `all_points_for_validation` (solo existe en el camino legacy); lee las dos
    # capas de sus rutas con su propio campo id.
    cross, legacy = _branch_text(SRC[CELL5], MARKER5)
    assert "ORIGIN_POINTS_PATH" in cross
    assert "DEST_POINTS_PATH" in cross
    assert "ORIGIN_ID_FIELD" in cross
    assert "DEST_ID_FIELD" in cross
    assert "dl.load_points_as_dict(" in cross
    assert "all_points_for_validation" not in cross
    # el camino legacy conserva su fuente de puntos y su resaltado de origen
    assert "all_points_for_validation" in legacy
    assert 'marker="*"' in legacy
    # ambas ramas exportan el mismo archivo de diagnostico
    assert "visualizacion_diagnostica.png" in cross
    assert "visualizacion_diagnostica.png" in legacy
    # capas distintas: color y marcador distintos, con la leyenda nombrandolas
    assert 'marker="o"' in cross and 'marker="^"' in cross
    assert "Origenes (" in cross and "Destinos (" in cross


def test_cell7_cross_branch_renders_one_map_per_root():
    # S10: una figura por raiz, guardada en figure_artifact y anotada en el
    # manifiesto; la figura se cierra dentro del bucle (M raices, M cierres).
    cross, _legacy = _branch_text(SRC[CELL7], MARKER7)
    assert "outputs.read_manifest(" in cross
    assert "outputs.session_paths(" in cross
    assert "outputs.route_artifact(" in cross
    assert "outputs.figure_artifact(" in cross
    assert "outputs.write_manifest(" in cross
    assert '_row["mapa"] = _png' in cross
    assert "plt.close(_fig)" in cross
    # solo las raices en estado 'ok' producen mapa
    assert 'row.get("estado") == "ok"' in cross
    # una linea de progreso por raiz
    assert "for _i, _row in enumerate(_rows_ok, start=1):" in cross
    assert 'f"  [mapa {_i}/{len(_rows_ok)}] raiz {_root} ' in cross
    # origenes y destinos se marcan por separado, con las etiquetas de capa
    assert "origin_layer" in cross and "dest_layer" in cross
    assert "_origin_label" in cross and "_dest_label" in cross


def test_cell8_cross_branch_densifies_one_gpkg_per_root():
    # S10: un GPKG densificado por raiz, reutilizando el worker vectorizado y
    # escribiendo la columna route_idx que consume la Celda 9.
    cross, _legacy = _branch_text(SRC[CELL8], MARKER8)
    assert "outputs.read_manifest(" in cross
    assert "outputs.route_artifact(" in cross
    assert "outputs.dense_artifact(" in cross
    assert "outputs.write_manifest(" in cross
    assert '_row["densificado"] = _out_path' in cross
    assert "plt.close(" in cross
    assert 'f"puntos_origen_{_root}"' in cross
    assert '"route_idx"' in cross
    # el worker densify_chunk es el mismo del camino legacy (sin reimplementar)
    assert "delayed(densify_chunk)" in cross
    assert "for _i, _row in enumerate(_rows_ok, start=1):" in cross


def test_cell9_cross_branch_kde_one_tif_per_root():
    # S10: un GeoTIFF por raiz, secuencial (sin workers) y con el recuento y el
    # coste estimado impresos ANTES de arrancar.
    cross, _legacy = _branch_text(SRC[CELL9], MARKER9)
    assert "outputs.read_manifest(" in cross
    assert "outputs.dense_artifact(" in cross
    assert "outputs.heatmap_artifact(" in cross
    assert "outputs.write_manifest(" in cross
    assert '_row["heatmap"] = _tif' in cross
    assert "plt.close(_fig)" in cross
    assert "HEATMAP_ORIGIN_IDS" in cross
    assert "Raices a procesar:" in cross
    assert "Coste estimado:" in cross
    # el KDE corre secuencial: nunca en workers paralelos
    assert "Parallel" not in cross
    assert "n_jobs" not in cross


def test_cell9_kernel_fft_computed_once_outside_the_root_loop():
    # S10: la FFT del kernel se calcula en UN solo sitio y ese sitio esta antes
    # del bucle por raiz (cache compartida), nunca dentro del bucle.
    lines, start, end = _split_branch(SRC[CELL9], MARKER9)
    calls = [i for i, line in enumerate(lines) if "rfft2(kernel_padded" in line]
    assert len(calls) == 1, "debe haber una sola FFT del kernel en la Celda 9"
    assert calls[0] < start, (
        "la FFT del kernel debe calcularse fuera del bucle por raiz"
    )
    cross = "\n".join(lines[start:end])
    assert "rfft2(" not in cross
    assert "kernel_cache=_KERNEL_FFT_CACHE" in cross
    assert "_KERNEL_FFT_CACHE = {}" in "\n".join(lines[:start])


def test_cell9_legacy_and_cross_share_the_same_kde_maths():
    # La rama por raiz no reimplementa el KDE: llama a _compute_heatmap, la
    # misma funcion de la rama legacy (grid, pixel_size real, binning y crop).
    lines, start, end = _split_branch(SRC[CELL9], MARKER9)
    cross = "\n".join(lines[start:end])
    legacy = "\n".join(lines[end:])
    assert "_compute_heatmap(" in cross
    assert "_compute_heatmap(" in legacy
    assert "from_bounds(" in cross and "from_bounds(" in legacy


def test_legacy_branches_keep_unified_outputs():
    # Los modos legacy conservan sus archivos unificados exactos.
    _cross, legacy7 = _branch_text(SRC[CELL7], MARKER7)
    assert "red_completa_unificada.gpkg" in legacy7
    assert "red_de_rutas_final.png" in legacy7

    _cross, legacy8 = _branch_text(SRC[CELL8], MARKER8)
    assert "puntos_rutas_{int(SPACING_M)}m.gpkg" in legacy8

    _cross, legacy9 = _branch_text(SRC[CELL9], MARKER9)
    assert "heatmap_{HEATMAP_RES_M}m_{int(HEATMAP_BANDWIDTH_M/1000)}km_" in legacy9
    assert ".tif'" in legacy9


# ---------------------------------------------------------------------------
# 2. Documentacion del modo dentro del notebook (S11)
# ---------------------------------------------------------------------------
def test_markdown_cell_documents_cross_layers_mode():
    # Celda markdown en espanol insertada justo despues de la que documenta los
    # otros dos modos y antes de la Celda 3 (patron de test_star_origin_mode).
    code_idx = next(
        i
        for i, c in enumerate(NB.cells)
        if c.cell_type == "code" and 'ANALYSIS_MODE = "all_pairs"' in c.source
    )
    modes_idx = next(
        i
        for i, c in enumerate(NB.cells)
        if c.cell_type == "markdown"
        and "all_pairs" in c.source
        and "single_origin" in c.source
    )
    doc_idx = modes_idx + 1
    assert NB.cells[doc_idx].cell_type == "markdown"
    assert doc_idx < code_idx
    md = NB.cells[doc_idx].source
    for token in (
        "cross_layers",
        "M x N",
        "ORIGIN_POINTS_PATH",
        "DEST_POINTS_PATH",
        "INVERT_DIRECTION",
        '"auto"',
        '"never"',
        '"always"',
        "length_m",
        "cost_total",
        "manifest_rutas_por_origen.csv",
        "rutas_por_origen",
        "Cuándo usarlo",
        "Qué calcula",
    ):
        assert token in md, token
    # sin Unicode decorativo (flechas, productos, em dashes)
    for codepoint in (0x2192, 0x2190, 0x00D7, 0x2014, 0x00B2):
        assert chr(codepoint) not in md, hex(codepoint)


def test_cell3_declares_heatmap_origin_ids_limit():
    # El limite opcional del KDE es un parametro (tag 'parameters') de la Celda 3.
    assert "HEATMAP_ORIGIN_IDS = None" in SRC[2]
    cells = [c for c in NB.cells if c.cell_type == "code"]
    assert "parameters" in cells[2].metadata.get("tags", [])


# ---------------------------------------------------------------------------
# 3. Corrida end-to-end: celdas 1..9 con dos capas de fixtures
# ---------------------------------------------------------------------------
def _write_flat_raster(path):
    """Raster 100x100 EPSG:32719, 1 m, coste 1.0 y sin nodata."""
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


def _patched_sources(tmp_path, raster, origins, dests):
    """Las nueve celdas de codigo con Celda 3 (y 8, 9) parchadas a los fixtures."""
    session = tmp_path / "session_e2e"
    srcs = nh.load_code_sources()
    srcs[2] = nh.patch_cell3(
        srcs[2],
        {
            "OUTPUT_DIR = os.path.join(BASE_DIR, 'output', "
            "f'session_{timestamp}_Refactored')": f"OUTPUT_DIR = {str(session)!r}",
            "COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')": (
                f"COST_RASTER_PATH = {str(raster)!r}"
            ),
            'ANALYSIS_MODE = "all_pairs"': 'ANALYSIS_MODE = "cross_layers"',
            "ORIGIN_POINTS_PATH = None": f"ORIGIN_POINTS_PATH = {str(origins)!r}",
            "DEST_POINTS_PATH = None": f"DEST_POINTS_PATH = {str(dests)!r}",
            "MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')": (
                "MASK_SHAPEFILE_PATH = None"
            ),
            "MAX_WORKERS_PHASE1 = None": "MAX_WORKERS_PHASE1 = 1",
            'INVERT_DIRECTION = "auto"': 'INVERT_DIRECTION = "never"',
        },
    )
    # Celda 8: espaciado chico y sin workers (fixtures minusculas).
    srcs[7] = nh.patch_cell3(
        srcs[7],
        {
            "SPACING_M = 25.0": "SPACING_M = 5.0",
            "N_WORKERS = min(18, psutil.cpu_count(logical=True) or 1)": "N_WORKERS = 1",
        },
        tag="cell8",
    )
    # Celda 9: grilla chica y rapida.
    srcs[8] = nh.patch_cell3(
        srcs[8],
        {
            "HEATMAP_RES_M = 30": f"HEATMAP_RES_M = {HEATMAP_RES_M}",
            "HEATMAP_BANDWIDTH_M = 1000": f"HEATMAP_BANDWIDTH_M = {HEATMAP_BANDWIDTH_M}",
        },
        tag="cell9",
    )
    return srcs, session


def _manifest(session):
    return outputs.read_manifest(
        {"manifest": str(session / "manifest_rutas_por_origen.csv")}
    )


def _progress_roots(out, tag):
    """Ids de raiz de las lineas de progreso `[<tag> i/n] raiz <id>: ...`.

    El orden de las raices no es determinista (Fase 2 usa generador sin orden).
    """
    roots = set()
    for line in out.splitlines():
        if f"[{tag} " not in line or " raiz " not in line:
            continue
        roots.add(line.split(" raiz ", 1)[1].split(" ", 1)[0].rstrip(":"))
    return roots


def test_e2e_cross_layers_cells_1_to_9_produce_artifacts_per_root(tmp_path):
    raster, origins, dests = _fixtures(tmp_path)
    srcs, session = _patched_sources(tmp_path, raster, origins, dests)
    out, err = nh.run_code(srcs, timeout=1200)
    assert err is None, err

    # --- Celda 5: diagnostico con las dos capas, exportado y sin errores ---
    # (antes del arreglo imprimia "Ocurrió un error durante la visualización:
    #  name 'all_points_for_validation' is not defined" y no generaba el PNG)
    assert "Ocurrió un error durante la visualización" not in out
    assert "NameError" not in out
    assert "Capas cargadas: 2 origenes (Sitios) y 2 destinos (Fuentes)" in out
    diag_png = session / "visualizacion_diagnostica.png"
    assert diag_png.exists()
    assert diag_png.stat().st_size > 0

    # --- Celda 7: un PNG por raiz ---
    assert _progress_roots(out, "mapa") == {"1", "2"}, out[-3000:]
    pngs = sorted((session / "mapas_por_origen").glob("mapa_origen_*.png"))
    assert [p.name for p in pngs] == ["mapa_origen_1.png", "mapa_origen_2.png"]
    assert all(p.stat().st_size > 0 for p in pngs)

    # --- Celda 8: un GPKG densificado por raiz, con la columna route_idx ---
    assert _progress_roots(out, "dens") == {"1", "2"}
    dense_paths = sorted((session / "puntos_por_origen").glob("puntos_origen_*.gpkg"))
    assert [p.name for p in dense_paths] == [
        "puntos_origen_1.gpkg",
        "puntos_origen_2.gpkg",
    ]
    for path in dense_paths:
        gdf = gpd.read_file(path)
        assert "route_idx" in gdf.columns
        assert len(gdf) > 0
        assert gdf.crs is not None

    # --- Celda 9: un GeoTIFF por raiz, georreferenciado con el pixel real ---
    assert "Raices a procesar: 2" in out
    assert "Coste estimado:" in out
    # la FFT del kernel se calcula una sola vez, antes del bucle, y se reutiliza
    assert "FFT del kernel en cache antes del bucle: 1" in out
    assert "FFT de kernel en cache: 1" in out
    assert _progress_roots(out, "kde") == {"1", "2"}
    tifs = sorted((session / "heatmaps_por_origen").glob("heatmap_origen_*.tif"))
    assert len(tifs) == 2

    # Grilla unica compartida: union de los bounds de los puntos densificados
    # de todas las raices, mas un margen = bandwidth (misma receta que legacy).
    margin = float(HEATMAP_BANDWIDTH_M)
    bounds = [gpd.read_file(p).total_bounds for p in dense_paths]
    minx = min(b[0] for b in bounds) - margin
    miny = min(b[1] for b in bounds) - margin
    maxx = max(b[2] for b in bounds) + margin
    maxy = max(b[3] for b in bounds) + margin
    span_x = maxx - minx
    span_y = maxy - miny
    expected_w = int(np.ceil(span_x / HEATMAP_RES_M))
    expected_h = int(np.ceil(span_y / HEATMAP_RES_M))
    for path in tifs:
        with rasterio.open(path) as src:
            assert src.crs is not None
            # pixel_size = span / width (NO HEATMAP_RES_M) y mismo para ambos ejes
            assert src.width == expected_w
            assert src.height == expected_h
            assert src.transform.a == pytest.approx(span_x / expected_w)
            assert -src.transform.e == pytest.approx(span_y / expected_h)
            assert src.bounds.left == pytest.approx(minx)
            assert src.bounds.right == pytest.approx(maxx)
            assert src.transform.a <= HEATMAP_RES_M
            assert float(src.read(1).sum()) > 0

    # --- Manifiesto: todas las raices ok y con las cuatro columnas llenas ---
    rows = _manifest(session)
    assert len(rows) == 2
    assert {row["root_id"] for row in rows} == {"1", "2"}
    for row in rows:
        assert row["estado"] == "ok"
        for column in ("gpkg", "mapa", "densificado", "heatmap"):
            assert row[column], (row["root_id"], column)
            assert pathlib.Path(row[column]).exists(), (row["root_id"], column)
        assert row["mapa"].endswith(".png")
        assert row["densificado"].endswith(".gpkg")
        assert row["heatmap"].endswith(".tif")
        assert f"rutas_origen_{row['root_id']}.gpkg" in row["gpkg"]
        assert f"mapa_origen_{row['root_id']}.png" in row["mapa"]
        assert f"puntos_origen_{row['root_id']}.gpkg" in row["densificado"]
        assert f"heatmap_origen_{row['root_id']}_" in row["heatmap"]

    # El modo nuevo no escribe el GPKG unificado en ninguna etapa.
    assert not list(session.glob("red_completa_unificada.gpkg"))
