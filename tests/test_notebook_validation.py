"""S4: validacion de la Celda 4 con fixtures negativas via mini-notebook."""

import notebook_harness as nh


def test_cells_1_to_4_pass_on_sample_data(tmp_path):
    # camino positivo con los datos reales del repo (solo OUTPUT_DIR a tmp
    # para no acumular sesiones en output/)
    srcs = nh.load_code_sources()
    srcs[2] = nh.patch_cell3(
        srcs[2],
        {
            "OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}_Refactored')": (
                f"OUTPUT_DIR = os.path.join({str(tmp_path)!r}, f'session_{{timestamp}}')"
            )
        },
    )
    out, err = nh.run_code(srcs[:4])
    assert err is None, err
    assert "Todos los datos son coherentes" in out


def test_cell4_rejects_mismatched_crs(tmp_path):
    pts = tmp_path / "pts_4326.shp"
    nh.write_points(pts, [(1, -58.4, -34.6), (2, -58.5, -34.7)], crs="EPSG:4326")
    srcs = nh.build_cells_1_to_4(tmp_path, points=pts, mask=None)
    _, err = nh.run_code(srcs)
    assert err is not None, "la validacion debia fallar por CRS desalineado"
    assert (
        "El CRS de los PUNTOS no coincide con el del RASTER".replace("RASTER", "RÁSTER")
        in err
    )


def test_cell4_lists_points_on_invalid_pixels(tmp_path):
    # raster sintetico: columnas >= 50 son nodata; punto 2 cae ahi dentro de
    # la extension -> debe listarse su ID y detener la ejecucion.
    raster = tmp_path / "cost.tif"
    nh.write_synthetic_raster(raster)
    pts = tmp_path / "pts_mixed.shp"
    nh.write_points(pts, [(1, 10.5, 89.5), (2, 60.5, 89.5)], crs="EPSG:32719")
    srcs = nh.build_cells_1_to_4(tmp_path, raster=raster, points=pts, mask=None)
    _, err = nh.run_code(srcs)
    assert err is not None, "el punto sobre pixel nodata debia detener la ejecucion"
    assert "puntos sobre píxeles INVÁLIDOS" in err
    assert "IDs de muestra: 2" in err


def test_cell6_writes_unreachable_layer(tmp_path):
    # S8 (corrida con barrera): raster con una columna nodata que desconecta
    # dos regiones validas; 4 puntos (2 por lado) -> solo 2 rutas alcanzables
    # de 6 esperadas y la capa 'destinos_inalcanzables' con los 4 pares cruzados.
    import fiona

    raster = tmp_path / "cost.tif"
    nh.write_barrier_raster(raster, barrier_col=50)
    pts = tmp_path / "pts_barriera.shp"
    nh.write_points(
        pts,
        [(1, 10.5, 89.5), (2, 20.5, 89.5), (3, 60.5, 89.5), (4, 70.5, 89.5)],
        crs="EPSG:32719",
    )
    srcs = nh.build_cells_1_to_4(tmp_path, raster=raster, points=pts, mask=None)
    cell6 = nh.load_code_sources()[5]  # sin Celda 5 (plot): no aporta al calculo
    out, err = nh.run_code([*srcs, cell6], timeout=600)
    assert err is None, err

    gpkg = next(tmp_path.glob("session_*/red_completa_unificada.gpkg"))
    layers = fiona.listlayers(gpkg)
    assert "rutas_unificadas" in layers
    assert "destinos_inalcanzables" in layers

    with fiona.open(gpkg, layer="rutas_unificadas") as src:
        assert len(src) == 2  # (1,2) y (3,4): solo dentro de cada region
    with fiona.open(gpkg, layer="destinos_inalcanzables") as src:
        rows = [(f["properties"]["origin_id"], f["properties"]["dest_id"]) for f in src]
    assert sorted(rows) == [(1, 3), (1, 4), (2, 3), (2, 4)]  # len==4: sin dupes

    # ademas: el aviso de reales != esperadas (2 de 6) y el de inalcanzables
    assert "!= esperadas" in out
    assert "inalcanzables" in out


def test_cell6_resume_reuses_phase1(tmp_path):
    # S9 (corrida funcional): dos corridas sobre la MISMA sesion; la segunda
    # debe reutilizar los 3 tracebacks de Fase 1 (huella coincide, pendientes 0)
    # y producir exactamente los mismos resultados.
    import fiona

    raster = tmp_path / "cost.tif"
    nh.write_barrier_raster(raster, barrier_col=50)
    pts = tmp_path / "pts_barriera.shp"
    nh.write_points(
        pts,
        [(1, 10.5, 89.5), (2, 20.5, 89.5), (3, 60.5, 89.5), (4, 70.5, 89.5)],
        crs="EPSG:32719",
    )

    def sources():
        srcs = nh.build_cells_1_to_4(
            tmp_path,
            raster=raster,
            points=pts,
            mask=None,
            fixed_output_name="session_resume",
            keep_phase1=True,
        )
        return [*srcs, nh.load_code_sources()[5]]

    # corrida 1: todo fresco, sin reutilizacion
    out1, err1 = nh.run_code(sources(), timeout=600)
    assert err1 is None, err1
    assert "RESUME: reutilizando" not in out1

    # corrida 2: huella coincide -> 3 tracebacks reutilizados, 0 pendientes
    out2, err2 = nh.run_code(sources(), timeout=600)
    assert err2 is None, err2
    assert "RESUME: reutilizando 3 tracebacks" in out2
    assert "pendientes: 0" in out2

    # resultados identicos a los de la corrida fresca
    gpkg = tmp_path / "session_resume" / "red_completa_unificada.gpkg"
    with fiona.open(gpkg, layer="rutas_unificadas") as src:
        assert len(src) == 2
    with fiona.open(gpkg, layer="destinos_inalcanzables") as src:
        rows = [(f["properties"]["origin_id"], f["properties"]["dest_id"]) for f in src]
    assert sorted(rows) == [(1, 3), (1, 4), (2, 3), (2, 4)]  # len==4: sin dupes


def test_s8_unreachable_layer_is_idempotent_across_reruns(tmp_path):
    """S8: re-ejecutar la Celda 6 en la MISMA sesion (flujo RESUME de S9)
    debe dejar exactamente los mismos pares inalcanzables, sin duplicados."""
    import fiona

    raster = tmp_path / "cost.tif"
    nh.write_barrier_raster(raster, barrier_col=50)
    pts = tmp_path / "pts_idem.shp"
    nh.write_points(
        pts,
        [(1, 10.5, 89.5), (2, 20.5, 89.5), (3, 60.5, 89.5), (4, 70.5, 89.5)],
        crs="EPSG:32719",
    )

    def sources():
        srcs = nh.build_cells_1_to_4(
            tmp_path,
            raster=raster,
            points=pts,
            mask=None,
            fixed_output_name="session_idem",
            keep_phase1=True,
        )
        return [*srcs, nh.load_code_sources()[5]]

    for _ in range(3):
        _out, err = nh.run_code(sources(), timeout=600)
        assert err is None, err

    gpkg = tmp_path / "session_idem" / "red_completa_unificada.gpkg"
    # 3 corridas: exactamente 2 capas (principal intacta), esquema estable y
    # 4 filas sin duplicados en la capa de inalcanzables.
    assert sorted(fiona.listlayers(gpkg)) == [
        "destinos_inalcanzables",
        "rutas_unificadas",
    ]
    with fiona.open(gpkg, layer="rutas_unificadas") as src:
        assert len(src) == 2
    with fiona.open(gpkg, layer="destinos_inalcanzables") as src:
        assert list(src.schema["properties"]) == ["origin_id", "dest_id", "cost_total"]
        rows = [(f["properties"]["origin_id"], f["properties"]["dest_id"]) for f in src]
    assert sorted(rows) == [(1, 3), (1, 4), (2, 3), (2, 4)], f"duplicados: {rows}"


def test_s8_no_stale_rows_when_rerun_has_zero_unreachable(tmp_path):
    """Advisory L15/L16: una re-corrida en la MISMA sesion que ahora deja
    0 pares inalcanzables no debe conservar filas stale en la capa."""
    import fiona

    raster = tmp_path / "cost.tif"
    nh.write_barrier_raster(raster, barrier_col=50)
    pts = tmp_path / "pts_stale.shp"
    nh.write_points(
        pts,
        [(1, 10.5, 89.5), (2, 20.5, 89.5), (3, 60.5, 89.5), (4, 70.5, 89.5)],
        crs="EPSG:32719",
    )

    def sources():
        srcs = nh.build_cells_1_to_4(
            tmp_path,
            raster=raster,
            points=pts,
            mask=None,
            fixed_output_name="session_stale",
            keep_phase1=True,
        )
        return [*srcs, nh.load_code_sources()[5]]

    # corrida 1: barrera -> 4 pares inalcanzables
    _out1, err1 = nh.run_code(sources(), timeout=600)
    assert err1 is None, err1
    gpkg = tmp_path / "session_stale" / "red_completa_unificada.gpkg"
    with fiona.open(gpkg, layer="destinos_inalcanzables") as src:
        assert len(src) == 4

    # corrida 2: mismos archivos pero TODOS los puntos del mismo lado de la
    # barrera -> 0 inalcanzables (la huella cambia: puntos distintos)
    nh.write_points(
        pts,
        [(1, 10.5, 89.5), (2, 20.5, 89.5), (3, 30.5, 89.5), (4, 40.5, 89.5)],
        crs="EPSG:32719",
    )
    _out2, err2 = nh.run_code(sources(), timeout=600)
    assert err2 is None, err2

    with fiona.open(gpkg, layer="destinos_inalcanzables") as src:
        stale = [
            (f["properties"]["origin_id"], f["properties"]["dest_id"]) for f in src
        ]
    assert stale == [], f"filas stale: {stale}"
    # la capa principal se reescribio con las 6 rutas ahora alcanzables
    with fiona.open(gpkg, layer="rutas_unificadas") as src:
        assert len(src) == 6


def test_s9_fingerprint_mismatch_recomputes_phase1(tmp_path):
    """S9: huella distinta -> nunca reutilizar (wipe + recalculo de Fase 1)."""
    import json as _json

    raster = tmp_path / "cost.tif"
    nh.write_barrier_raster(raster, barrier_col=50)
    pts = tmp_path / "pts_fp.shp"
    nh.write_points(
        pts,
        [(1, 10.5, 89.5), (2, 20.5, 89.5), (3, 60.5, 89.5), (4, 70.5, 89.5)],
        crs="EPSG:32719",
    )
    srcs = nh.build_cells_1_to_4(
        tmp_path,
        raster=raster,
        points=pts,
        mask=None,
        fixed_output_name="session_fp",
        keep_phase1=True,
    )
    cell6 = nh.load_code_sources()[5]

    _out1, err1 = nh.run_code([*srcs, cell6], timeout=600)
    assert err1 is None, err1
    meta = tmp_path / "session_fp" / "phase1_tracebacks" / "phase1_meta.json"
    m = _json.loads(meta.read_text(encoding="utf-8"))
    m["fingerprint"] = "0" * 64
    meta.write_text(_json.dumps(m), encoding="utf-8")

    out2, err2 = nh.run_code([*srcs, cell6], timeout=600)
    assert err2 is None, err2
    assert "reutilizando" not in out2
