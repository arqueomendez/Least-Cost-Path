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
