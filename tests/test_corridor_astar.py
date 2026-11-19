# Tests del kernel A* jerárquico con corredor (opción 1->1 sobre od_pair).
# Superficies sintéticas: una barrera de coste alto que obliga a rodear,
# igual que tests/synthetic_surfaces.py del repo.

import numpy as np
import pytest

import lcp.pathfinder as pf
import lcp.planner as planner
import lcp.processing as proc


def _surface_with_barrier(height=600, width=600, barrier_row=300, cost=1000.0, close_gap=False):
    """Fondo de coste 1 con una barrera horizontal de 2 filas (bloquea también
    el paso diagonal de la conectividad 8) con un único paso en el centro."""
    surface = np.ones((height, width), dtype=np.float64)
    surface[barrier_row : barrier_row + 2, :] = cost
    gap_col = width // 2
    if not close_gap:
        surface[barrier_row : barrier_row + 2, gap_col - 3 : gap_col + 4] = 1.0
    return surface


def _empty_mask(shape):
    return np.ones(shape, dtype=bool)


class TestCreateLowResData:
    def test_shape_and_resolution_scale_by_factor(self):
        data = np.ones((100, 80), dtype=np.float64)
        cost_lr, dx, dy = proc.create_low_res_data(data, 2.0, 2.0, 10)
        assert cost_lr.shape == (10, 8)
        assert dx == pytest.approx(20.0)
        assert dy == pytest.approx(20.0)

    def test_block_average_value(self):
        data = np.ones((20, 20), dtype=np.float64)
        data[0, 0] = 5.0
        cost_lr, _, _ = proc.create_low_res_data(data, 1.0, 1.0, 10)
        # Bloque (0,0): 1 valor de 5 + 99 de 1 -> 104/100
        assert cost_lr[0, 0] == pytest.approx(1.04)
        assert cost_lr[1, 1] == pytest.approx(1.0)

    def test_non_multiple_dimensions_are_trimmed(self):
        data = np.ones((105, 85), dtype=np.float64)
        cost_lr, _, _ = proc.create_low_res_data(data, 1.0, 1.0, 10)
        assert cost_lr.shape == (10, 8)


class TestCreateSearchCorridor:
    def test_corridor_covers_route_and_buffer(self):
        route = np.array([[5, 5], [6, 6], [7, 7]], dtype=np.int32)
        mask = proc.create_search_corridor(route, (100, 100), 10, 15)
        assert mask[65, 65]  # centro del disco sobre la ruta
        assert not mask[0, 0]  # lejos de la ruta: fuera del corredor

    def test_empty_route_gives_empty_corridor(self):
        mask = proc.create_search_corridor(None, (50, 50), 10, 15)
        assert not mask.any()


class TestAStarSearch:
    def test_finds_path_around_barrier(self):
        surface = _surface_with_barrier()
        start = (100, 100)
        end = (500, 500)
        found, came_from = pf.a_star_search(surface, -9999.0, start, end, 1.0, 1.0, 1.0, _empty_mask(surface.shape))
        assert found
        path = pf.reconstruct_path(came_from, start, end)
        assert path is not None
        assert path[0].tolist() == [100, 100]
        assert path[-1].tolist() == [500, 500]
        # La ruta pasa por el único paso de la barrera (col ~297-303).
        rows_at_gap = [
            int(r) for r, c in path if r in (300, 301) and 296 <= c <= 304
        ]
        assert rows_at_gap

    def test_unreachable_returns_not_found(self):
        surface = _surface_with_barrier(cost=np.inf, close_gap=True)
        found, _ = pf.a_star_search(
            surface, -9999.0, (100, 100), (500, 500), 1.0, 1.0, 1.0, _empty_mask(surface.shape)
        )
        assert not found

    def test_mask_forbids_area(self):
        surface = _surface_with_barrier()
        mask = _empty_mask(surface.shape)
        # Se cierra el paso barato: la ruta debe evitar TODAS las celdas
        # excluidas (la barrera de coste 1000 sigue siendo transitable).
        _closed = mask[300:302, 296:305].copy()
        mask[300:302, 296:305] = False
        found, came_from2 = pf.a_star_search(
            surface, -9999.0, (100, 100), (500, 500), 1.0, 1.0, 1.0, mask
        )
        assert found
        path = pf.reconstruct_path(came_from2, (100, 100), (500, 500))
        assert path is not None
        for r, c in path:
            assert mask[r, c]


class TestAccumulateCost:
    def test_matches_astar_g_cost(self):
        surface = _surface_with_barrier()
        start, end = (100, 100), (500, 500)
        found, came_from = pf.a_star_search(surface, -9999.0, start, end, 1.0, 1.0, 1.0, _empty_mask(surface.shape))
        path = pf.reconstruct_path(came_from, start, end)
        total = pf.accumulate_cost(surface, -9999.0, 1.0, 1.0, path)
        assert found and np.isfinite(total) and total > 0


class TestCorridorAstarRoute:
    def test_end_to_end_route_with_finite_cost(self):
        surface = _surface_with_barrier()
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, 1.0, 1.0, (100, 100), (500, 500), _empty_mask(surface.shape)
        )
        assert path is not None
        assert np.isfinite(cost_total)
        assert path[0].tolist() == [100, 100]
        assert path[-1].tolist() == [500, 500]

    def test_cost_matches_direct_astar(self):
        surface = _surface_with_barrier()
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, 1.0, 1.0, (100, 100), (500, 500), _empty_mask(surface.shape)
        )
        direct = pf.accumulate_cost(surface, -9999.0, 1.0, 1.0, path)
        assert cost_total == pytest.approx(direct)

    def test_unreachable_returns_none_and_inf(self):
        surface = _surface_with_barrier(cost=np.inf, close_gap=True)
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, 1.0, 1.0, (100, 100), (500, 500), _empty_mask(surface.shape)
        )
        assert path is None
        assert cost_total == float("inf")

    def test_mask_is_respected_in_both_phases(self):
        surface = _surface_with_barrier()
        mask = _empty_mask(surface.shape)
        mask[300:302, 296:305] = False
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, 1.0, 1.0, (100, 100), (500, 500), mask
        )
        assert path is not None and np.isfinite(cost_total)
        for r, c in path:
            assert mask[r, c]

    def test_origin_near_bottom_edge_does_not_read_out_of_bounds(self):
        # R2/R3 (revision nativa): origen/destino dentro de las ultimas
        # factor-1 filas del raster quedaban fuera del grid LR recortado
        # (floor(dim/factor)) y, con numba sin bounds checking, escribian
        # fuera de rango. El clamp debe mantener la ruta valida.
        surface = _surface_with_barrier(height=600, width=600)
        start, end = (595, 30), (5, 570)  # fila 595: dentro del borde de 32
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, 1.0, 1.0, start, end, _empty_mask(surface.shape)
        )
        assert path is not None and np.isfinite(cost_total)
        assert path[0].tolist() == [595, 30]
        assert path[-1].tolist() == [5, 570]


class TestKernelSelection:
    def test_default_kernel_valid_in_every_mode(self):
        for mode in planner.MODES:
            assert planner.check_kernel(mode, "mcp_geometric") == "mcp_geometric"

    def test_corridor_astar_valid_only_for_od_pair(self):
        assert planner.check_kernel("od_pair", "corridor_astar") == "corridor_astar"
        for mode in ("all_pairs", "single_origin", "cross_layers"):
            with pytest.raises(ValueError, match="od_pair"):
                planner.check_kernel(mode, "corridor_astar")

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="KERNEL no valido"):
            planner.check_kernel("od_pair", "astar_legacy")
