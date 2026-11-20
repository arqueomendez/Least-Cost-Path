# Tests del kernel A* jerárquico con corredor (opción 1->1 sobre od_pair).
# Superficies sintéticas: una barrera de coste alto que obliga a rodear,
# igual que tests/synthetic_surfaces.py del repo.

import math
from itertools import pairwise

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


class TestReconstructPathLimitGuard:
    def test_came_from_cycle_returns_none_without_oob_write(self):
        # S8/R3-reconstruct-path-limit-oob: un came_from con un ciclo fuerza
        # count == limit en el bucle de reconstruccion; el write final debe
        # acotarse y devolver None, nunca escribir en path[limit] (escritura
        # fuera de rango bajo numba sin bounds checking).
        came_from = np.full((3, 3), -1, dtype=np.int16)
        came_from[1, 1] = 5  # dr=0, dc=+1 -> predecesor (1,0)
        came_from[1, 0] = 3  # dr=0, dc=-1 -> predecesor (1,1): ciclo con (1,1)
        result = pf.reconstruct_path(came_from, (0, 0), (1, 1))
        assert result is None


class TestAStarOpenSetHeap:
    """S7/R3-plan-b-quadratic-openset: el open set de a_star_search es un heap
    binario manual (antes: vstack + argmin por pop, cuadratico en celdas
    exploradas). Misma semantica de f/g; capacidad crece por duplicacion."""

    def test_heap_helpers_maintain_min_heap(self):
        heap = np.array(
            [[5.0, 0, 0, 0], [3.0, 0, 0, 0], [8.0, 0, 0, 0], [1.0, 0, 0, 0]],
            dtype=np.float64,
        )
        for i in range(1, 4):
            pf._heap_sift_up(heap, i)
        # Un min-heap no esta ordenado: se verifica la propiedad de heap.
        assert heap[0, 0] == pytest.approx(1.0)
        assert heap[0, 0] <= heap[1, 0] and heap[0, 0] <= heap[2, 0]
        assert heap[1, 0] <= heap[3, 0]
        # pop simulado: ultima entrada a la raiz y hundir
        heap[0] = heap[3]
        pf._heap_sift_down(heap, 3, 0)
        assert heap[0, 0] == pytest.approx(3.0)
        assert heap[0, 0] <= heap[1, 0] and heap[0, 0] <= heap[2, 0]

    def test_heap_grows_beyond_initial_capacity(self):
        # La busqueda en la barrera 600x600 empuja mucho mas entradas que la
        # capacidad inicial del heap: el crecimiento por duplicacion debe ser
        # transparente y la ruta sigue siendo valida.
        surface = _surface_with_barrier()
        assert pf._HEAP_INITIAL_CAPACITY < 4096
        start, end = (100, 100), (500, 500)
        found, came_from = pf.a_star_search(
            surface, -9999.0, start, end, 1.0, 1.0, 1.0, _empty_mask(surface.shape)
        )
        assert found
        path = pf.reconstruct_path(came_from, start, end)
        assert path is not None
        assert path[0].tolist() == [100, 100]
        assert path[-1].tolist() == [500, 500]
        assert np.isfinite(pf.accumulate_cost(surface, -9999.0, 1.0, 1.0, path))


class TestHeuristicAdmissibility:
    """S6/R3-heuristic-admissibility-no-runtime-guard + quinto review
    (R2-overbroad/R3-constant-threshold): la heuristica euclidiana ponderada
    es admisible solo si weight <= minimo de coste transitable; el aviso
    runtime es weight-relativo (no el umbral fijo 1.0) y un weight negativo
    levanta ValueError."""

    def test_warns_when_costs_below_one(self, capsys):
        # costs 0.5 con weight default 1.0: weight > min_cost -> aviso.
        surface = np.full((40, 40), 0.5, dtype=np.float64)
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, 1.0, 1.0, (5, 5), (35, 35), _empty_mask(surface.shape)
        )
        assert path is not None and np.isfinite(cost_total)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" in out
        assert "heuristic_weight" in out
        assert "HEURISTIC_WEIGHT" in out

    def test_no_warning_when_all_costs_above_one(self, capsys):
        surface = _surface_with_barrier()
        assert float(surface[surface != np.inf].min()) >= 1.0
        pf.corridor_astar_route(
            surface,
            -9999.0,
            1.0,
            1.0,
            (100, 100),
            (500, 500),
            _empty_mask(surface.shape),
        )
        out = capsys.readouterr().out
        assert "ADVERTENCIA" not in out

    def test_no_warning_when_weight_at_or_below_min_cost(self, capsys):
        # R2-admissibility-warning-overbroad (quinto review): con weight <=
        # min_coste la busqueda es admisible aunque los costes sean < 1.0;
        # avisar seria un falso positivo.
        surface = np.full((40, 40), 0.5, dtype=np.float64)
        path, cost_total = pf.corridor_astar_route(
            surface,
            -9999.0,
            1.0,
            1.0,
            (5, 5),
            (35, 35),
            _empty_mask(surface.shape),
            heuristic_weight=0.5,
        )
        assert path is not None and np.isfinite(cost_total)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" not in out

    def test_warns_when_weight_exceeds_min_cost_above_one(self, capsys):
        # R3-admissibility-guard-constant-threshold (quinto review): costes en
        # [1.0, weight) con weight > 1.0 son inadmisibles y ANTES no avisaban.
        surface = np.full((40, 40), 1.5, dtype=np.float64)
        path, cost_total = pf.corridor_astar_route(
            surface,
            -9999.0,
            1.0,
            1.0,
            (5, 5),
            (35, 35),
            _empty_mask(surface.shape),
            heuristic_weight=2.0,
        )
        assert path is not None and np.isfinite(cost_total)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" in out

    def test_negative_weight_raises(self):
        # R3-heuristic-weight-unvalidated (quinto review): un weight negativo
        # premia alejarse del objetivo; rechazo explicito, nunca ruta silenciosa.
        surface = _surface_with_barrier()
        with pytest.raises(ValueError, match="invalido"):
            pf.corridor_astar_route(
                surface,
                -9999.0,
                1.0,
                1.0,
                (100, 100),
                (500, 500),
                _empty_mask(surface.shape),
                heuristic_weight=-0.5,
            )

    @pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_weight_raises(self, bad_weight):
        # R3-001 (sexto review): NaN/inf pasan ambos umbrales (< 0 y > 0) y
        # corrompen el puntaje f = g + w*h en silencio; rechazo explicito.
        surface = _surface_with_barrier()
        with pytest.raises(ValueError, match="invalido"):
            pf.corridor_astar_route(
                surface,
                -9999.0,
                1.0,
                1.0,
                (100, 100),
                (500, 500),
                _empty_mask(surface.shape),
                heuristic_weight=bad_weight,
            )

    def test_no_warning_with_weight_zero_on_cheap_surface(self, capsys):
        # weight=0 equivale a Dijkstra exacto: siempre admisible, sin aviso.
        surface = np.full((40, 40), 0.5, dtype=np.float64)
        path, cost_total = pf.corridor_astar_route(
            surface,
            -9999.0,
            1.0,
            1.0,
            (5, 5),
            (35, 35),
            _empty_mask(surface.shape),
            heuristic_weight=0.0,
        )
        assert path is not None and np.isfinite(cost_total)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" not in out

    def test_warning_ignores_nodata_and_masked_cells(self, capsys):
        # nodata finito (-9999) y celdas fuera de la mascara no participan del
        # minimo: solo las celdas transitables con coste finito cuentan.
        surface = np.full((40, 40), 2.0, dtype=np.float64)
        surface[0, 0] = -9999.0  # nodata finito dentro de la mascara
        surface[39, 39] = np.inf  # fuera de la mascara real
        mask = _empty_mask(surface.shape)
        mask[39, 39] = False
        pf.corridor_astar_route(surface, -9999.0, 1.0, 1.0, (5, 5), (35, 35), mask)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" not in out


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

    def test_cost_matches_independent_recomputation(self):
        # S5/R3-cost-consistency-tests-tautological (revision): el coste del
        # kernel se compara contra una RECOMPUTACION INDEPENDIENTE del modelo
        # de coste hecha aqui (suma de coste medio entre celdas consecutivas
        # x distancia entre centros), NO contra una re-llamada de
        # pf.accumulate_cost sobre la misma ruta (tautologia).
        surface = _surface_with_barrier()
        res = 1.0
        path, cost_total = pf.corridor_astar_route(
            surface, -9999.0, res, res, (100, 100), (500, 500), _empty_mask(surface.shape)
        )
        assert path is not None
        independent = 0.0
        for (r0, c0), (r1, c1) in pairwise(path):
            cost_0 = float(surface[r0, c0])
            cost_1 = float(surface[r1, c1])
            assert np.isfinite(cost_0) and np.isfinite(cost_1)
            dist_m = math.sqrt(((r1 - r0) * res) ** 2 + ((c1 - c0) * res) ** 2)
            independent += ((cost_0 + cost_1) / 2.0) * dist_m
        assert cost_total == pytest.approx(independent)

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
