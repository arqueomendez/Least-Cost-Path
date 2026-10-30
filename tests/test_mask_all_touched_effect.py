"""Pinned measurements of what ``all_touched`` actually changes in the mask.

Every number below was measured on the fixture defined in the same test and is
asserted verbatim, so a change in how the rasteriser treats the boundary fails
here instead of drifting silently. All assertions go through the public API
(``create_mask_from_vector`` / ``run_mcp``) and read observable values -- mask
contents, path vertices, path cost -- never internal state.

Pixel geometry (see ``synthetic_surfaces.build_transform``): on a ``rows x cols``
raster with ``scale=1`` pixel ``(r, c)`` covers ``x in [c, c + 1]`` and
``y in [rows - r - 1, rows - r]``, so its centre sits at
``(c + 0.5, rows - r - 0.5)``. A boundary that falls in ``(c, c + 0.5)`` leaves
the centre of column ``c`` outside the polygon (that column is only *touched*);
a boundary in ``(c + 0.5, c + 1)`` leaves it inside (no ring on that side).
"""

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from synthetic_surfaces import (
    CRS,
    StubRaster,
    build_transform,
    cheap_outside_polygon,
    path_total_cost,
    run_mcp,
    write_polygon,
)

import lcp.processing as proc


def _masks(tmp_path, geometry, rows, cols, name):
    """Write ``geometry`` and return ``(strict, touched, transform)``."""
    vector_path = tmp_path / f"{name}.shp"
    write_polygon(vector_path, geometry)
    transform = build_transform(rows, cols)
    stub = StubRaster(CRS, (rows, cols), transform)
    strict = proc.create_mask_from_vector(str(vector_path), stub)
    touched = proc.create_mask_from_vector(str(vector_path), stub, all_touched=True)
    return strict, touched, transform


def _max_row_run(mask):
    """Longest run of consecutive ``True`` cells on any single row of ``mask``.

    This is how many cells of the mask a horizontal scan line crosses, i.e. the
    ring thickness seen by a horizontal scan through the boundary.
    """
    best = 0
    for row in mask:
        run = 0
        for value in row:
            run = run + 1 if value else 0
            best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# 1. The strict mask is always contained in the touched mask.
# ---------------------------------------------------------------------------
SUBSET_ROWS = 12
SUBSET_COLS = 12
# Measured (strict, touched) pixel counts on the 12x12 raster.
SUBSET_CASES = [
    # Square whose four edges cross the interior of a pixel row/column: every
    # edge falls in (c, c + 0.5), so each side adds a ring column/row.
    ("square_crossing", box(2.6, 2.6, 8.4, 8.4), 25, 49),
    # 45-degree hypotenuse x + y = 12.3, deliberately off the pixel centres so
    # no centre lies exactly on the boundary (that would be a rasteriser knife
    # edge, not a property of the geometry).
    ("triangle45", Polygon([(1.3, 1.3), (11.0, 1.3), (1.3, 11.0)]), 55, 64),
    # Polygon strictly larger than the whole raster extent [0, 12] x [0, 12].
    ("contains_raster", box(-3.0, -3.0, 15.0, 15.0), 144, 144),
]


def test_strict_mask_is_always_a_subset_of_the_touched_mask(tmp_path):
    for name, geometry, expected_strict, expected_touched in SUBSET_CASES:
        strict, touched, _ = _masks(tmp_path, geometry, SUBSET_ROWS, SUBSET_COLS, name)
        assert int(strict.sum()) == expected_strict
        assert int(touched.sum()) == expected_touched
        # The defining relation: nothing is in the strict mask only.
        assert not (strict & ~touched).any()
        ring = touched & ~strict
        assert int(ring.sum()) == expected_touched - expected_strict


# ---------------------------------------------------------------------------
# 2. The user's real case: a polygon that contains the raster => no effect.
# ---------------------------------------------------------------------------
def test_no_effect_when_the_polygon_contains_the_raster(tmp_path):
    # box(-3, -3, 15, 15) strictly contains the raster extent [0, 12] x [0, 12],
    # so all four raster edges lie inside the polygon and every pixel centre is
    # inside it too.
    strict, touched, _ = _masks(tmp_path, box(-3.0, -3.0, 15.0, 15.0), 12, 12, "cont")

    assert int(strict.sum()) == 144
    assert int(touched.sum()) == 144

    # A cost surface with an expensive 6x6 block in the middle, so the route has
    # a reason to bend: if the two masks differed, the two routes would differ.
    costs = np.full((12, 12), 5.0)
    costs[3:9, 3:9] = 50.0
    start, end = (1, 1), (10, 10)

    strict_path, _ = run_mcp(np.where(strict, costs, np.inf), start, end)
    touched_path, _ = run_mcp(np.where(touched, costs, np.inf), start, end)

    # Identical vertices, hence identical endpoints and identical cost.
    assert np.array_equal(strict_path, touched_path)
    assert int(strict_path.shape[0]) == 16
    assert [int(v) for v in strict_path[0]] == [1, 1]
    assert [int(v) for v in strict_path[-1]] == [10, 10]
    assert path_total_cost(costs, strict_path) == pytest.approx(81.21320343559643)


# ---------------------------------------------------------------------------
# 3. Pixel-aligned edges: no ring at all.
# ---------------------------------------------------------------------------
def test_no_effect_when_the_polygon_edges_are_pixel_aligned(tmp_path):
    # Edges on x = 2, x = 8, y = 2, y = 8: they run along pixel boundaries, so
    # the polygon grazes its outer neighbours along a line of zero area. GDAL's
    # all_touched does NOT admit a pixel that the polygon touches only along an
    # edge, so both masks are the same 6x6 block (rows 4..9, columns 2..7).
    strict, touched, _ = _masks(tmp_path, box(2.0, 2.0, 8.0, 8.0), 12, 12, "aligned")

    assert np.array_equal(strict, touched)
    assert int(strict.sum()) == 36
    assert sorted({int(r) for r in np.where(touched)[0]}) == list(range(4, 10))
    assert sorted({int(c) for c in np.where(touched)[1]}) == list(range(2, 8))
    # The four pixel rows/columns just outside the polygon are in neither mask.
    assert not touched[3, :].any()
    assert not touched[10, :].any()
    assert not touched[:, 1].any()
    assert not touched[:, 8].any()

    # Same conclusion when the aligned edges are the raster edges themselves.
    strict_full, touched_full, _ = _masks(
        tmp_path, box(0.0, 0.0, 12.0, 12.0), 12, 12, "aligned_full"
    )
    assert np.array_equal(strict_full, touched_full)
    assert int(strict_full.sum()) == 144


# ---------------------------------------------------------------------------
# 4. The ring, where the boundary crosses pixels: exactly one pixel wide.
# ---------------------------------------------------------------------------
RING_ROWS = 26
RING_COLS = 24
# Left edge 3.9 (inside column 3, centre 3.5 outside => ring column 3), right
# edge 19.2 (inside column 19, centre 19.5 outside => ring column 19), bottom
# edge 7.6 (ring row 18, centre 7.5 outside), top edge 21.4 (ring row 4, centre
# 21.5 outside).
RING_GEOMETRY = box(3.9, 7.6, 19.2, 21.4)
RING_START = (6, 6)
RING_END = (16, 16)


def test_ring_appears_only_where_the_boundary_crosses_pixels(tmp_path):
    strict, touched, _ = _masks(
        tmp_path, RING_GEOMETRY, RING_ROWS, RING_COLS, "ring_shape"
    )
    ring = touched & ~strict

    assert int(strict.sum()) == 195  # 13 rows x 15 columns of centres inside
    assert int(touched.sum()) == 255  # 15 rows x 17 columns overlap the polygon
    assert int(ring.sum()) == 60
    assert sorted({int(r) for r in np.where(strict)[0]}) == list(range(5, 18))
    assert sorted({int(c) for c in np.where(strict)[1]}) == list(range(4, 19))

    # The ring is exactly one pixel wide on the axis-aligned sides: the first
    # pixel outside the strict mask is in the ring, the next one outward is not
    # navigable at all.
    assert ring[4, 10] and not touched[3, 10]
    assert ring[18, 10] and not touched[19, 10]
    assert ring[10, 3] and not touched[10, 2]
    assert ring[10, 19] and not touched[10, 20]

    # Per-side ring counts: the four sides with the corners counted once.
    assert int(ring[4, :].sum()) == 17
    assert int(ring[18, :].sum()) == 17
    assert int(ring[:, 3].sum()) == 15
    assert int(ring[:, 19].sum()) == 15
    assert ring[4, 3] and ring[18, 19]  # the four corners are in the ring


# ---------------------------------------------------------------------------
# 5. A diagonal boundary does not widen the ring; a shallow one does.
# ---------------------------------------------------------------------------
DIAG_ROWS = 24
DIAG_COLS = 24
# Hypotenuse x + y = 24.3: at 45 degrees and 0.3 off the pixel corners, so no
# pixel centre lies exactly on it.
DIAG_TRIANGLE = Polygon([(1.3, 1.3), (23.0, 1.3), (1.3, 23.0)])
# Hypotenuse from (22.7, 1.3) to (1.3, 12.0): slope 1:2, i.e. 26.6 degrees.
SHALLOW_TRIANGLE = Polygon([(1.3, 1.3), (22.7, 1.3), (1.3, 12.0)])


def test_diagonal_boundaries_can_widen_the_ring(tmp_path):
    # Measured thickness at exactly 45 degrees is ONE pixel: a horizontal scan
    # line crosses the boundary in a single ring cell. The ring is the staircase
    # c = r + 1, whose consecutive cells share only a corner, so the ring is
    # 8-connected but not 4-connected at all (0 four-adjacent pairs) -- MCP
    # still walks it because run_mcp builds MCP_Geometric(fully_connected=True).
    # Mechanism: at 45 degrees the boundary steps corner to corner, so each
    # pixel it grazes is a single cell, and the two cells that meet at a pixel
    # corner are diagonal neighbours, not side neighbours.
    strict, touched, _ = _masks(tmp_path, DIAG_TRIANGLE, DIAG_ROWS, DIAG_COLS, "diag45")
    ring = touched & ~strict
    assert int(strict.sum()) == 253
    assert int(touched.sum()) == 274
    assert int(ring.sum()) == 21
    assert {tuple(int(v) for v in cell) for cell in zip(*np.where(ring))} == {
        (row, row + 1) for row in range(1, 22)
    }
    assert _max_row_run(ring) == 1
    corners_only = 0
    for row, col in zip(*np.where(ring)):
        for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            corners_only += int(bool(ring[row + d_row, col + d_col]))
    assert corners_only == 0

    # Widen the mechanism: as soon as the boundary is not exactly 45 degrees, a
    # horizontal scan crosses two ring cells, because the boundary enters a cell
    # on one side and leaves it through the neighbouring cell of the same row.
    shallow_strict, shallow_touched, _ = _masks(
        tmp_path, SHALLOW_TRIANGLE, DIAG_ROWS, DIAG_COLS, "diag_shallow"
    )
    shallow_ring = shallow_touched & ~shallow_strict
    assert int(shallow_strict.sum()) == 121
    assert int(shallow_touched.sum()) == 142
    assert int(shallow_ring.sum()) == 21
    assert _max_row_run(shallow_ring) == 2
    # Rows 12..21 hold the two-cell pairs, row 22 only the closing corner cell.
    assert [int(n) for n in shallow_ring.sum(axis=1) if n] == [2] * 10 + [1]


# ---------------------------------------------------------------------------
# 6. Cost monotonicity of the wider mask.
# ---------------------------------------------------------------------------
def test_routes_never_become_more_expensive_with_the_touched_mask(tmp_path):
    strict, touched, transform = _masks(
        tmp_path, RING_GEOMETRY, RING_ROWS, RING_COLS, "ring_costs"
    )
    ring = touched & ~strict

    # Measured (strict cost, touched cost) for interior 50, 10, 3 and 1.5 with
    # the exterior at 1.0.
    expected = {
        50.0: (707.1067811865476, 179.82842712474618),
        10.0: (141.4213562373095, 59.828427124746185),
        3.0: (42.426406871192846, 37.071067811865476),
        1.5: (21.213203435596423, 21.213203435596423),
    }
    for interior_cost in (50.0, 10.0, 3.0, 1.5):
        costs = cheap_outside_polygon(
            RING_ROWS, RING_COLS, transform, RING_GEOMETRY, interior_cost, 1.0
        )
        strict_path, _ = run_mcp(np.where(strict, costs, np.inf), RING_START, RING_END)
        touched_path, _ = run_mcp(
            np.where(touched, costs, np.inf), RING_START, RING_END
        )
        strict_cost = path_total_cost(costs, strict_path)
        touched_cost = path_total_cost(costs, touched_path)

        assert strict_cost == pytest.approx(expected[interior_cost][0])
        assert touched_cost == pytest.approx(expected[interior_cost][1])
        # The wider mask can only add navigable cells, never remove them: the
        # cheapest route over a superset is never more expensive.
        assert touched_cost <= strict_cost
        # And the strict route never leaves the strict mask, hence never uses
        # the ring, while the touched route may.
        assert int(ring[strict_path[:, 0], strict_path[:, 1]].sum()) == 0

    # Reachability is monotone in the same direction: because strict is a subset
    # of touched, every pair connected under the strict mask is also connected
    # under the touched mask. Both runs above found a route for every ratio.
    assert not (strict & ~touched).any()


# ---------------------------------------------------------------------------
# 7. The ring only becomes attractive past a cost ratio.
# ---------------------------------------------------------------------------
def test_ring_only_becomes_attractive_beyond_a_cost_ratio(tmp_path):
    strict, touched, transform = _masks(
        tmp_path, RING_GEOMETRY, RING_ROWS, RING_COLS, "ring_threshold"
    )
    ring = touched & ~strict

    # Measured with the exterior fixed at 1.0: the touched route ignores the
    # ring at interior 1.0, 1.5 and 2.0 and starts using it at 2.5, so the
    # threshold for this start/end pair lies between 2.0 and 2.5.
    expected_enters = {
        50.0: True,
        10.0: True,
        3.0: True,
        2.5: True,
        2.0: False,
        1.5: False,
    }
    for interior_cost, enters in expected_enters.items():
        costs = cheap_outside_polygon(
            RING_ROWS, RING_COLS, transform, RING_GEOMETRY, interior_cost, 1.0
        )
        strict_path, _ = run_mcp(np.where(strict, costs, np.inf), RING_START, RING_END)
        touched_path, _ = run_mcp(
            np.where(touched, costs, np.inf), RING_START, RING_END
        )
        ring_vertices = int(ring[touched_path[:, 0], touched_path[:, 1]].sum())

        assert (ring_vertices > 0) is enters
        if enters:
            assert not np.array_equal(strict_path, touched_path)
        else:
            # Not worth it: the cheap-interior run reproduces the strict route
            # vertex for vertex.
            assert np.array_equal(strict_path, touched_path)

    # The largest measured saving: 707.11 for the strict route against 179.83
    # for the ring-corridor route, i.e. a ratio of 0.254.
    costs = cheap_outside_polygon(
        RING_ROWS, RING_COLS, transform, RING_GEOMETRY, 50.0, 1.0
    )
    strict_path, _ = run_mcp(np.where(strict, costs, np.inf), RING_START, RING_END)
    touched_path, _ = run_mcp(np.where(touched, costs, np.inf), RING_START, RING_END)
    ratio = path_total_cost(costs, touched_path) / path_total_cost(costs, strict_path)
    assert ratio == pytest.approx(0.2543, abs=1e-4)


# ---------------------------------------------------------------------------
# 8. The only functional justification: a point on the polygon boundary.
# ---------------------------------------------------------------------------
def test_boundary_point_is_the_only_practical_use_of_the_parameter(tmp_path):
    strict, touched, _ = _masks(
        tmp_path, RING_GEOMETRY, RING_ROWS, RING_COLS, "ring_point"
    )

    # World (3.4, 15.0) lands in pixel (10, 3): the pixel spans x in [3, 4], so
    # the point is inside it, but the pixel CENTRE (3.5) is outside the polygon
    # (which starts at x = 3.9). With the strict mask the pixel is not
    # navigable; with all_touched=True the polygon merely grazes the pixel and
    # it becomes navigable.
    assert not strict[10, 3]
    assert touched[10, 3]
    # The neighbouring pixels show this is the ring and nothing wider.
    assert strict[10, 4]
    assert not strict[10, 2] and not touched[10, 2]

    costs = np.full((RING_ROWS, RING_COLS), 1.0)
    source = (14, 10)
    with pytest.raises(AssertionError, match="no route"):
        run_mcp(np.where(strict, costs, np.inf), source, (10, 3))

    path, _ = run_mcp(np.where(touched, costs, np.inf), source, (10, 3))
    assert [int(v) for v in path[0]] == [14, 10]
    assert [int(v) for v in path[-1]] == [10, 3]
    assert int(path.shape[0]) == 8

    # This is the sole functional justification for the argument: admitting a
    # legitimate point that sits on (or a fraction of a pixel outside) the study
    # boundary. It is narrower to admit just that point's pixel than to make the
    # whole boundary ring navigable, which is what reopens the border corridor.
