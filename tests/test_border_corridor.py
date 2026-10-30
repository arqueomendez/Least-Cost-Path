"""Proofs that least-cost routes do not drift to the study-area border.

Two border corridors are covered:

- the raster edge frame, which ``edge_frame_mask`` / ``EDGE_BUFFER_PX`` excludes;
- the ring of extra pixels an ``all_touched=True`` vector mask adds around the
  study-area polygon, which the strict (default) mask keeps out.

Every assertion is an observable property of the resulting path -- vertex
coordinates and/or total cost -- never internal state.
"""

import numpy as np
from shapely.geometry import box
from synthetic_surfaces import (
    CRS,
    StubRaster,
    build_transform,
    cheap_edge_frame,
    cheap_outside_polygon,
    path_total_cost,
    run_mcp,
    write_polygon,
)

import lcp.processing as proc

# ---------------------------------------------------------------------------
# Raster edge frame: 41x41, a 2-pixel frame 100x cheaper than the interior.
# ---------------------------------------------------------------------------
EDGE_ROWS = 41
EDGE_COLS = 41
FRAME_PX = 2
INTERIOR_COST = 100.0
FRAME_COST = 1.0
EDGE_START = (3, 20)
EDGE_END = (37, 20)


def _edge_frame_costs():
    return cheap_edge_frame(EDGE_ROWS, EDGE_COLS, FRAME_PX, INTERIOR_COST, FRAME_COST)


def test_raster_edge_frame_attracts_routes_when_not_excluded():
    # With no edge exclusion the frame is 100x cheaper than the interior, so the
    # optimal route abandons the straight line (34 interior steps, 3400.0) and
    # detours around the raster edge. This test documents the phenomenon that
    # EDGE_BUFFER_PX exists to prevent.
    costs = _edge_frame_costs()
    frame = proc.edge_frame_mask(costs.shape, FRAME_PX)

    path, _ = run_mcp(costs, EDGE_START, EDGE_END)

    # Same condition as the explicit form: row < 2 or row > 38 or col < 2 or
    # col > 38.
    on_frame = [
        tuple(int(v) for v in pixel) for pixel in path if frame[pixel[0], pixel[1]]
    ]
    assert on_frame, "expected the route to use the cheap edge frame"

    straight_cost = (EDGE_END[0] - EDGE_START[0]) * INTERIOR_COST
    assert path_total_cost(costs, path) < straight_cost


def test_edge_frame_mask_keeps_routes_off_the_raster_edge():
    costs = _edge_frame_costs()
    free_path, _ = run_mcp(costs, EDGE_START, EDGE_END)

    # Exactly what the notebook does with main_search_mask & ~edge_frame: the
    # excluded frame becomes impassable before MCP is built.
    frame = proc.edge_frame_mask(costs.shape, FRAME_PX)
    masked = np.where(frame, np.inf, costs)
    path, _ = run_mcp(masked, EDGE_START, EDGE_END)

    assert path.size > 0, "exclusion must not disconnect the endpoints"
    assert not [pixel for pixel in path if frame[pixel[0], pixel[1]]]
    # The cheap detour is gone: the route now pays the interior cost end to end.
    assert path_total_cost(masked, path) > path_total_cost(costs, free_path)


# ---------------------------------------------------------------------------
# Vector mask ring: 21x21, expensive inside the polygon, cheap outside.
# Pixel centres sit at (c + 0.5, rows - r - 0.5), so these bounds enclose the
# centres of rows/cols 6..15 and merely touch rows/cols 5 and 16. The bounds are
# kept off the pixel centres on purpose: a centre exactly on the polygon
# boundary is included asymmetrically by GDAL rasterisation, which would make
# the ring depend on a floating-point knife edge instead of on the geometry.
# ---------------------------------------------------------------------------
BOX_ROWS = 21
BOX_COLS = 21
STUDY_AREA = box(5.9, 4.9, 16.1, 15.1)
INSIDE_COST = 50.0
OUTSIDE_COST = 1.0
BOX_START = (7, 7)
BOX_END = (14, 14)


def _study_area(tmp_path):
    """Write the polygon and return ``(vector_path, stub, costs)``."""
    vector_path = tmp_path / "area_estudio.shp"
    write_polygon(vector_path, STUDY_AREA)
    transform = build_transform(BOX_ROWS, BOX_COLS)
    stub = StubRaster(CRS, (BOX_ROWS, BOX_COLS), transform)
    costs = cheap_outside_polygon(
        BOX_ROWS, BOX_COLS, transform, STUDY_AREA, INSIDE_COST, OUTSIDE_COST
    )
    return str(vector_path), stub, costs


def test_all_touched_mask_creates_a_border_corridor(tmp_path):
    vector_path, stub, costs = _study_area(tmp_path)
    touched = proc.create_mask_from_vector(vector_path, stub, all_touched=True)
    strict = proc.create_mask_from_vector(vector_path, stub)
    ring = touched & ~strict
    assert ring.any(), "the touched mask must add a ring over the strict one"

    path, _ = run_mcp(np.where(touched, costs, np.inf), BOX_START, BOX_END)

    in_ring = [
        tuple(int(v) for v in pixel) for pixel in path if ring[pixel[0], pixel[1]]
    ]
    assert in_ring, "expected the route to hug the cheap touched ring"
    # Inside the mask but outside the strict mask, i.e. outside the polygon: the
    # route leaves the interior through column 5 and returns along row 16.
    assert 5 in {col for _, col in in_ring}
    assert 16 in {row for row, _ in in_ring}

    # The corridor is a genuine shortcut: cheaper than crossing the expensive
    # interior diagonally, which is what the strict mask forces.
    interior_traverse = (
        float(np.hypot(BOX_END[0] - BOX_START[0], BOX_END[1] - BOX_START[1]))
        * INSIDE_COST
    )
    assert path_total_cost(np.where(touched, costs, np.inf), path) < interior_traverse


def test_strict_mask_keeps_routes_inside_the_polygon(tmp_path):
    vector_path, stub, costs = _study_area(tmp_path)
    # Default call on purpose, without all_touched: this test fails if the
    # default ever regresses to True, because the route would leave the polygon.
    strict = proc.create_mask_from_vector(vector_path, stub)
    touched = proc.create_mask_from_vector(vector_path, stub, all_touched=True)

    path, _ = run_mcp(np.where(strict, costs, np.inf), BOX_START, BOX_END)

    rows = path[:, 0]
    cols = path[:, 1]
    assert rows.min() >= 6 and rows.max() <= 15
    assert cols.min() >= 6 and cols.max() <= 15

    ring = touched & ~strict
    assert not [pixel for pixel in path if ring[pixel[0], pixel[1]]]


def test_strict_and_touched_masks_differ_on_the_ring(tmp_path):
    vector_path, stub, _ = _study_area(tmp_path)
    strict = proc.create_mask_from_vector(vector_path, stub)
    touched = proc.create_mask_from_vector(vector_path, stub, all_touched=True)

    # strict is a subset of touched, and touched only adds the 1-pixel ring.
    assert int((strict & ~touched).sum()) == 0
    assert int(strict.sum()) == 100  # 10 x 10 centres inside: rows/cols 6..15
    assert int(touched.sum()) == 144  # 12 x 12: the same plus the ring 5 and 16
    assert int(strict.sum()) < int(touched.sum())

    ring = touched & ~strict
    assert int(ring.sum()) == 44  # 12*12 - 10*10

    # Row 5 (centre y = 15.5) and row 16 (centre y = 4.5) hold no centre inside
    # the polygon, yet the polygon crosses those pixels, so only all_touched
    # includes them. Same for columns 5 and 16 (centres x = 5.5 and 16.5).
    assert touched[5, 10] and not strict[5, 10]
    assert touched[16, 10] and not strict[16, 10]
    assert touched[10, 5] and not strict[10, 5]
    assert touched[10, 16] and not strict[10, 16]
    # The ring is exactly one pixel wide: row 4 is never touched.
    assert not touched[4, 10]
    assert not touched[17, 10]
