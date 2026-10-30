"""Synthetic cost surfaces and path helpers for the border-corridor tests.

Everything here is deterministic (no RNG) and stays on small rasters, so the
tests can assert observable path properties -- vertex coordinates and total
cost -- instead of inspecting internal state.
"""

from itertools import pairwise

import fiona
import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from shapely.geometry import mapping
from skimage.graph import MCP_Geometric

import lcp.pathfinder as pf

CRS = "EPSG:32719"


class StubRaster:
    """Stand-in for a rasterio dataset: only what the mask builders read."""

    def __init__(self, crs, shape, transform):
        self.crs = crs
        self.shape = shape
        self.transform = transform


def build_transform(rows, cols, scale=1.0):
    """North-up transform where pixel (r, c) spans one scale-wide world cell.

    Pixel (r, c) covers x in [c*scale, (c+1)*scale] and y in
    [(rows-r-1)*scale, (rows-r)*scale], so its centre is at
    (c*scale + scale/2, (rows-r-0.5)*scale).
    """
    rows = int(rows)
    scale = float(scale)
    return Affine(scale, 0.0, 0.0, 0.0, -scale, rows * scale)


def cheap_edge_frame(rows, cols, frame_px, interior_cost, frame_cost):
    """Cost surface with a cheap frame of ``frame_px`` outer rows and columns."""
    rows, cols, frame_px = int(rows), int(cols), int(frame_px)
    if frame_px < 0:
        raise ValueError(f"frame_px must not be negative: {frame_px}")
    if 2 * frame_px >= rows or 2 * frame_px >= cols:
        raise ValueError(
            f"frame_px={frame_px} leaves no interior in a {rows}x{cols} raster"
        )

    costs = np.full((rows, cols), float(interior_cost), dtype=np.float64)
    costs[:frame_px, :] = float(frame_cost)
    costs[rows - frame_px :, :] = float(frame_cost)
    costs[:, :frame_px] = float(frame_cost)
    costs[:, cols - frame_px :] = float(frame_cost)
    return costs


def cheap_outside_polygon(rows, cols, transform, polygon, inside_cost, outside_cost):
    """Cost surface cheap outside and expensive inside ``polygon``.

    The value follows the pixel CENTRE (``all_touched=False``), matching the
    strict vector mask so a polygon pixel is expensive in both.
    """
    inside = geometry_mask(
        [mapping(polygon)],
        out_shape=(int(rows), int(cols)),
        transform=transform,
        all_touched=False,
        invert=True,
    )
    costs = np.full((int(rows), int(cols)), float(outside_cost), dtype=np.float64)
    costs[inside] = float(inside_cost)
    return costs


def run_mcp(costs, start_pixel, end_pixel):
    """Run MCP_Geometric over ``costs`` and return ``(path, offsets)``.

    ``offsets`` are the real MCP offsets, as ``pathfinder`` stores them for
    Phase 2. Raises AssertionError if no route exists.
    """
    mcp = MCP_Geometric(costs, fully_connected=True)
    _, traceback = mcp.find_costs([start_pixel], ends=[end_pixel], find_all_ends=True)
    offsets = np.asarray(mcp.offsets, dtype=np.int32)
    path = pf.reconstruct_path_from_traceback(
        traceback, start_pixel, end_pixel, offsets
    )
    assert path is not None, f"no route from {start_pixel} to {end_pixel}"
    return path, offsets


def path_total_cost(costs, path):
    """Total MCP_Geometric cost of ``path``: step length times mean cell cost."""
    total = 0.0
    for (row0, col0), (row1, col1) in pairwise(path):
        step = float(np.hypot(row1 - row0, col1 - col0))
        total += step * 0.5 * (float(costs[row0, col0]) + float(costs[row1, col1]))
    return total


def write_polygon(path, geometry, crs=CRS):
    """Write a single-feature ESRI Shapefile holding ``geometry``."""
    schema = {"geometry": "Polygon", "properties": {}}
    with fiona.open(path, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as dst:
        dst.write({"geometry": mapping(geometry), "properties": {}})
