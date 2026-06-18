"""
Generar heatmap KDE desde puntos densificados (GeoPackage).

Uso:
  uv run python generar_heatmap.py <path_al_gpkg> [--res 30] [--bw 10000] [--kernel quartic]

Ejemplos:
  uv run python generar_heatmap.py output/session_20260415_213517_Refactored/puntos_rutas_25m.gpkg
  uv run python generar_heatmap.py output/session_20260415_213517_Refactored/puntos_rutas_25m.gpkg --res 50 --bw 5000 --kernel gaussian
"""
import os, sys, time, argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from shapely import get_coordinates
from scipy.fft import rfft2, irfft2

parser = argparse.ArgumentParser(description="Generar heatmap KDE desde puntos densificados")
parser.add_argument("points_path", help="Ruta al GeoPackage de puntos densificados")
parser.add_argument("--res", type=float, default=30, help="Resolucion en metros (default: 30)")
parser.add_argument("--bw", type=float, default=10000, help="Bandwidth KDE en metros (default: 10000)")
parser.add_argument("--kernel", default="quartic", choices=["quartic","gaussian","triangular","uniform"],
                    help="Tipo de kernel (default: quartic)")
args = parser.parse_args()

POINTS_PATH = args.points_path
OUTPUT_DIR  = os.path.dirname(POINTS_PATH)
HEATMAP_RES_M = args.res
HEATMAP_BANDWIDTH_M = args.bw
HEATMAP_KERNEL = args.kernel

print(f"Cargando puntos: {POINTS_PATH}")
points_gdf = gpd.read_file(POINTS_PATH)
crs = points_gdf.crs
print(f"  CRS: {crs}  |  Puntos: {len(points_gdf):,}")

# Extent con margen = bandwidth
minx, miny, maxx, maxy = points_gdf.total_bounds
margin = HEATMAP_BANDWIDTH_M
minx -= margin; miny -= margin; maxx += margin; maxy += margin

width  = int(np.ceil((maxx - minx) / HEATMAP_RES_M))
height = int(np.ceil((maxy - miny) / HEATMAP_RES_M))
pixel_size_x = (maxx - minx) / width
pixel_size_y = (maxy - miny) / height
transform = from_bounds(minx, miny, maxx, maxy, width, height)

print(f"  Grid: {width}x{height} pix")
print(f"  Pixel real: {pixel_size_x:.6f} x {pixel_size_y:.6f} m")

# Binning con pixel_size REAL (NO HEATMAP_RES_M)
print("Binning...")
coords = get_coordinates(points_gdf.geometry)
cols = ((coords[:, 0] - minx) / pixel_size_x).astype(np.int32)
rows = ((maxy - coords[:, 1]) / pixel_size_y).astype(np.int32)
valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
count_grid = np.zeros((height, width), dtype=np.float64)
np.add.at(count_grid, (rows[valid], cols[valid]), 1)
print(f"  Puntos binnados: {valid.sum():,} / {len(coords):,}")

# Kernel (usando pixel_size real para coincidencia exacta con el grid)
ps = (pixel_size_x + pixel_size_y) / 2.0
radius_px = int(np.ceil(HEATMAP_BANDWIDTH_M / ps))
y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
u = np.sqrt(x**2 + y**2) * ps / HEATMAP_BANDWIDTH_M
if HEATMAP_KERNEL == "quartic":
    kernel = np.where(u <= 1, (15.0/16.0)*(1-u**2)**2, 0.0)
elif HEATMAP_KERNEL == "gaussian":
    kernel = np.exp(-0.5*u**2)
elif HEATMAP_KERNEL == "triangular":
    kernel = np.where(u <= 1, 1-u, 0.0)
elif HEATMAP_KERNEL == "uniform":
    kernel = np.where(u <= 1, 1.0, 0.0)
else:
    raise ValueError(f"Kernel: {HEATMAP_KERNEL}")
kernel = kernel / (np.sum(kernel) * ps**2)
print(f"  Kernel: {kernel.shape} px, bandwidth={HEATMAP_BANDWIDTH_M/1000:.0f}km")

# FFT
print("Convolucion FFT...")
count_padded = np.pad(count_grid, pad_width=radius_px, mode='constant')
kp = np.zeros_like(count_padded)
kp[:kernel.shape[0], :kernel.shape[1]] = kernel
heatmap = irfft2(rfft2(count_padded) * rfft2(kp), s=count_padded.shape)
heatmap = heatmap[2*radius_px:2*radius_px+height, 2*radius_px:2*radius_px+width]
print(f"  Densidad: {heatmap.min():.6f} a {heatmap.max():.6f}")

# Guardar GeoTIFF
tif_path = os.path.join(OUTPUT_DIR,
    f'heatmap_{HEATMAP_RES_M}m_{int(HEATMAP_BANDWIDTH_M/1000)}km_{HEATMAP_KERNEL}.tif')
with rasterio.open(tif_path, 'w', driver='GTiff',
    height=height, width=width, count=1, dtype=heatmap.dtype,
    crs=crs, transform=transform, compress='lzw') as dst:
    dst.write(heatmap, 1)
print(f"GeoTIFF: {tif_path}")

# Verificar
with rasterio.open(tif_path) as src:
    print(f"\nVERIFICACION:")
    print(f"  Bounds: {src.bounds}")
    print(f"  Transform: {src.transform}")
    print(f"  CRS: {src.crs}")
print("\nLISTO.")
