"""
Prueba definitiva: genera heatmap desde UN SOLO punto de origen
y verifica que el pico del KDE caiga exactamente en esa coordenada.
"""
import os, sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.fft import rfft2, irfft2
import fiona

HEATMAP_RES_M = 30
HEATMAP_BANDWIDTH_M = 10000
HEATMAP_KERNEL = "quartic"

ORIGIN_PATH = r"C:\Users\LAV\Desktop\Proyecto Vannia\Proceso\coste\puntos del perimetro del poligono.gpkg"
outdir = os.path.join(os.getcwd(), "output")
sessions = sorted([d for d in os.listdir(outdir) if d.startswith("session_")],
                  key=lambda s: os.path.getmtime(os.path.join(outdir, s)), reverse=True)
sess = os.path.join(outdir, sessions[0])

# Tomar el PRIMER punto de origen
print("Leyendo primer punto de origen...")
with fiona.open(ORIGIN_PATH) as src:
    feat = next(iter(src))
    geom = feat["geometry"]
    x0, y0 = geom["coordinates"]
print(f"  Punto: ({x0:.4f}, {y0:.4f})")

# Crear un count_grid con UN UNICO punto
margin = HEATMAP_BANDWIDTH_M
minx = x0 - margin
miny = y0 - margin
maxx = x0 + margin
maxy = y0 + margin

width  = int(np.ceil((maxx - minx) / HEATMAP_RES_M))
height = int(np.ceil((maxy - miny) / HEATMAP_RES_M))
pixel_size_x = (maxx - minx) / width
pixel_size_y = (maxy - miny) / height
transform = from_bounds(minx, miny, maxx, maxy, width, height)

print(f"  Grid: {width} x {height}")
print(f"  Pixel real: {pixel_size_x:.10f} x {pixel_size_y:.10f}")

# Binning: un solo punto
col = int((x0 - minx) / pixel_size_x)
row = int((maxy - y0) / pixel_size_y)

# Centro del pixel donde cae el punto
pixel_cx = minx + (col + 0.5) * pixel_size_x
pixel_cy = maxy - (row + 0.5) * pixel_size_y
print(f"\n  Punto -> pixel (fila={row}, col={col})")
print(f"  Centro del pixel: ({pixel_cx:.4f}, {pixel_cy:.4f})")
print(f"  Error de cuantizacion: ({x0-pixel_cx:.4f}, {y0-pixel_cy:.4f}) m")

# Grid con un solo punto
count_grid = np.zeros((height, width), dtype=np.float64)
count_grid[row, col] = 1.0

# Kernel
radius_px = int(np.ceil(HEATMAP_BANDWIDTH_M / HEATMAP_RES_M))
y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
dist_px = np.sqrt(x**2 + y**2)
# Usar pixel_size para que el kernel use la escala espacial CORRECTA
pixel_mean = (pixel_size_x + pixel_size_y) / 2.0
u = (dist_px * pixel_mean) / HEATMAP_BANDWIDTH_M

if HEATMAP_KERNEL == "quartic":
    kernel = np.where(u <= 1, (15.0 / 16.0) * (1 - u**2)**2, 0.0)

kernel_area = np.sum(kernel) * (pixel_mean ** 2)
kernel = kernel / kernel_area

# FFT convolution
count_padded = np.pad(count_grid, pad_width=radius_px, mode='constant')
kernel_padded = np.zeros_like(count_padded)
kh, kw = kernel.shape
kernel_padded[:kh, :kw] = kernel

fft_grid = rfft2(count_padded)
fft_kernel = rfft2(kernel_padded)
fft_result = fft_grid * fft_kernel
heatmap = irfft2(fft_result, s=count_padded.shape)
heatmap = heatmap[radius_px:radius_px+height, radius_px:radius_px+width]

# Encontrar el pico del KDE
peak_row, peak_col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
peak_x = minx + (peak_col + 0.5) * pixel_size_x
peak_y = maxy - (peak_row + 0.5) * pixel_size_y
print(f"\n  Pico del KDE: pixel (fila={peak_row}, col={peak_col})")
print(f"  Coordenadas del pico: ({peak_x:.4f}, {peak_y:.4f})")
print(f"  Punto original:       ({x0:.4f}, {y0:.4f})")
print(f"  Diferencia (pico - punto): ({peak_x-x0:.4f}, {peak_y-y0:.4f}) m")

if abs(peak_x - x0) < pixel_size_x and abs(peak_y - y0) < pixel_size_y:
    print(f"\n  *** VERIFICACION OK: el pico del KDE esta dentro del pixel correcto ***")
else:
    print(f"\n  *** ALERTA: el pico del KDE NO coincide con el punto! ***")

# Guardar TIF
out_tif = os.path.join(sess, "_diag_single_point.tif")
heatmap_display = heatmap / heatmap.max()  # normalizar para visualizar
with rasterio.open(
    out_tif, 'w', driver='GTiff', height=height, width=width, count=1,
    dtype=heatmap_display.dtype, crs='EPSG:32719', transform=transform, compress='lzw'
) as dst:
    dst.write(heatmap_display, 1)
print(f"\n  TIF guardado: {out_tif}")

print(f"\n=== INSTRUCCIONES ===")
print(f"1. Abre {out_tif} en QGIS")
print(f"2. Abre el punto de origen")
print(f"3. El pico del KDE (maximo) DEBE coincidir con la ubicacion del punto")
print(f"4. Si no coincide, el error es {peak_x-x0:.2f}m X, {peak_y-y0:.2f}m Y")
