"""
Diagnostico final: genera un heatmap MINI desde los primeros 5000 puntos
y lo compara con un scatter de los mismos puntos para ver el shift.
"""
import os, sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.fft import rfft2, irfft2
from shapely import get_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyogrio

# Parametros fijos
HEATMAP_RES_M = 30
HEATMAP_BANDWIDTH_M = 10000
HEATMAP_KERNEL = "quartic"

outdir = os.path.join(os.getcwd(), "output")
sessions = sorted([d for d in os.listdir(outdir) if d.startswith("session_")],
                  key=lambda s: os.path.getmtime(os.path.join(outdir, s)), reverse=True)
if not sessions:
    print("ERROR: no hay sesiones")
    sys.exit(1)
sess = os.path.join(outdir, sessions[0])

pp = os.path.join(sess, "puntos_rutas_25m.gpkg")
if not os.path.exists(pp):
    print(f"ERROR: no existe {pp}")
    sys.exit(1)

# Tomar solo una muestra MINIMA (5000 puntos) para que sea rapido
# Usar pyogrio con skip_features para leer solo los primeros
import fiona

# Leer solo primeros 5000 puntos
print("Leyendo 5000 puntos de muestra...")
pts_list = []
with fiona.open(pp) as src:
    for i, feat in enumerate(src):
        if i >= 5000:
            break
        geom = feat["geometry"]
        if geom and geom["type"] == "Point":
            pts_list.append(geom["coordinates"])
        elif geom and geom["type"] == "MultiPoint":
            for c in geom["coordinates"]:
                pts_list.append(c)

coords = np.array(pts_list)
print(f"  {len(coords)} puntos cargados")
print(f"  Rango X: [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]")
print(f"  Rango Y: [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")

# --- Misma logica que Celda 9 del notebook (con fix) ---
minx, miny = coords[:, 0].min(), coords[:, 1].min()
maxx, maxy = coords[:, 0].max(), coords[:, 1].max()
margin = HEATMAP_BANDWIDTH_M
minx -= margin
miny -= margin
maxx += margin
maxy += margin

width  = int(np.ceil((maxx - minx) / HEATMAP_RES_M))
height = int(np.ceil((maxy - miny) / HEATMAP_RES_M))
pixel_size_x = (maxx - minx) / width
pixel_size_y = (maxy - miny) / height
transform = from_bounds(minx, miny, maxx, maxy, width, height)

print(f"\n  Grid: {width} x {height}")
print(f"  Pixel real: {pixel_size_x:.6f} x {pixel_size_y:.6f}")
print(f"  HEATMAP_RES_M: {HEATMAP_RES_M}")

# Binning
cols = ((coords[:, 0] - minx) / pixel_size_x).astype(np.int32)
rows = ((maxy - coords[:, 1]) / pixel_size_y).astype(np.int32)
valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
print(f"  Puntos validos: {valid.sum()} / {len(valid)}")

count_grid = np.zeros((height, width), dtype=np.float64)
np.add.at(count_grid, (rows[valid], cols[valid]), 1)

# --- Verificacion de posicion: donde esta el punto central? ---
# Tomar el punto mas cercano al centroide
centroid_x = (coords[:, 0].min() + coords[:, 0].max()) / 2
centroid_y = (coords[:, 1].min() + coords[:, 1].max()) / 2
dists = (coords[:, 0] - centroid_x)**2 + (coords[:, 1] - centroid_y)**2
mid_idx = np.argmin(dists)
mid_x, mid_y = coords[mid_idx]
mid_col = int((mid_x - minx) / pixel_size_x)
mid_row = int((maxy - mid_y) / pixel_size_y)
mid_pixel_x = minx + (mid_col + 0.5) * pixel_size_x
mid_pixel_y = maxy - (mid_row + 0.5) * pixel_size_y
print(f"\n  Punto central ({mid_x:.2f}, {mid_y:.2f}) -> pixel ({mid_row}, {mid_col})")
print(f"  Centro del pixel: ({mid_pixel_x:.2f}, {mid_pixel_y:.2f})")
print(f"  Error de cuantizacion: ({mid_x-mid_pixel_x:.2f}, {mid_y-mid_pixel_y:.2f}) m")

# --- Chequeo de shift sistematico ---
print(f"\n  Chequeo de shift sistematico (promediando todos los puntos):")
errors_x = (coords[valid, 0] - (minx + (cols[valid] + 0.5) * pixel_size_x))
errors_y = (coords[valid, 1] - (maxy - (rows[valid] + 0.5) * pixel_size_y))
print(f"    Error X medio: {errors_x.mean():.4f} m (std: {errors_x.std():.4f})")
print(f"    Error Y medio: {errors_y.mean():.4f} m (std: {errors_y.std():.4f})")
if abs(errors_x.mean()) < 1 and abs(errors_y.mean()) < 1:
    print(f"    *** NO HAY SHIFT SISTEMATICO en el binning")
else:
    print(f"    *** ALERTA: Hay shift sistematico de ~{errors_x.mean():.1f}m X, {errors_y.mean():.1f}m Y")

# --- Construir kernel y convolucion ---
radius_px = int(np.ceil(HEATMAP_BANDWIDTH_M / HEATMAP_RES_M))
y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
dist_px = np.sqrt(x**2 + y**2)
u = (dist_px * HEATMAP_RES_M) / HEATMAP_BANDWIDTH_M

if HEATMAP_KERNEL == "quartic":
    kernel = np.where(u <= 1, (15.0 / 16.0) * (1 - u**2)**2, 0.0)

kernel_area = np.sum(kernel) * (HEATMAP_RES_M ** 2)
kernel = kernel / kernel_area

count_padded = np.pad(count_grid, pad_width=radius_px, mode='constant')
kernel_padded = np.zeros_like(count_padded)
kh, kw = kernel.shape
kernel_padded[:kh, :kw] = kernel

fft_grid = rfft2(count_padded)
fft_kernel = rfft2(kernel_padded)
fft_result = fft_grid * fft_kernel
heatmap = irfft2(fft_result, s=count_padded.shape)
heatmap = heatmap[radius_px:radius_px+height, radius_px:radius_px+width]

# --- Guardar TIF y PNG ---
out_tif = os.path.join(sess, "_diag_heatmap_check.tif")
with rasterio.open(
    out_tif, 'w', driver='GTiff',
    height=height, width=width, count=1,
    dtype=heatmap.dtype, crs='EPSG:32719', transform=transform,
    compress='lzw'
) as dst:
    dst.write(heatmap, 1)

print(f"\n  TIF guardado: {out_tif}")

# --- Visualizacion: scatter de puntos sobre heatmap ---
fig, ax = plt.subplots(figsize=(16, 14))
extent = [minx, maxx, miny, maxy]

vmin = heatmap[heatmap > 0].min() if heatmap.max() > 0 else 0
ax.imshow(heatmap, extent=extent, origin='upper', cmap='hot',
          norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=heatmap.max()))

# Sample de 200 puntos para el scatter
np.random.seed(42)
sample_idx = np.random.choice(len(coords), min(200, len(coords)), replace=False)
ax.scatter(coords[sample_idx, 0], coords[sample_idx, 1],
           c='cyan', s=5, alpha=0.6, label=f'Muestra de {len(sample_idx)} puntos')

ax.set_title("Diagnostico: Heatmap + puntos de muestra", fontsize=16)
ax.legend()
out_png = os.path.join(sess, "_diag_heatmap_check.png")
fig.savefig(out_png, dpi=200, bbox_inches='tight')
plt.close()
print(f"  PNG guardado: {out_png}")
print(f"\n=== Fin. Abre el PNG para verificar si el maximo del heatmap coincide con los puntos ===")
