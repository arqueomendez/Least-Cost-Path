"""
Diagnostico: compara bounds/CRS del heatmap, puntos densificados,
y poligono de mascara. Ejecutar en la maquina con los datos.
"""
import os, sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from shapely import get_coordinates

MASK_PATH = r"C:\Users\LAV\Desktop\Proyecto Vannia\QGIS\Poligono ampliado AE.gpkg"

# 1. Encontrar ultima sesion
outdir = os.path.join(os.getcwd(), "output")
sessions = sorted(
    [d for d in os.listdir(outdir) if d.startswith("session_")],
    key=lambda s: os.path.getmtime(os.path.join(outdir, s)),
    reverse=True,
)
if not sessions:
    print("ERROR: no hay sesiones en output/")
    sys.exit(1)

sess = os.path.join(outdir, sessions[0])
print(f"=== Sesion: {sessions[0]} ===\n")

# 2. Cargar mascara
print("--- MASK (Poligono ampliado AE) ---")
if not os.path.exists(MASK_PATH):
    print(f"  NO ENCONTRADO: {MASK_PATH}")
    mask_bounds = None
else:
    mask = gpd.read_file(MASK_PATH)
    print(f"  CRS:    {mask.crs}")
    print(f"  Bounds: {mask.total_bounds}")
    print(f"  Tipo:   {mask.geometry.iloc[0].geom_type if len(mask) > 0 else 'N/A'}")
    mask_bounds = mask.total_bounds
print()

# 3. Cargar puntos densificados
pp = os.path.join(sess, "puntos_rutas_25m.gpkg")
print("--- PUNTOS densificados ---")
if not os.path.exists(pp):
    print(f"  NO ENCONTRADO: {pp}")
else:
    pts = gpd.read_file(pp)
    print(f"  CRS:    {pts.crs}")
    print(f"  Bounds: {pts.total_bounds}")
    print(f"  Count:  {len(pts)}")
    pt_bounds = pts.total_bounds

    if mask_bounds is not None:
        print(f"  Diferencia vs mask (minx,miny,maxx,maxy):")
        print(f"    ({pt_bounds[0]-mask_bounds[0]:.2f}, {pt_bounds[1]-mask_bounds[1]:.2f}, {pt_bounds[2]-mask_bounds[2]:.2f}, {pt_bounds[3]-mask_bounds[3]:.2f}) m")
print()

# 4. Cargar heatmap GeoTIFFs
print("--- HEATMAP GeoTIFFs ---")
tifs = sorted([f for f in os.listdir(sess) if f.startswith("heatmap_") and f.endswith(".tif")])
if not tifs:
    print("  NO HAY archivos heatmap_*.tif en la sesion")
else:
    for tif in tifs:
        tp = os.path.join(sess, tif)
        with rasterio.open(tp) as src:
            left, bottom, right, top = src.bounds
            print(f"  {tif}")
            print(f"    CRS:      {src.crs}")
            print(f"    Shape:    {src.width} x {src.height}")
            print(f"    Bounds:   {[left, bottom, right, top]}")
            print(f"    Res:      {src.res}")
            if mask_bounds is not None:
                print(f"    Diferencia minx vs mask:  {left - mask_bounds[0]:.2f} m")
                print(f"    Diferencia miny vs mask:  {bottom - mask_bounds[1]:.2f} m")
                print(f"    Diferencia maxx vs mask:  {right - mask_bounds[2]:.2f} m")
                print(f"    Diferencia maxy vs mask:  {top - mask_bounds[3]:.2f} m")
print()

# 5. Verificacion: la linea de puntos debe caer DENTRO del poligono mascara
if mask_bounds is not None and os.path.exists(pp):
    print("--- VERIFICACION: puntos dentro de mask? ---")
    mask_gdf = gpd.read_file(MASK_PATH)
    pts2 = gpd.read_file(pp)
    # CRS check
    if pts2.crs != mask_gdf.crs:
        print(f"  CRS MISMATCH! Puntos: {pts2.crs}, Mask: {mask_gdf.crs}")
    else:
        # Quick sample: check 1000 random points
        sample = pts2.sample(min(1000, len(pts2)), random_state=42)
        inside = gpd.sjoin(sample, mask_gdf, predicate="within")
        ratio = len(inside) / len(sample) * 100
        print(f"  Puntos dentro del poligono: {ratio:.1f}% (n={len(sample)})")

        if ratio < 90:
            print(f"  *** ALERTA: muchos puntos fuera del poligono! Posible shift o CRS incorrecto")
            # Show some outside points
            outside = sample[~sample.index.isin(inside.index)]
            print(f"  Primeros 5 puntos fuera del poligono:")
            for _, row in outside.head(5).iterrows():
                print(f"    ({row.geometry.x:.1f}, {row.geometry.y:.1f})")
print()

# 6. Recalcular el extent del heatmap con el codigo actual del notebook
if os.path.exists(pp):
    print("--- SIMULACION: extent del notebook ---")
    pts3 = gpd.read_file(pp)
    HEATMAP_BANDWIDTH_M = 10000
    minx, miny, maxx, maxy = pts3.total_bounds
    minx -= HEATMAP_BANDWIDTH_M
    miny -= HEATMAP_BANDWIDTH_M
    maxx += HEATMAP_BANDWIDTH_M
    maxy += HEATMAP_BANDWIDTH_M
    print(f"  Extent con margen de {HEATMAP_BANDWIDTH_M}m:")
    print(f"    [{minx}, {miny}, {maxx}, {maxy}]")
    if mask_bounds is not None:
        print(f"  El extent del heatmap cubre el poligono mascara?")
        covers = minx <= mask_bounds[0] and miny <= mask_bounds[1] and maxx >= mask_bounds[2] and maxy >= mask_bounds[3]
        print(f"    {covers}")
        if not covers:
            misses = []
            if minx > mask_bounds[0]: misses.append("oeste")
            if miny > mask_bounds[1]: misses.append("sur")
            if maxx < mask_bounds[2]: misses.append("este")
            if maxy < mask_bounds[3]: misses.append("norte")
            print(f"    *** El heatmap NO cubre la mascara por {', '.join(misses)}")

print("\n=== FIN ===")
