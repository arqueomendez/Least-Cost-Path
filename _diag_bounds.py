"""
Diagnostico: compara bounds/CRS del heatmap, puntos densificados,
y poligono de mascara. Ejecutar en la maquina con los datos.
"""
import os, sys
import pyogrio
import rasterio

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
    mask_crs = None
else:
    mask_info = pyogrio.read_info(MASK_PATH)
    mask_crs = mask_info["crs"]
    mb = mask_info["total_bounds"]
    mask_bounds = [mb[0], mb[1], mb[2], mb[3]]
    print(f"  CRS:    {mask_crs}")
    print(f"  Bounds: {mask_bounds}")
    print(f"  Tipo:   {mask_info['geometry_type']}")
print()

# 3. Puntos densificados (metadatos, no carga geometrias)
pp = os.path.join(sess, "puntos_rutas_25m.gpkg")
print("--- PUNTOS densificados ---")
if not os.path.exists(pp):
    print(f"  NO ENCONTRADO: {pp}")
    pt_crs = None
    pt_bounds = None
else:
    info = pyogrio.read_info(pp)
    pt_crs = info["crs"]
    b = info["total_bounds"]
    pt_bounds = [b[0], b[1], b[2], b[3]]
    print(f"  CRS:    {pt_crs}")
    print(f"  Bounds: {pt_bounds}")
    print(f"  Count:  {info['features']}")

    if mask_bounds is not None:
        dx0 = pt_bounds[0] - mask_bounds[0]
        dy0 = pt_bounds[1] - mask_bounds[1]
        dx1 = pt_bounds[2] - mask_bounds[2]
        dy1 = pt_bounds[3] - mask_bounds[3]
        print(f"  Diferencia pts - mask (minx,miny,maxx,maxy):")
        print(f"    ({dx0:.2f}, {dy0:.2f}, {dx1:.2f}, {dy1:.2f}) m")
        if pt_crs != mask_crs:
            print(f"  *** CRS MISMATCH! Puntos: {pt_crs}, Mask: {mask_crs}")
        else:
            print(f"  CRS OK: coinciden")
print()

# 4. Heatmap GeoTIFFs
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
            print(f"    Bounds:   [{left}, {bottom}, {right}, {top}]")
            print(f"    Res:      {src.res}")
            if mask_bounds is not None:
                print(f"    Diferencia minx vs mask:  {left - mask_bounds[0]:.2f} m")
                print(f"    Diferencia miny vs mask:  {bottom - mask_bounds[1]:.2f} m")
                print(f"    Diferencia maxx vs mask:  {right - mask_bounds[2]:.2f} m")
                print(f"    Diferencia maxy vs mask:  {top - mask_bounds[3]:.2f} m")
            if mask_crs is not None and str(src.crs) != str(mask_crs):
                print(f"    *** CRS MISMATCH vs mask! TIF: {src.crs}, Mask: {mask_crs}")
print()

# 5. Simular extent del notebook
if pt_bounds is not None:
    print("--- SIMULACION: extent del notebook ---")
    HEATMAP_BANDWIDTH_M = 10000
    minx, miny, maxx, maxy = pt_bounds
    minx -= HEATMAP_BANDWIDTH_M
    miny -= HEATMAP_BANDWIDTH_M
    maxx += HEATMAP_BANDWIDTH_M
    maxy += HEATMAP_BANDWIDTH_M
    print(f"  Extent con margen de {HEATMAP_BANDWIDTH_M}m:")
    print(f"    [{minx}, {miny}, {maxx}, {maxy}]")
    if mask_bounds is not None:
        covers = minx <= mask_bounds[0] and miny <= mask_bounds[1] and maxx >= mask_bounds[2] and maxy >= mask_bounds[3]
        print(f"  El extent del heatmap cubre el poligono mascara? {covers}")
        if not covers:
            misses = []
            if minx > mask_bounds[0]: misses.append("oeste")
            if miny > mask_bounds[1]: misses.append("sur")
            if maxx < mask_bounds[2]: misses.append("este")
            if maxy < mask_bounds[3]: misses.append("norte")
            print(f"    *** NO cubre por {', '.join(misses)}")

print("\n=== FIN ===")
