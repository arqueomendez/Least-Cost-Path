"""
Compara puntos de origen (perimetro) vs puntos densificados vs heatmap vs mascara.
"""
import os, sys
import pyogrio
import rasterio

MASK_PATH = r"C:\Users\LAV\Desktop\Proyecto Vannia\QGIS\Poligono ampliado AE.gpkg"
ORIGIN_PATH = r"C:\Users\LAV\Desktop\Proyecto Vannia\Proceso\coste\puntos del perimetro del poligono.gpkg"

outdir = os.path.join(os.getcwd(), "output")
sessions = sorted([d for d in os.listdir(outdir) if d.startswith("session_")],
                  key=lambda s: os.path.getmtime(os.path.join(outdir, s)), reverse=True)
sess = os.path.join(outdir, sessions[0])

print("=== MASCARA ===")
if os.path.exists(MASK_PATH):
    m = pyogrio.read_info(MASK_PATH)
    mb = m["total_bounds"]
    print(f"  Bounds: [{mb[0]:.2f}, {mb[1]:.2f}, {mb[2]:.2f}, {mb[3]:.2f}]")
else:
    print("  NO ENCONTRADA")
    mb = None
print()

print("=== PUNTOS DE ORIGEN (perimetro) ===")
if os.path.exists(ORIGIN_PATH):
    o = pyogrio.read_info(ORIGIN_PATH)
    ob = o["total_bounds"]
    print(f"  CRS:    {o['crs']}")
    print(f"  Count:  {o['features']}")
    print(f"  Bounds: [{ob[0]:.2f}, {ob[1]:.2f}, {ob[2]:.2f}, {ob[3]:.2f}]")
    if mb:
        print(f"  Dif vs mask: [{ob[0]-mb[0]:.2f}, {ob[1]-mb[1]:.2f}, {ob[2]-mb[2]:.2f}, {ob[3]-mb[3]:.2f}]")
else:
    print(f"  NO ENCONTRADO: {ORIGIN_PATH}")
print()

print("=== PUNTOS DENSIFICADOS (rutas) ===")
pp = os.path.join(sess, "puntos_rutas_25m.gpkg")
if os.path.exists(pp):
    p = pyogrio.read_info(pp)
    pb = p["total_bounds"]
    print(f"  CRS:    {p['crs']}")
    print(f"  Count:  {p['features']}")
    print(f"  Bounds: [{pb[0]:.2f}, {pb[1]:.2f}, {pb[2]:.2f}, {pb[3]:.2f}]")
    if mb:
        print(f"  Dif vs mask: [{pb[0]-mb[0]:.2f}, {pb[1]-mb[1]:.2f}, {pb[2]-mb[2]:.2f}, {pb[3]-mb[3]:.2f}]")
    if os.path.exists(ORIGIN_PATH):
        print(f"  Dif vs origen: [{pb[0]-ob[0]:.2f}, {pb[1]-ob[1]:.2f}, {pb[2]-ob[2]:.2f}, {pb[3]-ob[3]:.2f}]")
else:
    print(f"  NO ENCONTRADO: {pp}")
print()

print("=== HEATMAP TIF ===")
tifs = sorted([f for f in os.listdir(sess) if f.startswith("heatmap_") and f.endswith(".tif")])
if tifs:
    tp = os.path.join(sess, tifs[0])
    with rasterio.open(tp) as src:
        l, b, r, t = src.bounds
        print(f"  Bounds: [{l:.2f}, {b:.2f}, {r:.2f}, {t:.2f}]")
        print(f"  Res:    {src.res}")
        print(f"  CRS:    {src.crs}")
        if mb:
            print(f"  Dif minx vs mask:  {l - mb[0]:.2f}")
            print(f"  Dif miny vs mask:  {b - mb[1]:.2f}")
            print(f"  Dif maxx vs mask:  {r - mb[2]:.2f}")
            print(f"  Dif maxy vs mask:  {t - mb[3]:.2f}")
        if os.path.exists(ORIGIN_PATH):
            print(f"  Dif minx vs origen: {l - ob[0]:.2f}")
            print(f"  Dif miny vs origen: {b - ob[1]:.2f}")
            print(f"  Dif maxx vs origen: {r - ob[2]:.2f}")
            print(f"  Dif maxy vs origen: {t - ob[3]:.2f}")
else:
    print("  NO HAY TIFs")
print()

print("=== VERIFICACION: origen vs mascara ===")
if os.path.exists(ORIGIN_PATH) and mb:
    # Los puntos de origen deberian coincidir con el borde del poligono
    print(f"  Los puntos de origen estan en el borde de la mascara?")
    for i, label in enumerate(["minx", "miny", "maxx", "maxy"]):
        diff = ob[i] - mb[i]
        print(f"    {label}: diff = {diff:.2f} m {'(OK, cerca del borde)' if abs(diff) < 100 else '(LEJOS del borde!)'}")

print("\n=== FIN ===")
