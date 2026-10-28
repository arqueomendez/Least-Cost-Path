#!/usr/bin/env python
# coding: utf-8

# ## Celda 1 — Importaciones y módulos
# 
# Carga las librerías del proyecto (rasterio, fiona, geopandas, joblib, matplotlib, psutil) y los módulos del paquete `lcp/` (`data_loader`, `processing`, `pathfinder`, `utils`, `postprocessing`). Si falla esta celda, el entorno no está sincronizado: ejecuta `uv sync`.

# In[ ]:


# ==============================================================================
# --- CELDA 1: IMPORTACIONES Y MÓDULOS ---
# ==============================================================================
import os
import numpy as np
import rasterio
import fiona
from datetime import datetime
import time
import gc
import shutil

# --- Adiciones para Paralelización, Monitoreo y Visualización ---
import psutil
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import geopandas as gpd
import rasterio.plot
from rasterio.windows import from_bounds
from shapely.geometry import box

# --- Importar los módulos personalizados del paquete lcp ---
import lcp.data_loader as dl
import lcp.processing as proc
import lcp.pathfinder as pf
import lcp.utils as utils
import lcp.postprocessing as postp 

print("Módulos cargados exitosamente.")


# ## Celda 2 — Funciones de soporte
# 
# - `_calculate_n_jobs_for_phase`: estima cuántos workers caben según la RAM disponible (Fase 1 ≈ `raster × 5.5` por worker; Fase 2 ≈ 80 MB; tope de 61 procesos en Windows).
# - `warm_up_numba`: pre-compila la reconstrucción JIT antes del bucle principal y verifica que los offsets de `MCP_Geometric` sigan siendo los esperados.

# In[ ]:


# ==============================================================================
# --- CELDA 2: DEFINICIÓN DE FUNCIONES ---
# ==============================================================================

def _calculate_n_jobs_for_phase(raster_shape, raster_dtype, phase):
    """
    Estima cuántos workers paralelos caben en la RAM disponible para cada fase.
    (El traceback de skimage mide int16 (2 B/celda); las estimaciones son conservadoras.)
    """
    import psutil
    import numpy as np
    import sys

    available_ram = psutil.virtual_memory().available
    raster_bytes = np.prod(raster_shape) * np.dtype(raster_dtype).itemsize

    if phase == 1:
        # costes planos (f64) + heap + offsets + traceback (i16); SIN copia
        # del raster: el worker usa la vista del memmap (S3 perf).
        mem_per_worker = raster_bytes * 4.5
        n_jobs = max(1, int(available_ram * 0.80 / mem_per_worker))
        n_cpu = psutil.cpu_count(logical=True) or 1
        n_jobs = min(n_jobs, n_cpu)
    else:
        # Fase 2: traceback comprimido en disco -> mucho menos RAM.
        mem_per_worker = 80 * 1024 * 1024
        n_jobs = max(1, int(available_ram * 0.80 / mem_per_worker))
        n_cpu = psutil.cpu_count(logical=True) or 1
        n_jobs = min(n_jobs, n_cpu * 2)

    if sys.platform == "win32":
        n_jobs = min(n_jobs, 61)   # límite de WaitForMultipleObjects en Windows

    return n_jobs


def warm_up_numba():
    """
    Pre-compila reconstruct_path_from_traceback con Numba JIT usando datos de
    prueba y, de paso, VERIFICA que el orden de offsets esperado por defecto
    coincide con el del MCP de la versión instalada de scikit-image.

    Con @njit(cache=True) en pathfinder, la compilación se guarda en disco y los
    workers loky la reutilizan (no recompilan en su primer origen).
    """
    print("Pre-compilando función Numba (warm-up) y verificando offsets...")
    import numpy as np
    from skimage.graph import MCP_Geometric

    dummy_raster = np.ones((10, 10), dtype=np.float64)
    mcp = MCP_Geometric(dummy_raster, fully_connected=True)
    _, dummy_traceback = mcp.find_costs([(5, 5)])
    offsets = np.asarray(mcp.offsets, dtype=np.int32)

    # Red de seguridad: si skimage reordenara los offsets, el respaldo por defecto
    # quedaría inválido. Avisamos en lugar de generar rutas erróneas en silencio.
    if not np.array_equal(offsets, pf.DEFAULT_OFFSETS):
        print(
            "  ADVERTENCIA: el orden de mcp.offsets difiere de DEFAULT_OFFSETS. "
            "Se usarán los offsets reales guardados en cada .npz (esto es correcto), "
            "pero revisa pf.DEFAULT_OFFSETS para compatibilidad con .npz antiguos."
        )

    pf.reconstruct_path_from_traceback(dummy_traceback, (5, 5), (2, 2), offsets)
    print("Warm-up completado.")


# ## Modos de análisis — `all_pairs` vs `single_origin`
# 
# Este notebook calcula rutas de menor coste (`MCP_Geometric` de scikit-image, 8-conectado, coste geométrico) entre los puntos de la red. La **Celda 3** (justo debajo de este texto) permite elegir entre dos modos de análisis:
# 
# ### 1. `ANALYSIS_MODE = "all_pairs"` — todos contra todos (default)
# 
# - Calcula **todas** las rutas entre **todos** los pares de puntos: para N puntos, N·(N−1)/2 rutas (triángulo superior de la matriz OD). Con los datos de ejemplo (N = 14): **91 rutas**.
# - Cada origen lanza su propio MCP hacia los destinos con id mayor: el filtro `did > origin_id` es la deduplicación que evita recalcular pares simétricos (la ruta A→B es idéntica a B→A).
# - **Cuándo usarlo**: cuando necesitas la red completa — matrices OD, corredores entre todos los puntos, densificación de rutas (Celda 8) y mapas de calor (Celda 9).
# 
# ### 2. `ANALYSIS_MODE = "single_origin"` — un origen contra todos (modo estrella / SSSP)
# 
# - Calcula **solo** las rutas desde el punto `ORIGIN_ID` hacia **todos los demás** puntos, y **ninguna entre los otros puntos**. Con N = 14: **13 rutas**, todas con `origin_id = ORIGIN_ID`.
# - Equivale a *single-source shortest paths* (SSSP) frente a *all-pairs* (APSP): un parámetro que acota el alcance, al estilo de `source=` en NetworkX.
# - Como `MCP_Geometric` es simétrico, la ruta O→P calculada es idéntica a P→O: para ese origen no se pierde información respecto al modo completo.
# - **Cuándo usarlo**: cuando te interesa **un sitio concreto** — accesibilidad de un punto aislado, coste de acceso a un rasgo puntual, o comparar un origen frente al resto sin pagar el análisis N×N completo (1 MCP en vez de N−1; ~13× más rápido con los datos de ejemplo).
# 
# ### Comparación rápida
# 
# | | `all_pairs` | `single_origin` |
# |---|---|---|
# | Rutas (N=14) | 91 | 13 |
# | Dedup | filtro `did > origin_id` | `did != ORIGIN_ID` (sin triángulo) |
# | MCPs en Fase 1 | 13 (uno por origen) | 1 (solo el origen) |
# | Validación (Celda 4) | CRS, superposición y puntos | + modo válido, `ORIGIN_ID` definido y existente (lista los IDs disponibles) |
# | Mapas (Celdas 5 y 7) | todos los puntos en rojo | origen resaltado con **estrella** lime |
# 
# ### Cómo elegirlo y qué esperar
# 
# - Edita `ANALYSIS_MODE` y `ORIGIN_ID` en la **Celda 3** (etiquetada `parameters`), o ejecutalo sin abrir Jupyter:
# 
# ```sh
# uv run papermill LCP_MCP_Geometric.ipynb out.ipynb -p ANALYSIS_MODE single_origin -p ORIGIN_ID 7
# ```
# 
# - **Salida**: el mismo GeoPackage de siempre (`red_completa_unificada.gpkg`) con los mismos campos (`origin_id`, `dest_id`, `cost_total`, `length_m`, `n_vertices`), solo que con N−1 filas en modo estrella.
# - La **Celda 6** imprime `MODO: ... | rutas esperadas: N` al arrancar y, al final, compara las rutas reales con las esperadas: si difieren, revisa los destinos con coste infinito (inalcanzables) que se informan arriba.
# 

# In[ ]:


# ==============================================================================
# --- CELDA 3: CONFIGURACIÓN CENTRAL DEL ANÁLISIS ---
# ==============================================================================
# --- 1. Parámetros de Rutas ---
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', f'session_{timestamp}_Refactored')

COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')
ALL_POINTS_SHAPEFILE = os.path.join(DATA_DIR, 'points.shp')
MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')
ID_FIELD_NAME = 'id'

# --- 2. Modo de analisis ---
# "all_pairs": matriz completa N x N (triangulo superior OD) - comportamiento historico.
# "single_origin": UN solo origen ORIGIN_ID hacia todos los demas puntos (SSSP),
# sin pares entre los destinos (estilo source= de NetworkX).
ANALYSIS_MODE = "all_pairs"   # "all_pairs" | "single_origin"
ORIGIN_ID = None              # requerido en modo single_origin (ID del punto)
RESUME = True                 # reutilizar tracebacks de Fase 1 si la huella (S9) coincide

# --- 3. Opciones de Almacenamiento ---
KEEP_PHASE1_FILES = False
VERBOSE = False

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directorio de salida creado/verificado en: {OUTPUT_DIR}")
except OSError as e:
    print(f"!! ERROR CRÍTICO !! No se pudo crear el directorio de salida. Error: {e}")
    raise

print("Parámetros de configuración cargados. Los resultados se guardarán en:", OUTPUT_DIR)


# ## Celda 4 — Verificación de coherencia de los datos
# 
# Comprueba, en orden: CRS idénticos (ráster / puntos / máscara), solape espacial de extensiones, y que **todos** los puntos caigan en píxeles válidos (fuera de máscara o nodata). En modo `single_origin` además valida el modo, que haya ≥ 2 puntos y que `ORIGIN_ID` exista (lista los IDs disponibles). Cualquier problema **detiene la ejecución** con un mensaje concreto; nada se calcula con datos incoherentes.

# In[ ]:


# ==============================================================================
# --- CELDA 4: VERIFICACIÓN DE COHERENCIA DE DATOS ---
# ==============================================================================
from pyproj import CRS  # comparación de CRS robusta entre clases fiona/rasterio

print("\n--- VERIFICANDO LA COHERENCIA DE TODOS LOS DATOS DE ENTRADA ---")

# --- Validacion del modo de analisis (S3): fallar temprano y con mensajes claros ---
_VALID_MODES = ("all_pairs", "single_origin")
if ANALYSIS_MODE not in _VALID_MODES:
    raise ValueError(
        f"ANALYSIS_MODE invalido: {ANALYSIS_MODE!r}. Modos validos: {_VALID_MODES}"
    )
if ANALYSIS_MODE == "single_origin" and ORIGIN_ID is None:
    raise ValueError("Modo single_origin requiere definir ORIGIN_ID en la Celda 3.")

main_search_mask = None
invalid_point_ids = []

try:
    # --- Etapa 1: CRS ---
    print("--- Etapa 1: Comprobación de Sistemas de Coordenadas (CRS) ---")

    with rasterio.open(COST_RASTER_PATH) as src:
        raster_crs = CRS.from_user_input(src.crs)

    with fiona.open(ALL_POINTS_SHAPEFILE) as src:
        points_crs = CRS.from_user_input(src.crs)

    print(f"  > CRS del Ráster: {raster_crs.to_string()}")
    print(f"  > CRS de los Puntos: {points_crs.to_string()}")

    crs_errors = []
    if not raster_crs.equals(points_crs):
        crs_errors.append("El CRS de los PUNTOS no coincide con el del RÁSTER.")

    if MASK_SHAPEFILE_PATH and os.path.exists(MASK_SHAPEFILE_PATH):
        with fiona.open(MASK_SHAPEFILE_PATH) as src:
            mask_crs = CRS.from_user_input(src.crs)
        print(f"  > CRS de la Máscara: {mask_crs.to_string()}")
        if not raster_crs.equals(mask_crs):
            crs_errors.append("El CRS de la MÁSCARA no coincide con el del RÁSTER.")

    if not crs_errors:
        print("¡Correcto! Todos los sistemas de coordenadas coinciden.")

    # --- Etapa 2: Superposición espacial ---
    print("\n--- Etapa 2: Comprobación de Superposición Espacial ---")

    with rasterio.open(COST_RASTER_PATH) as src:
        raster_bounds = src.bounds
        raster_box = box(*raster_bounds)

    points_gdf = gpd.read_file(ALL_POINTS_SHAPEFILE)
    points_bounds = points_gdf.total_bounds
    points_box = box(*points_bounds)

    print(f"  > Extensión del Ráster: {raster_bounds}")
    print(f"  > Extensión de los Puntos: {tuple(points_bounds)}")

    superposition_errors = []
    if not points_box.within(raster_box):
        superposition_errors.append(
            "La extensión de los PUNTOS no está completamente contenida dentro de la "
            "extensión del RÁSTER."
        )

    if MASK_SHAPEFILE_PATH and os.path.exists(MASK_SHAPEFILE_PATH):
        mask_gdf = gpd.read_file(MASK_SHAPEFILE_PATH)
        if not mask_gdf.empty:
            mask_bounds = mask_gdf.total_bounds
            mask_box = box(*mask_bounds)
            print(f"  > Extensión de la Máscara: {tuple(mask_bounds)}")
            if not mask_box.intersects(raster_box):
                superposition_errors.append(
                    "La extensión de la MÁSCARA no se superpone con la del RÁSTER."
                )

    if not superposition_errors:
        print("¡Correcto! Todas las capas se superponen geográficamente.")

    # --- Etapa 3: Máscara + nodata + validación de puntos ---
    print("\n--- Etapa 3: Creación y Validación Lógica Final ---")

    with rasterio.open(COST_RASTER_PATH) as src:
        if MASK_SHAPEFILE_PATH and os.path.exists(MASK_SHAPEFILE_PATH):
            vector_mask = proc.create_mask_from_vector(MASK_SHAPEFILE_PATH, src)
        else:
            print("  ADVERTENCIA: Sin archivo de máscara. Se usará toda la extensión.")
            vector_mask = np.ones(src.shape, dtype=bool)

        original_cost_data = src.read(1)
        nodata_value = src.nodata
        if nodata_value is not None:
            nodata_mask = (original_cost_data == nodata_value)
            print(f"  > {np.sum(nodata_mask)} píxeles con valor 'nodata' en el ráster.")
        else:
            nodata_mask = np.zeros(src.shape, dtype=bool)

        main_search_mask = vector_mask & ~nodata_mask

        all_points_for_validation, _ = dl.load_points_as_dict(
            ALL_POINTS_SHAPEFILE, ID_FIELD_NAME
        )
        if ANALYSIS_MODE == "single_origin" and len(all_points_for_validation) < 2:
            raise ValueError(
                "Modo single_origin requiere al menos 2 puntos (N >= 2); "
                f"se cargaron {len(all_points_for_validation)}."
            )
        if ANALYSIS_MODE == "single_origin" and ORIGIN_ID not in all_points_for_validation:
            raise ValueError(
                f"ORIGIN_ID {ORIGIN_ID!r} no existe. "
                f"IDs disponibles: {sorted(all_points_for_validation)}"
            )
        for point_id, coords in all_points_for_validation.items():
            row, col = rasterio.transform.rowcol(src.transform, coords[0], coords[1])
            if not (0 <= row < src.height and 0 <= col < src.width) \
               or not main_search_mask[row, col]:
                invalid_point_ids.append(str(point_id))

        if not invalid_point_ids:
            print("¡Correcto! Todos los puntos están dentro del área válida final.")

    # --- Etapa 4: Veredicto ---
    final_errors = crs_errors + superposition_errors

    if invalid_point_ids:
        final_errors.append(
            f"Se encontraron {len(invalid_point_ids)} puntos sobre píxeles INVÁLIDOS "
            f"(fuera de la máscara o con 'nodata'). IDs de muestra: "
            f"{', '.join(invalid_point_ids[:10])}"
            f"{'...' if len(invalid_point_ids) > 10 else ''}."
        )

    if final_errors:
        error_details = "\n- ".join(final_errors)
        raise ValueError(
            f"\n{'='*70}\n"
            f"!! ERROR CRÍTICO DE VALIDACIÓN DE DATOS !!\n"
            f"Se encontraron las siguientes inconsistencias:\n- {error_details}\n\n"
            f"El análisis no puede continuar. Corrige tus datos de entrada.\n"
            f"{'='*70}"
        )
    else:
        print("\n¡Excelente! Todos los datos son coherentes y listos para el análisis.")

except Exception as e:
    print(f"Ocurrió un error inesperado durante la validación de datos: {e}")
    raise


# ## Celda 5 — Visualización diagnóstica
# 
# Mapa del ráster de coste en escala **lineal y logarítmica** con los puntos superpuestos, guardado como `visualizacion_diagnostica.png`. Usalo para verificar la configuración antes de lanzar el análisis; en modo `single_origin` el origen aparece resaltado con una estrella.

# In[ ]:


# ==============================================================================
# --- CELDA 5: VISUALIZACIÓN DIAGNÓSTICA (CON EXPORTACIÓN) ---
# ==============================================================================
print("\nIniciando visualización diagnóstica del ráster de coste y puntos...")
try:
    # Cargar el ráster de coste para obtener sus datos y metadatos
    with rasterio.open(COST_RASTER_PATH) as src:
        cost_data = src.read(1)
        transform = src.transform  # Guardar la transformación para convertir coordenadas

        # Preparar datos para la escala logarítmica, manejando valores no positivos
        min_pos_val = np.min(cost_data[cost_data > 0]) if np.any(cost_data > 0) else 1e-9
        cost_data_log = np.where(cost_data <= 0, min_pos_val, cost_data)

        # Graficar los nodos ya cargados y validados en la Celda 4 (dict id -> (x, y)).
        # No se accede a la geometria cruda releida del archivo: una feature
        # MultiPoint no expone .x/.y, y sus partes ya viven en el dict explotado.
        point_pixels_x = []
        point_pixels_y = []
        for _px, _py in all_points_for_validation.values():
            # Usar rasterio.transform.rowcol para obtener fila y columna
            row, col = rasterio.transform.rowcol(transform, _px, _py)
            point_pixels_x.append(col)
            point_pixels_y.append(row)

        # Crear la figura y los ejes para los dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

        # --- Subplot 1: Visualización con Escala Lineal ---
        im1 = ax1.imshow(cost_data, cmap='viridis')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title('Escala Lineal con Puntos', fontsize=16)
        ax1.scatter(point_pixels_x, point_pixels_y, c='red', edgecolor='black', s=50, label='Puntos de Análisis', zorder=5)
        ax1.legend()

        # --- Subplot 2: Visualización con Escala Logarítmica ---
        im2 = ax2.imshow(cost_data_log, cmap='viridis', norm=LogNorm())
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Escala Logarítmica con Puntos', fontsize=16)
        ax2.scatter(point_pixels_x, point_pixels_y, c='red', edgecolor='black', s=50, label='Puntos de Análisis', zorder=5)
        ax2.legend()

        # Resaltar el origen en modo single_origin (S5; seleccion simple, no interactiva)
        if (
            "ANALYSIS_MODE" in globals()
            and ANALYSIS_MODE == "single_origin"
            and "all_points_for_validation" in globals()
            and ORIGIN_ID in all_points_for_validation
        ):
            _ox, _oy = all_points_for_validation[ORIGIN_ID]
            _orow, _ocol = rasterio.transform.rowcol(transform, _ox, _oy)
            for _ax in (ax1, ax2):
                _ax.scatter([_ocol], [_orow], c='lime', edgecolor='black', s=400,
                            marker="*", zorder=6, label=f'Origen {ORIGIN_ID}')
                _ax.legend()

        # Ajustes finales
        plt.tight_layout()

        # --- INICIO DE LA MODIFICACIÓN: EXPORTACIÓN DE LA IMAGEN ---
        # Guardar la figura en el directorio de salida antes de mostrarla
        output_image_path = os.path.join(OUTPUT_DIR, 'visualizacion_diagnostica.png')
        # Usamos dpi=150 para una buena resolución y bbox_inches='tight' para eliminar bordes blancos
        fig.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualización diagnóstica guardada en: {output_image_path}")
        # --- FIN DE LA MODIFICACIÓN ---

        # Mostrar el gráfico en la pantalla
        plt.show()

except FileNotFoundError as e:
    print(f"ADVERTENCIA: No se encontró un archivo necesario para la visualización: {e}. Saltando visualización.")
except Exception as e:
    print(f"Ocurrió un error durante la visualización: {e}")
    import traceback
    traceback.print_exc()


# ## Celda 6 — Ejecución en dos fases
# 
# - **Fase 1**: un `MCP_Geometric` por origen (solo el origen elegido en modo estrella) que guarda traceback **y escalares** en `phase1_tracebacks/`. Con `RESUME = True`, si la huella (puntos + ráster + modo) coincide se reutilizan los `.npz` existentes y solo se computan los pendientes.
# - **Fase 2**: reconstrucción Numba de cada ruta y escritura de `red_completa_unificada.gpkg` (más la capa `destinos_inalcanzables` si los hay).
# - Imprime `MODO:` con las rutas esperadas y compara al final esperadas vs reales (diferencia = destinos con coste infinito). Si un worker falla, la ejecución **se detiene** antes de escribir el GPKG.

# In[ ]:


# ==============================================================================
# --- CELDA 6: EJECUCIÓN EN DOS FASES (FASE 1: MCP + FASE 2: RECONSTRUCCIÓN) ---
# ==============================================================================
print("\n--- INICIANDO PROCESO DE ANÁLISIS EN DOS FASES ---\n")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE1_DIR = os.path.join(OUTPUT_DIR, "phase1_tracebacks")
os.makedirs(PHASE1_DIR, exist_ok=True)

memmap_path = os.path.join(OUTPUT_DIR, "_raster_shared.dat")
gpkg_output_path = os.path.join(OUTPUT_DIR, "red_completa_unificada.gpkg")
gpkg_tmp_path = gpkg_output_path + ".tmp"  # escritura incremental (S1); replace atomico al final
routes_gdf = None  # None => la Celda 7 lee el GPKG desde disco (nada en memoria)
n_rutas_escritas = 0
dur_p1 = dur_p2 = 0.0
tasks_p1, tasks_p2 = [], []
n_jobs_p1 = n_jobs_p2 = 0
pipeline_ok = False  # True solo si las 5 fases terminan; controla la limpieza en finally.

try:
    # --- PASO 1: Preparar datos ---
    print("[Paso 1/5] Preparando datos en memoria...")
    all_points, points_crs = dl.load_points_as_dict(ALL_POINTS_SHAPEFILE, ID_FIELD_NAME)

    with dl.load_raster(COST_RASTER_PATH) as src:
        transform = src.transform
        raster_crs = src.crs
        original_cost_data = src.read(1)
        final_cost_raster = np.where(
            main_search_mask, original_cost_data, np.inf
        ).astype(np.float64)
        raster_shape = final_cost_raster.shape
        raster_dtype = final_cost_raster.dtype
        del original_cost_data

    print(f"  Forma raster: {raster_shape}, dtype: {raster_dtype}")

    print("[Paso 2/5] Escribiendo raster en memmap compartido...")
    mm = np.memmap(memmap_path, dtype=raster_dtype, mode="w+", shape=raster_shape)
    mm[:] = final_cost_raster[:]
    mm.flush()
    del mm, final_cost_raster
    print(f"  Memmap escrito en: {memmap_path}")

    shared_data = {
        "memmap_path":  memmap_path,
        "raster_dtype": raster_dtype,
        "raster_shape": raster_shape,
        "src_transform": transform,
        # S2 (perf): comprimir los npz solo si persisten (reanudacion con
        # archivos chicos); temporales -> np.savez plano, ~29 s/origen mas rapido.
        "compress_npz": KEEP_PHASE1_FILES,
        # "bbox_buffer": None  # None = raster completo (garantiza optimalidad).
    }

    warm_up_numba()

    dest_pixels_all = {
        pid: proc.world_to_pixel(transform, coords[0], coords[1])
        for pid, coords in all_points.items()
    }

    # --- S9: huella de reanudacion (modo + origen + puntos + identidad del raster) ---
    import hashlib
    import json as _json

    _fp_data = {
        "mode": ANALYSIS_MODE,
        "origin": ORIGIN_ID,
        "points": sorted(
            (int(k), round(float(v[0]), 6), round(float(v[1]), 6))
            for k, v in all_points.items()
        ),
        "raster_size": os.path.getsize(COST_RASTER_PATH),
        "raster_mtime": int(os.path.getmtime(COST_RASTER_PATH)),
    }
    fingerprint = hashlib.sha256(
        _json.dumps(_fp_data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    meta_path = os.path.join(PHASE1_DIR, "phase1_meta.json")
    _meta_ok = False
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as _fh:
                _meta_ok = _json.load(_fh).get("fingerprint") == fingerprint
        except (OSError, ValueError):
            _meta_ok = False
    if not _meta_ok:
        # huella distinta o meta ilegible: los npz previos no sirven
        shutil.rmtree(PHASE1_DIR, ignore_errors=True)
        os.makedirs(PHASE1_DIR, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as _fh:
        _json.dump({"fingerprint": fingerprint}, _fh)

    # --- PASO 3: FASE 1 (segun modo de analisis) ---
    if ANALYSIS_MODE == "single_origin":
        # Modo estrella (SSSP): UN solo MCP desde ORIGIN_ID hacia todos los demas
        # puntos; sin filtro did > origin_id (los ids menores tambien corresponden)
        # y sin pares entre los destinos.
        n_expected = len(all_points) - 1
        origin_coords = all_points[ORIGIN_ID]
        origin_pixel = proc.world_to_pixel(transform, origin_coords[0], origin_coords[1])
        dest_pixels = {did: dest_pixels_all[did] for did in all_points if did != ORIGIN_ID}
        tasks_p1.append((ORIGIN_ID, origin_pixel, dest_pixels, shared_data, PHASE1_DIR))
    else:
        # all_pairs: N x (N-1) / 2 via filtro did > origin_id (triangulo superior).
        n_expected = len(all_points) * (len(all_points) - 1) // 2
        for origin_id, origin_coords in all_points.items():
            origin_pixel = proc.world_to_pixel(transform, origin_coords[0], origin_coords[1])
            dest_pixels = {
                did: dest_pixels_all[did] for did in all_points if did > origin_id
            }
            if not dest_pixels:
                # El origen de mayor id no tiene destinos: evitamos un Dijkstra global
                # inutil y un .npz enorme que la Fase 2 descartaria.
                continue
            tasks_p1.append((origin_id, origin_pixel, dest_pixels, shared_data, PHASE1_DIR))

    n_jobs_p1 = _calculate_n_jobs_for_phase(raster_shape, raster_dtype, phase=1)
    modo_desc = (
        f"single_origin (origen {ORIGIN_ID})"
        if ANALYSIS_MODE == "single_origin"
        else "all_pairs (N x N)"
    )
    print(f"\nMODO: {modo_desc} | rutas esperadas: {n_expected}")
    print(f"[Paso 3/5] FASE 1 — MCP en paralelo ({n_jobs_p1} workers)...")

    # --- S9: reutilizar tracebacks con huella coincidente (RESUME) ---
    reusable = {}
    tasks_p1_run = tasks_p1
    if RESUME:
        tasks_p1_run = []
        for task in tasks_p1:
            _npz = os.path.join(PHASE1_DIR, f"traceback_{task[0]}.npz")
            try:
                with np.load(_npz) as data:
                    if "dest_ids" not in data.files or "cost_scalars" not in data.files:
                        raise ValueError("npz sin escalares de coste")
                    reusable[task[0]] = {
                        int(d): float(c)
                        for d, c in zip(data["dest_ids"], data["cost_scalars"])
                    }
            except Exception:
                tasks_p1_run.append(task)
        if reusable:
            print(
                f"  RESUME: reutilizando {len(reusable)} tracebacks de Fase 1 "
                f"(huella coincide); pendientes: {len(tasks_p1_run)}"
            )

    print(f"  Total de orígenes a procesar: {len(tasks_p1_run)}")
    t0_p1 = time.time()

    # S3: progreso por resultado consumido (tqdm.auto) y sin ruido de joblib.
    # generator_ordered: los escalares son dicts pequenos; el buffer de orden
    # es irrelevante en memoria y conserva el orden de envio.
    from tqdm.auto import tqdm

    results_p1 = Parallel(
        n_jobs=n_jobs_p1, backend="loky", verbose=0, return_as="generator"
    )(delayed(pf.compute_and_save_cost_surface)(args) for args in tasks_p1_run)

    cost_scalars_index = {}
    with tqdm(
        total=len(tasks_p1), desc="Fase 1 (MCP)", unit="origen", disable=None
    ) as pbar1:
        for oid, cdict in results_p1:
            cost_scalars_index[oid] = cdict
            pbar1.update(1)

    dur_p1 = time.time() - t0_p1
    print(f"\n[Paso 3/5] Fase 1 completada en {dur_p1:.1f} s.")

    cost_scalars_index.update(reusable)
    failed_origins = sorted(oid for oid, cdict in cost_scalars_index.items() if not cdict)
    n_computed = len(cost_scalars_index) - len(failed_origins)
    print(f"  Tracebacks guardados: {n_computed}/{len(tasks_p1)}")

    # Fallo duro: si algun worker fallo, la red resultante quedaria incompleta
    # y escribir un GPKG parcial seria enganoso. Detenemos antes de la Fase 2.
    if failed_origins:
        raise RuntimeError(
            f"FASE 1 INCOMPLETA: {len(failed_origins)} origen(es) fallaron. "
            f"IDs: {failed_origins[:20]}{'...' if len(failed_origins) > 20 else ''}. "
            "Revisa los mensajes 'ERROR Fase 1' de arriba; no se escribira ningun GPKG."
        )

    # Destinos inalcanzables (coste infinito): se informan, no bloquean, y en
    # el Paso 5 se materializan en una capa aditiva del GPKG (S8).
    pares_inf = [
        (oid, did)
        for oid, cdict in cost_scalars_index.items()
        for did, cval in cdict.items()
        if not np.isfinite(cval)
    ]
    if pares_inf:
        muestra = ", ".join(f"{oid}->{did}" for oid, did in pares_inf[:10])
        print(
            f"  ADVERTENCIA: {len(pares_inf)} par(es) origen-destino con coste "
            f"infinito (inalcanzables). Muestra: {muestra}"
            f"{'...' if len(pares_inf) > 10 else ''}."
        )

    # --- PASO 4: FASE 2 (streaming: cada lote se escribe, sin acumular en RAM) ---
    n_jobs_p2 = _calculate_n_jobs_for_phase(raster_shape, raster_dtype, phase=2)
    print(f"\n[Paso 4/5] FASE 2 — Reconstrucción en paralelo ({n_jobs_p2} workers)...")

    for origin_id, origin_coords in all_points.items():
        cost_scalars = cost_scalars_index.get(origin_id, {})
        if not cost_scalars:
            continue
        dest_items = [(did, all_points[did]) for did in cost_scalars]
        tasks_p2.append(
            (origin_id, origin_coords, dest_items, cost_scalars, transform, PHASE1_DIR)
        )

    print(f"  Total de orígenes para reconstrucción: {len(tasks_p2)}")
    t0_p2 = time.time()

    # S1: generador sin orden -> la memoria solo sostiene UN lote de origen;
    # cada lote se vuelca al GPKG temporal (modo 'a' tras crear con 'w').
    results_p2 = Parallel(
        n_jobs=n_jobs_p2, backend="loky", verbose=0, return_as="generator_unordered"
    )(delayed(postp.reconstruct_routes_from_disk)(args) for args in tasks_p2)

    if os.path.exists(gpkg_tmp_path):
        os.remove(gpkg_tmp_path)
    with tqdm(
        total=len(tasks_p2), desc="Fase 2 (rutas)", unit="origen", disable=None
    ) as pbar2:
        for _origin_id, records in results_p2:
            pbar2.update(1)
            if not records:
                continue
            gdf_lote = gpd.GeoDataFrame(records, crs=raster_crs)
            gdf_lote.to_file(
                gpkg_tmp_path,
                layer="rutas_unificadas",
                driver="GPKG",
                mode="a" if os.path.exists(gpkg_tmp_path) else "w",
            )
            n_rutas_escritas += len(records)

    dur_p2 = time.time() - t0_p2
    print(f"\n[Paso 4/5] Fase 2 completada en {dur_p2:.1f} s.")
    print(f"  Rutas reconstruidas: {n_rutas_escritas}")

    # --- PASO 5: GeoPackage (rename atomico del temporal) ---
    print("\n[Paso 5/5] Escribiendo GeoPackage unificado...")

    if not n_rutas_escritas:
        print("  ADVERTENCIA: no se generó ninguna ruta; se omite la escritura del GPKG.")
        routes_gdf = gpd.GeoDataFrame(geometry=[], crs=raster_crs)
    else:
        os.replace(gpkg_tmp_path, gpkg_output_path)
        print(f"[Paso 5/5] {n_rutas_escritas} rutas guardadas en: {gpkg_output_path}")

    # --- S8: destinos sin ruta materializados en capa aditiva ---
    # Se escribe SIEMPRE (posiblemente vacia): una re-corrida en la misma
    # sesion que ahora deja 0 pares inalcanzables no conserva filas stale.
    # mode='w' reemplaza SOLO esta capa (idempotente) y conserva las demas
    # capas del archivo; no toca el esquema de 'rutas_unificadas'.
    inf_gdf = gpd.GeoDataFrame(
        {
            "origin_id": [int(o) for o, _ in pares_inf],
            "dest_id": [int(d) for _, d in pares_inf],
            "cost_total": [float("inf")] * len(pares_inf),
        },
        geometry=gpd.points_from_xy(
            [all_points[d][0] for _, d in pares_inf],
            [all_points[d][1] for _, d in pares_inf],
        ),
        crs=raster_crs,
    )
    inf_gdf.to_file(
        gpkg_output_path,
        layer="destinos_inalcanzables",
        driver="GPKG",
        mode="w",
    )
    if pares_inf:
        print(
            f"  {len(pares_inf)} destinos inalcanzables escritos en capa 'destinos_inalcanzables'."
        )
    else:
        print("  Sin destinos inalcanzables: capa vacia.")

    print(f"  Rutas escritas: {n_rutas_escritas} | esperadas: {n_expected}")
    if n_rutas_escritas != n_expected:
        print(
            f"  ADVERTENCIA: rutas reales {n_rutas_escritas} != esperadas {n_expected}; "
            "revisa los pares con coste infinito informados arriba."
        )
    pipeline_ok = True
    total_duration = dur_p1 + dur_p2
    print("\n" + "="*60)
    print("--- RESUMEN DE EJECUCIÓN (DOS FASES) ---")
    print(f"  Fase 1 (MCP, {len(tasks_p1)} orígenes):    {dur_p1:.1f} s  ({n_jobs_p1} workers)")
    print(f"  Fase 2 (rutas, {len(tasks_p2)} orígenes):  {dur_p2:.1f} s  ({n_jobs_p2} workers)")
    print(f"  Total:                         {total_duration:.1f} s")
    print(f"  Rutas generadas:               {n_rutas_escritas}")
    print("="*60)

except Exception as e:
    print(f"\nERROR CRÍTICO en Celda 6: {e}")
    import traceback as tb
    tb.print_exc()
    raise

finally:
    # Limpieza garantizada de archivos temporales, incluso ante excepción.
    if os.path.exists(memmap_path):
        try:
            os.remove(memmap_path)
            print("  Memmap temporal eliminado.")
        except OSError as ce:
            print(f"  No se pudo eliminar el memmap: {ce}")

    if not KEEP_PHASE1_FILES and pipeline_ok and os.path.isdir(PHASE1_DIR):
        shutil.rmtree(PHASE1_DIR, ignore_errors=True)
        print("  Tracebacks de Fase 1 eliminados (KEEP_PHASE1_FILES=False).")
    elif os.path.isdir(PHASE1_DIR):
        motivo = (
            "KEEP_PHASE1_FILES=True"
            if KEEP_PHASE1_FILES
            else "ejecucion con error (evidencia conservada)"
        )
        print(f"  Tracebacks de Fase 1 conservados en: {PHASE1_DIR} ({motivo})")

    if os.path.exists(gpkg_tmp_path):
        # corrida fallida a mitad de escritura: nunca dejar un GPKG parcial
        try:
            os.remove(gpkg_tmp_path)
            print("  GPKG temporal (escritura incompleta) eliminado.")
        except OSError:
            pass


# ## Celda 7 — Mapa final de la red
# 
# Carga la red desde memoria (o el GeoPackage), armoniza CRS con el ráster, recorta la ventana con margen y dibuja rutas + nodos sobre el coste en escala logarítmica. Guarda `red_de_rutas_final.png`.

# In[ ]:


# ==============================================================================
# --- CELDA 7: VISUALIZACIÓN Y EXPORTACIÓN FINAL DE RESULTADOS ---
# ==============================================================================
print("\nIniciando la visualización final de la red de rutas...")

try:
    if 'routes_gdf' not in globals() or routes_gdf is None or len(routes_gdf) == 0:
        print("routes_gdf no disponible en memoria; cargando desde el GeoPackage...")
        routes_gdf = gpd.read_file(
            os.path.join(OUTPUT_DIR, 'red_completa_unificada.gpkg'),
            layer='rutas_unificadas'
        )

    final_routes_gdf = routes_gdf

    if len(final_routes_gdf) == 0:
        print("No hay rutas para visualizar.")
    else:
        print(f"Visualizando {len(final_routes_gdf)} rutas.")

        with rasterio.open(COST_RASTER_PATH) as src:
            raster_crs = src.crs

        if final_routes_gdf.crs is None:
            print("  ADVERTENCIA: routes_gdf sin CRS; asignando raster CRS.")
            final_routes_gdf = final_routes_gdf.set_crs(raster_crs)
        elif final_routes_gdf.crs != raster_crs:
            print(f"  Reproyectando rutas de {final_routes_gdf.crs} → {raster_crs}")
            final_routes_gdf = final_routes_gdf.to_crs(raster_crs)

        points_gdf = gpd.read_file(ALL_POINTS_SHAPEFILE)
        if points_gdf.crs is None:
            points_gdf = points_gdf.set_crs(raster_crs)
        elif points_gdf.crs != raster_crs:
            points_gdf = points_gdf.to_crs(raster_crs)

        minx, miny, maxx, maxy = final_routes_gdf.total_bounds
        margin = (maxx - minx) * 0.1
        buffered_bounds = (minx - margin, miny - margin, maxx + margin, maxy + margin)

        with rasterio.open(COST_RASTER_PATH) as src:
            import math
            window = from_bounds(*buffered_bounds, transform=src.transform)
            col_off = int(math.floor(window.col_off))
            row_off = int(math.floor(window.row_off))
            col_stop = int(math.ceil(window.col_off + window.width))
            row_stop = int(math.ceil(window.row_off + window.height))
            window = rasterio.windows.Window(
                col_off, row_off, col_stop - col_off, row_stop - row_off
            )
            # Recortar la ventana a los límites reales del ráster para que
            # window_transform coincida con los datos efectivamente leídos.
            full = rasterio.windows.Window(0, 0, src.width, src.height)
            window = window.intersection(full)
            raster_data = src.read(1, window=window)
            window_transform = src.window_transform(window)

        fig, ax = plt.subplots(figsize=(18, 18))

        min_pos_val = np.min(raster_data[raster_data > 0]) if np.any(raster_data > 0) else 1e-9
        raster_data_log = np.where(raster_data <= 0, min_pos_val, raster_data)

        from rasterio.plot import plotting_extent
        extent = plotting_extent(raster_data_log, window_transform)
        ax.imshow(raster_data_log, extent=extent, origin='upper',
                  cmap='viridis', norm=LogNorm(), alpha=0.8)

        final_routes_gdf.plot(ax=ax, edgecolor='red', linewidth=0.8,
                              label='Rutas Calculadas', alpha=0.6, zorder=3)
        _node_xs = [xy[0] for xy in all_points_for_validation.values()]
        _node_ys = [xy[1] for xy in all_points_for_validation.values()]
        ax.scatter(_node_xs, _node_ys, color='yellow', s=50, edgecolor='black',
                   label='Nodos de la Red', zorder=5)

        # Resaltar el origen en modo single_origin (S5; seleccion simple, no interactiva)
        if (
            "ANALYSIS_MODE" in globals()
            and ANALYSIS_MODE == "single_origin"
            and "all_points_for_validation" in globals()
            and ORIGIN_ID in all_points_for_validation
        ):
            _ox, _oy = all_points_for_validation[ORIGIN_ID]
            ax.scatter([_ox], [_oy], c='lime', edgecolor='black', s=400,
                       marker="*", zorder=6, label=f'Origen {ORIGIN_ID}')
            ax.legend()

        ax.set_title('Red Completa de Rutas de Menor Costo', fontsize=20)
        ax.set_xlabel("Coordenada X"); ax.set_ylabel("Coordenada Y")
        ax.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        output_image_path = os.path.join(OUTPUT_DIR, 'red_de_rutas_final.png')
        fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualización final guardada en: {output_image_path}")

        plt.show()
except Exception as e:
    print(f"Ocurrió un error durante la visualización: {e}")
    import traceback as tb
    tb.print_exc()


# ## Celda 8 — Puntos densificados sobre las rutas
# 
# Coloca un punto cada `SPACING_M` metros (25 m por defecto) sobre cada ruta por interpolación vectorizada sobre la longitud acumulada. Guarda `puntos_rutas_<espaciado>m.gpkg` con la columna `route_idx` para trazabilidad. Es la entrada de la Celda 9.

# In[ ]:


# ==============================================================================
# --- CELDA 8: PUNTOS DENSIFICADOS (PYOGRIO DIRECTO, SIN SHAPELY POINTS) ---
# ==============================================================================
import os, time, gc
import numpy as np
import geopandas as gpd          # <-- movido al encabezado (lo usan TODAS las ramas)
import shapely
from shapely import get_coordinates
from joblib import Parallel, delayed

SPACING_M = 25.0
CHUNK_SIZE = 500
import psutil
N_WORKERS = min(18, psutil.cpu_count(logical=True) or 1)

t0 = time.time()
print(f"\n{'='*60}")
print(f"  CELDA 8: PUNTOS CADA {SPACING_M} m — FIONA DIRECTO")
print(f"{'='*60}")

# ── 1. AUTO-DETECTAR RUTAS ──
def _find_latest_gpkg(root):
    out = os.path.join(root, 'output')
    if not os.path.isdir(out): return None, None
    sessions = sorted(
        [d for d in os.listdir(out) if d.startswith('session_')],
        key=lambda s: os.path.getmtime(os.path.join(out, s)), reverse=True
    )
    for s in sessions:
        d = os.path.join(out, s)
        gpkg = os.path.join(d, 'red_completa_unificada.gpkg')
        if os.path.exists(gpkg): return gpkg, d
    return None, None

if 'routes_gdf' in globals() and routes_gdf is not None and len(routes_gdf) > 0:
    pass
elif 'OUTPUT_DIR' in globals() and os.path.exists(p := os.path.join(OUTPUT_DIR, 'red_completa_unificada.gpkg')):
    routes_gdf = gpd.read_file(p, layer='rutas_unificadas')
else:
    gpkg, sess_dir = _find_latest_gpkg(os.getcwd())
    if gpkg is None:
        raise FileNotFoundError("red_completa_unificada.gpkg no encontrado")
    OUTPUT_DIR = sess_dir
    routes_gdf = gpd.read_file(gpkg, layer='rutas_unificadas')
    print(f"  Última sesión: {os.path.basename(sess_dir)}")

n_routes = len(routes_gdf)
print(f"  [1/5] Rutas: {n_routes:,}")

# ── 2. PRE-CÓMPUTO ──
lengths = shapely.length(routes_gdf.geometry.values)
n_pts_arr = np.ceil(lengths / SPACING_M).astype(np.int32) + 1
total_pts = int(n_pts_arr.sum())
print(f"  [2/5] Longitud: {lengths.sum()/1000:,.0f} km | Puntos: {total_pts:,}")

# ── 3. COORDENADAS + CHUNKS ──
print(f"  [3/5] Extrayendo coordenadas...")
coords_list = [get_coordinates(g) for g in routes_gdf.geometry.values]
_crs_rutas = routes_gdf.crs
del routes_gdf; gc.collect()

chunks = [
    (np.arange(i, min(i+CHUNK_SIZE, n_routes), dtype=np.int32),
     coords_list[i:i+CHUNK_SIZE], lengths[i:i+CHUNK_SIZE],
     n_pts_arr[i:i+CHUNK_SIZE], float(SPACING_M))
    for i in range(0, n_routes, CHUNK_SIZE)
]
del coords_list, lengths, n_pts_arr; gc.collect()
print(f"        {len(chunks)} lotes × {N_WORKERS} workers")

# ── 4. WORKER (puro numpy, sin GEOS) ──
def densify_chunk(indices, coords_list, lengths, n_pts_arr, spacing):
    chunk_x, chunk_y, chunk_idx = [], [], []
    for idx, coords, length, n in zip(indices, coords_list, lengths, n_pts_arr):
        n = int(n)
        # --- Guarda de ruta degenerada: longitud 0 o un solo vértice ---
        if length <= 0 or len(coords) < 2:
            # Ruta degenerada: sin geometria utilizable no generamos puntos
            # (antes escribia un punto ficticio en (0, 0) del mundo).
            if len(coords) == 0:
                continue
            x0 = coords[0, 0]
            y0 = coords[0, 1]
            chunk_x.append(np.full(n, x0))
            chunk_y.append(np.full(n, y0))
            chunk_idx.append(np.full(n, idx, dtype=np.int32))
            continue

        dists = np.linspace(0, float(length), n)
        seg_vecs = np.diff(coords, axis=0)
        seg_lens = np.sqrt((seg_vecs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0], seg_lens.cumsum()])
        total = float(cum_len[-1])
        if total <= 0:   # coords duplicadas con length>0 espurio: degenerada
            chunk_x.append(np.full(n, coords[0, 0]))
            chunk_y.append(np.full(n, coords[0, 1]))
            chunk_idx.append(np.full(n, idx, dtype=np.int32))
            continue

        cum_norm = cum_len / total
        i = np.searchsorted(cum_norm, dists / float(length), side='right') - 1
        i = np.clip(i, 0, len(coords) - 2)
        t = (dists - cum_len[i]) / (seg_lens[i] + 1e-30)
        chunk_x.append(coords[i, 0] + t * seg_vecs[i, 0])
        chunk_y.append(coords[i, 1] + t * seg_vecs[i, 1])
        chunk_idx.append(np.full(n, idx, dtype=np.int32))
    if not chunk_x:
        # Lote completo de rutas sin geometria utilizable: sin puntos que anadir
        # (evita ValueError de np.concatenate sobre lista vacia).
        return (np.empty(0), np.empty(0), np.empty(0, dtype=np.int32))
    return (np.concatenate(chunk_x), np.concatenate(chunk_y),
            np.concatenate(chunk_idx))

# ── 5. EJECUTAR (streaming: cada lote se escribe, sin concat global) ──
t1 = time.time()
print(f"  [4/5] Interpolando {len(chunks)} lotes...")
results = Parallel(
    n_jobs=N_WORKERS, verbose=10, prefer="processes", return_as="generator_unordered"
)(delayed(densify_chunk)(*c) for c in chunks)

# ── 6. ESCRIBIR POR LOTE DIRECTO (S2 memoria acotada; S4 pyogrio ~2,9x) ──
print("  [5/5] Escribiendo puntos lote a lote (pyogrio, sin concatenar)...")
output_path = os.path.join(OUTPUT_DIR, f'puntos_rutas_{int(SPACING_M)}m.gpkg')
layer_name = os.path.splitext(os.path.basename(output_path))[0]
crs_wkt = None if _crs_rutas is None else str(_crs_rutas)
n_total = total_pts
escritos = 0

for xs_c, ys_c, idx_c in results:
    n_c = len(xs_c)
    if n_c == 0:
        continue
    gdf_chunk = gpd.GeoDataFrame(
        {"route_idx": np.asarray(idx_c, dtype=np.int64)},
        geometry=gpd.points_from_xy(xs_c, ys_c),
        crs=crs_wkt,
    )
    # capa = stem del archivo (el mismo default que usaba fiona); 'w' crea en
    # el primer lote y 'a' agrega los siguientes (pyogrio -> OGR en C++).
    gdf_chunk.to_file(
        output_path,
        layer=layer_name,
        driver="GPKG",
        mode="a" if os.path.exists(output_path) else "w",
    )
    escritos += n_c
    if n_total:
        print(
            f"    Escritos {escritos:,} / {n_total:,} ({escritos * 100 // n_total}%)",
            end='\r',
        )
t2 = time.time()
t3 = t2  # interp. y escritura se solapan (streaming): no hay fase de concatenacion

if escritos != n_total:
    print(f"  ADVERTENCIA: puntos escritos {escritos:,} != esperados {n_total:,}")

gpkg_mb = os.path.getsize(output_path) / 1e6
print(f"\n\n{'─'*60}")
print(f"  {escritos:,} puntos escritos (esperados {n_total:,}) | {gpkg_mb:.0f} MB")
print(f"  {output_path}")
print(f"  Interp.+escritura: {t2-t1:.1f} s | TOTAL: {t3-t0:.1f} s")
print(f"{'─'*60}")


# ## Celda 9 — Mapa de calor (KDE estilo QGIS)
# 
# Estimación de densidad sobre los puntos densificados: grilla con margen = bandwidth, binning numpy, kernel QGIS (`HEATMAP_KERNEL`, quartic por defecto) y convolución FFT con recorte "same". La lógica vive en `_compute_heatmap` (con tests de regresión: pico, georreferenciación y masa). Salidas: GeoTIFF + PNG. Parámetros: `HEATMAP_RES_M`, `HEATMAP_BANDWIDTH_M`, `HEATMAP_KERNEL`.

# In[ ]:


# ==============================================================================
# --- CELDA 9: MAPA DE CALOR ESTILO QGIS (KERNEL QUARTIC / BIWEIGHT) ---
# ==============================================================================
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fft import rfft2, irfft2
from shapely import get_coordinates

# Parámetros configurables
HEATMAP_RES_M = 30          # Resolución espacial del raster (m/px)
HEATMAP_BANDWIDTH_M = 1000 # Radio de búsqueda / bandwidth (metros)
HEATMAP_KERNEL = "quartic"  # Kernel de QGIS: "quartic", "gaussian", "triangular", "uniform"

print(f"\nGenerando mapa de calor (QGIS-style)...")
print(f"  Resolución: {HEATMAP_RES_M} m/px")
print(f"  Bandwidth:  {HEATMAP_BANDWIDTH_M / 1000:.1f} km")
print(f"  Kernel:     {HEATMAP_KERNEL}")

# Resuelve el mismo espaciado que la Celda 8; default para ejecucion suelta de la celda.
SPACING_M = globals().get('SPACING_M', 25.0)
_PUNTOS_NAME = f'puntos_rutas_{int(SPACING_M)}m.gpkg'
if 'OUTPUT_DIR' not in globals() or not os.path.exists(os.path.join(OUTPUT_DIR, _PUNTOS_NAME)):
    def _find_latest(out):
        if not os.path.isdir(out): return None
        sessions = sorted([d for d in os.listdir(out) if d.startswith('session_')],
                         key=lambda s: os.path.getmtime(os.path.join(out, s)), reverse=True)
        for s in sessions:
            d = os.path.join(out, s)
            if os.path.exists(os.path.join(d, _PUNTOS_NAME)): return d
        return None
    sess_dir = _find_latest(os.path.join(os.getcwd(), 'output'))
    if sess_dir is None:
        raise FileNotFoundError(f'No se encontro sesion con {_PUNTOS_NAME}')
    OUTPUT_DIR = sess_dir
    print(f'  OUTPUT_DIR auto-detectado: {os.path.basename(OUTPUT_DIR)}')

points_path = os.path.join(OUTPUT_DIR, _PUNTOS_NAME)
points_gdf = gpd.read_file(points_path)

def _compute_heatmap(coords, minx, miny, maxx, maxy, res, bandwidth, kernel_name):
    """Binning + kernel QGIS + convolucion FFT -> (heatmap, pixel_size_x, pixel_size_y).

    Extraido de la Celda 9 sin cambios matematicos para poder testear la
    regresion KDE/FFT (pico del punto, georreferenciacion y masa del kernel).
    coords: array (N,2) en coordenadas de mundo; los bounds definen la grilla
    (el caller agrega el margen = bandwidth).
    """
    # --- Grilla: el tamano del pixel se deriva de los bounds (NO de res) ---
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    pixel_size_x = (maxx - minx) / width
    pixel_size_y = (maxy - miny) / height

    # --- 2. Numpy binning vectorizado (usa el MISMO pixel_size del raster) ---
    cols = ((coords[:, 0] - minx) / pixel_size_x).astype(np.int32)
    rows = ((maxy - coords[:, 1]) / pixel_size_y).astype(np.int32)
    valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)

    count_grid = np.zeros((height, width), dtype=np.float64)
    np.add.at(count_grid, (rows[valid], cols[valid]), 1)

    # --- 3. Kernel de QGIS (usando pixel_size real, no res) ---
    ps = (pixel_size_x + pixel_size_y) / 2.0
    radius_px = int(np.ceil(bandwidth / ps))
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    dist_px = np.sqrt(x**2 + y**2)
    u = (dist_px * ps) / bandwidth

    if kernel_name == "quartic":
        kernel = np.where(u <= 1, (15.0 / 16.0) * (1 - u**2) ** 2, 0.0)
    elif kernel_name == "gaussian":
        kernel = np.exp(-0.5 * u**2)
    elif kernel_name == "triangular":
        kernel = np.where(u <= 1, 1 - u, 0.0)
    elif kernel_name == "uniform":
        kernel = np.where(u <= 1, 1.0, 0.0)
    else:
        raise ValueError(f"Kernel no soportado: {kernel_name}")

    kernel_area = np.sum(kernel) * (ps**2)
    kernel = kernel / kernel_area

    # --- 4. Convolucion via FFT (scipy.fft) + recorte "same" ---
    count_padded = np.pad(count_grid, pad_width=radius_px, mode="constant")
    kernel_padded = np.zeros_like(count_padded)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel

    fft_grid = rfft2(count_padded)
    fft_kernel = rfft2(kernel_padded)
    fft_result = fft_grid * fft_kernel
    heatmap = irfft2(fft_result, s=count_padded.shape)

    heatmap = heatmap[
        2 * radius_px : 2 * radius_px + height, 2 * radius_px : 2 * radius_px + width
    ]
    return heatmap, pixel_size_x, pixel_size_y


# --- 1. Extensión con margen = bandwidth (evita efectos de borde) ---
minx, miny, maxx, maxy = points_gdf.total_bounds
margin = HEATMAP_BANDWIDTH_M
minx -= margin; miny -= margin; maxx += margin; maxy += margin

# --- 2-4. Grilla + binning + kernel + FFT: extraido a _compute_heatmap ---
# (mismas formulas; ver el docstring. Los prints de resultado quedan abajo.)
print("  Rasterizando puntos via numpy binning...")
print("  Aplicando kernel via FFT...")
coords = get_coordinates(points_gdf.geometry)
heatmap, pixel_size_x, pixel_size_y = _compute_heatmap(
    coords, minx, miny, maxx, maxy,
    HEATMAP_RES_M, HEATMAP_BANDWIDTH_M, HEATMAP_KERNEL,
)
width, height = heatmap.shape[1], heatmap.shape[0]
transform = from_bounds(minx, miny, maxx, maxy, width, height)
ps = (pixel_size_x + pixel_size_y) / 2.0
radius_px = int(np.ceil(HEATMAP_BANDWIDTH_M / ps))

print(f"  Grid: {width} × {height} celdas ({width*height/1e6:.1f} Mpx)")
print(f"  Pixel real: {pixel_size_x:.4f} × {pixel_size_y:.4f} m")
print(f"  Kernel size: {2*radius_px+1} × {2*radius_px+1} px")

# --- 5. Guardar GeoTIFF ---
heatmap_tif = os.path.join(OUTPUT_DIR, f'heatmap_{HEATMAP_RES_M}m_{int(HEATMAP_BANDWIDTH_M/1000)}km_{HEATMAP_KERNEL}.tif')
with rasterio.open(
    heatmap_tif,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype=heatmap.dtype,
    crs=points_gdf.crs,
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(heatmap, 1)
    dst.update_tags(
        kernel=HEATMAP_KERNEL,
        bandwidth_m=HEATMAP_BANDWIDTH_M,
        resolution_m=HEATMAP_RES_M,
        method="QGIS-style KDE via FFT"
    )

print(f"  GeoTIFF: {heatmap_tif}")

# --- 6. Visualizar ---
fig, ax = plt.subplots(figsize=(18, 18))
extent = [minx, maxx, miny, maxy]
im = ax.imshow(heatmap, extent=extent, origin='upper', cmap='hot')
ax.set_title(f'Mapa de Calor - Kernel {HEATMAP_KERNEL.capitalize()} (BW: {HEATMAP_BANDWIDTH_M/1000:.0f} km)', fontsize=20)
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Densidad (rutas / km²)', rotation=270, labelpad=20)

plt.tight_layout()
heatmap_png = os.path.join(OUTPUT_DIR, f'heatmap_{HEATMAP_RES_M}m_{int(HEATMAP_BANDWIDTH_M/1000)}km_{HEATMAP_KERNEL}.png')
fig.savefig(heatmap_png, dpi=300, bbox_inches='tight')
print(f"  PNG:     {heatmap_png}")
plt.show()

