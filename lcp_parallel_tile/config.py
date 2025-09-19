# lcp_parallel_tile/config.py

import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.getcwd()
BASE_PROCESSING_FOLDER = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')

# --- [MODIFICACIÓN] Añadir la ruta a la máscara ---
# Establecer en None para desactivar el uso de la máscara.
MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')

# --- 2. PARÁMETROS DE TILING Y NODOS ---
TILE_SIZE = 2000
NODE_SPACING = 200

# --- 3. PARÁMETROS DE CÁLCULO JERÁRQUICO ---
DOWNSAMPLING_FACTORS = [32, 20, 10]
CORRIDOR_BUFFER_PIXELS = 150
HEURISTIC_WEIGHT = 1.0

# --- 4. PARÁMETROS DE EJECUCIÓN ---
N_JOBS = -2