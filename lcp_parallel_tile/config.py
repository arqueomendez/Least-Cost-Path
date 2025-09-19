# lcp_parallel_tile/config.py

import os

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.getcwd()
BASE_PROCESSING_FOLDER = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
COST_RASTER_PATH = os.path.join(DATA_DIR, 'cost.tif')
MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'area-mask.shp')

# --- 2. PARÁMETROS DE TILING Y NODOS ---
TILE_SIZE = 2000
NODE_SPACING = 200

# --- [NUEVO PARÁMETRO] ---
# Define un borde de exclusión dentro de cada tile para forzar las rutas
# a calcularse hacia el interior, eliminando los artefactos de los bordes.
# Un valor de 1 o 2 píxeles suele ser suficiente.
TILE_EDGE_BUFFER = 2

# --- 3. PARÁMETROS DE CÁLCULO JERÁRQUICO ---
DOWNSAMPLING_FACTORS = [32, 20, 10]
CORRIDOR_BUFFER_PIXELS = 150
HEURISTIC_WEIGHT = 1.0

# --- 4. PARÁMETROS DE EJECUCIÓN ---
N_JOBS = -2