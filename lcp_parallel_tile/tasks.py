# lcp_network/tasks.py

import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

from . import geoutils

def get_node_id(r, c, raster_width):
    return r * raster_width + c

def prepare_tasks(config, log):
    """
    Divide el raster en tareas, generando nodos de borde SOLO si caen
    dentro de la máscara vectorial proporcionada.
    """
    log.info("Iniciando Fase 1: Preparación de tareas de tiling y nodos de borde...")
    
    # --- [MODIFICACIÓN] Cargar la máscara primero ---
    mask_geom = geoutils.load_polygon_geometry(config.MASK_SHAPEFILE_PATH, log)
    
    tasks = []
    with rasterio.open(config.COST_RASTER_PATH) as src:
        raster_width, raster_height = src.width, src.height
        transform = src.transform
        
        for r_offset in range(0, raster_height, config.TILE_SIZE):
            for c_offset in range(0, raster_width, config.TILE_SIZE):
                width = min(config.TILE_SIZE, raster_width - c_offset)
                height = min(config.TILE_SIZE, raster_height - r_offset)
                window = Window(c_offset, r_offset, width, height)
                
                border_nodes = {}
                # Generar nodos candidatos
                candidate_pixels = []
                for c in range(0, width, config.NODE_SPACING):
                    for r in [0, height - 1]: candidate_pixels.append((r, c))
                for r in range(config.NODE_SPACING, height - config.NODE_SPACING, config.NODE_SPACING):
                    for c in [0, width - 1]: candidate_pixels.append((r, c))

                # Validar nodos contra la máscara si existe
                for r_local, c_local in candidate_pixels:
                    r_global, c_global = r_offset + r_local, c_offset + c_local
                    
                    is_valid = True
                    if mask_geom:
                        # Convertir pixel a coordenadas del mundo para la comprobación
                        x_world, y_world = transform * (c_global + 0.5, r_global + 0.5)
                        point_geom = Point(x_world, y_world)
                        if not mask_geom.contains(point_geom):
                            is_valid = False
                    
                    if is_valid:
                        node_id = get_node_id(r_global, c_global, raster_width)
                        border_nodes[node_id] = (r_local, c_local)

                if len(border_nodes) > 1: # Solo añadir tiles que tengan al menos 2 nodos válidos
                    tasks.append({'tile_id': f"tile_{r_offset//config.TILE_SIZE}_{c_offset//config.TILE_SIZE}", 'window': window, 'border_nodes': border_nodes})

    log.info(f"Preparación completa. Se generaron {len(tasks)} tareas de tile con nodos válidos.")
    return tasks