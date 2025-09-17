# lcp/utils.py

import fiona
import os
from fiona.crs import CRS
from shapely.geometry import LineString, mapping

def save_path_to_shapefile(pixel_path, transform, crs, output_path):
    """
    Guarda una ruta de píxeles en un archivo shapefile.
    Fiel a la implementación original.
    """
    if pixel_path is None or len(pixel_path) == 0:
        print(f"No se guardará {os.path.basename(output_path)}, la ruta está vacía o es inválida.")
        return
        
    # Convierte el centro de cada píxel a coordenadas del mundo
    world_coords = [transform * (p[1] + 0.5, p[0] + 0.5) for p in pixel_path]
    
    if len(world_coords) < 2:
        print(f"No se guardará {os.path.basename(output_path)}, la ruta necesita al menos 2 puntos.")
        return

    schema = {'geometry': 'LineString', 'properties': {'id': 'str'}}
    
    # Asegurar que el CRS se maneje correctamente
    fiona_crs = CRS.from_wkt(crs.to_wkt()) if crs else None

    with fiona.open(output_path, 'w', 'ESRI Shapefile', schema, crs=fiona_crs) as c:
        c.write({
            'geometry': mapping(LineString(world_coords)),
            'properties': {'id': os.path.basename(output_path)}
        })
    print(f"Ruta guardada exitosamente en: {os.path.basename(output_path)}")