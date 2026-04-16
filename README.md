# LCP_MCP_Geometric

## Descripción

LCP_MCP_Geometric es una herramienta de análisis espacial para el cálculo de rutas de menor coste entre múltiples puntos sobre superficies raster. Utiliza el algoritmo `MCP_Geometric` de scikit-image para propagar una superficie de coste acumulado desde cada origen y, a continuación, reconstruye los caminos óptimos mediante traceback con Numba JIT. Ha sido desarrollado para aplicaciones arqueológicas, con el objetivo de modelar movimientos en el territorio a partir de un ráster de fricción o coste.

La herramienta calcula el conjunto completo de rutas entre todos los pares de puntos de una red (análisis N×N), almacenando únicamente el triángulo superior de la matriz OD para evitar duplicar rutas simétricas.

## Características principales

- Implementa `MCP_Geometric` (scikit-image) como motor de propagación de costes, con soporte de conectividad completa (8 vecinos).
- Reconstrucción del traceback acelerada con Numba JIT (`@njit`).
- Análisis N×N deduplicado: para N puntos se calculan N×(N-1)/2 rutas únicas (triángulo superior de la matriz OD), evitando recalcular pares simétricos.
- **Ejecución paralela con joblib** (`loky` backend): cada origen se procesa en un worker independiente, con el ráster de coste compartido mediante un archivo `memmap` en disco. Con 16 núcleos y N=78 puntos, la ganancia teórica es ~15×.
- Permite restringir el área de búsqueda a un polígono vectorial (shapefile o GeoPackage) para excluir zonas no accesibles.
- Acumula todos los resultados en memoria y escribe un único GeoPackage al final, con atributos enriquecidos por ruta.
- Verbosidad configurable mediante el parámetro `VERBOSE`.
- Incluye celdas de validación de datos de entrada y visualización diagnóstica.

## Estructura del proyecto

### Carpetas principales

- **`data/`**: Archivos de entrada (ráster de coste, shapefile/GeoPackage de puntos, máscara poligonal).
- **`output/`**: Resultados de cada sesión, organizados por marca de tiempo. Contiene el GeoPackage unificado de rutas.
- **`lcp/`**: Módulo principal con la lógica del proyecto:
  - `data_loader.py`: Carga el ráster, los puntos y la máscara vectorial.
  - `processing.py`: Procesamiento raster, creación de máscaras y conversión de coordenadas mundo↔píxel.
  - `pathfinder.py`: Motor de coste (`MCP_Geometric`) y reconstrucción de traceback con Numba.
  - `postprocessing.py`: Extrae todas las rutas desde un origen dado, filtrando pares ya calculados (`dest_id > origin_id`).
  - `utils.py`: Construcción del registro de ruta (`build_route_record`) y guardado legacy a shapefile.
  - `__init__.py`: Inicialización del paquete.

### Archivos principales

- **`LCP_MCP_Geometric.ipynb`**: Notebook principal de ejecución paso a paso, visualización y exportación de resultados.
- **`pyproject.toml`**: Configuración y dependencias del proyecto.
- **`uv.lock`**: Bloqueo de versiones de dependencias.
- **`requirements.txt`**: Lista de dependencias en formato pip.
- **`LICENSE`**: Licencia de uso (CC BY-NC 4.0).
- **`CITATION.cff`**: Metadatos de citación.
- **`README.md`**: Este archivo.

## Formato de salida

Cada ruta calculada se almacena como un registro con los siguientes atributos:

| Campo        | Tipo    | Descripción                                              |
|--------------|---------|----------------------------------------------------------|
| `geometry`   | LineString | Geometría de la ruta en el CRS del proyecto           |
| `origin_id`  | entero  | ID del punto de origen (siempre el menor del par)        |
| `dest_id`    | entero  | ID del punto de destino (siempre el mayor del par)       |
| `cost_total` | real    | Coste acumulado MCP en el píxel de destino               |
| `length_m`   | real    | Longitud de la ruta en metros (CRS proyectado)           |
| `n_vertices` | entero  | Número de vértices del camino reconstruido               |

Todos los registros se exportan en un único archivo **`red_completa_unificada.gpkg`** dentro de la carpeta de sesión.

## Método de trabajo

El flujo implementado en `LCP_MCP_Geometric.ipynb` sigue estos pasos:

### Celda 1 — Importaciones
Carga de librerías: `numpy`, `rasterio`, `fiona`, `geopandas`, `scikit-image`, `numba`, `tqdm`, y los módulos del paquete `lcp`.

### Celda 2 — Definición de funciones
Define `lcp_worker_one_to_all` (worker para ejecución paralela con joblib) y `warm_up_numba` (primer llamado a la función Numba JIT para compilarla antes del bucle principal).

### Celda 3 — Configuración central
Parámetros editables por el usuario:

| Parámetro            | Descripción                                                       |
|----------------------|-------------------------------------------------------------------|
| `COST_RASTER_PATH`   | Ruta al ráster de coste/fricción (`.tif`)                        |
| `ALL_POINTS_SHAPEFILE` | Shapefile o GeoPackage con todos los puntos de la red          |
| `MASK_SHAPEFILE_PATH`  | Polígono de restricción espacial (`None` para desactivar)      |
| `ID_FIELD_NAME`      | Nombre del campo ID en la capa de puntos                         |
| `OUTPUT_DIR`         | Carpeta de salida (generada automáticamente con marca de tiempo) |
| `N_JOBS`             | Número de núcleos para el worker paralelo (`-2` = todos menos 1) |
| `VERBOSE`            | `True` imprime detalles por ruta; `False` suprime el ruido       |

### Celda 4 — Verificación de coherencia de datos
Valida que todos los puntos de entrada se encuentren dentro del área de búsqueda definida por la máscara. Si algún punto queda fuera, la ejecución se detiene con un mensaje de error.

### Celda 5 — Visualización diagnóstica
Genera un mapa del ráster de coste con los puntos superpuestos para verificar visualmente la configuración antes de lanzar el análisis.

### Celda 6 — Ejecución principal (modo serie)
1. Carga todos los datos en memoria una única vez.
2. Aplica la máscara vectorial al ráster (celdas fuera del polígono → `np.inf`).
3. Compila la función Numba JIT con `warm_up_numba()`.
4. Itera sobre cada punto origen; omite los que no tienen destinos con `dest_id > origin_id` (deduplicación).
5. Para cada origen activo: llama a `pf.cost_surface_calculator()` → `postp.extract_routes()` → acumula registros en `all_route_records`.
6. Construye un `GeoDataFrame` y escribe el GeoPackage unificado.

### Celda 7 — Visualización y exportación final
Carga el GeoPackage desde memoria (o disco si es necesario) y genera un mapa final de la red de rutas calculadas, con escala y norte.

## Dependencias

El proyecto utiliza las siguientes dependencias principales (ver `pyproject.toml`):

- `fiona`
- `geopandas`
- `jupyterlab`
- `matplotlib-scalebar`
- `numba`
- `numpy`
- `pandas`
- `rasterio`
- `scikit-image`
- `shapely`
- `tqdm`
- `joblib`

## Instalación y uso con UV y Jupyter Lab

1. Instala [uv](https://github.com/astral-sh/uv) si no lo tienes:

   a. Con pip:

   - **Windows**:
     ```sh
     pip install uv
     ```
   - **macOS**:
     ```sh
     brew install uv
     # o
     pip install uv
     ```
   - **Linux**:
     ```sh
     pipx install uv
     # o
     pip install uv
     ```

   b. Standalone:

   - **Windows**:
     ```sh
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **macOS y Linux**:
     ```sh
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

2. Instala todas las dependencias del proyecto:

   - **Con uv**:
     ```sh
     uv sync
     ```
   - **Con pip**:
     ```sh
     pip install -r requirements.txt
     ```

3. Inicia Jupyter Lab:
   ```sh
   jupyter lab
   ```

4. Abre y ejecuta el notebook `LCP_MCP_Geometric.ipynb`.

## Uso básico

1. Prepara los archivos de entrada:
   - Un ráster de coste/fricción (`*.tif`).
   - Un shapefile o GeoPackage con todos los puntos de la red (`*.shp` / `*.gpkg`).
   - **Opcional**: Un shapefile con el polígono de restricción espacial (`*.shp`).

2. Abre `LCP_MCP_Geometric.ipynb` en JupyterLab.

3. Ajusta los parámetros en **Celda 3**:
   - `COST_RASTER_PATH`: ruta a tu ráster de coste.
   - `ALL_POINTS_SHAPEFILE`: ruta a tus puntos.
   - `MASK_SHAPEFILE_PATH`: ruta al polígono de máscara, o `None`.
   - `ID_FIELD_NAME`: nombre del campo ID en la capa de puntos.

4. Ejecuta las celdas en orden (1 → 7).

5. Los resultados se guardan en `output/session_<timestamp>/red_completa_unificada.gpkg`.

## Notas técnicas

- El número de rutas calculadas para N puntos es N×(N-1)/2 gracias a la deduplicación. Para N=78 puntos: 3 003 rutas.
- `MCP_Geometric` emplea un modelo de coste simétrico: el coste de A→B es idéntico al de B→A, por lo que almacenar ambas direcciones sería redundante.
- La función `reconstruct_path_from_traceback` se compila con `@njit` de Numba la primera vez que se ejecuta. `warm_up_numba()` fuerza esta compilación antes del bucle principal para que no penalice el primer origen real.
- La variable `save_path_to_shapefile` en `utils.py` se mantiene por compatibilidad con código anterior; el flujo principal utiliza `build_route_record`.
- El ráster de coste debe estar en un CRS proyectado (p. ej. UTM) para que `length_m` sea significativo.

## Licencia y atribución

Este software se distribuye bajo una licencia de atribución, uso no comercial y compartida:

- Puedes modificar y compartir el código libremente, siempre y cuando:
  - No lo utilices con fines comerciales ni en productos o servicios comerciales.
  - Mantengas esta licencia y la sección de atribución en cualquier copia o derivado.
  - Cites explícitamente el siguiente texto en cualquier uso, publicación o derivado:

    > "Least-Cost-Path (LCP) desarrollado por Víctor Méndez, 2025."

- El uso en investigación, docencia y proyectos personales está permitido.
- Para cualquier uso comercial, se debe solicitar autorización expresa al autor.

## Autoría

Desarrollado por Víctor Méndez, con asistencia de Claude (Anthropic).

## How to Cite

If you use this software in your research, please cite it as follows:

```bibtex
@software{Mendez_Least_Cost_Path,
  author = {Méndez, Víctor},
  title  = {{LCP\_MCP\_Geometric: A Python tool for Least Cost Path analysis in archaeology}},
  url    = {https://github.com/arqueomendez/Least-Cost-Path},
  year   = {2025}
}
```
