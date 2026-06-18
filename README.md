# LCP_MCP_Geometric

## Descripción

LCP_MCP_Geometric es una herramienta de análisis espacial para el cálculo de rutas de menor coste entre múltiples puntos sobre superficies raster. Utiliza el algoritmo `MCP_Geometric` de scikit-image para propagar una superficie de coste acumulado desde cada origen y, a continuación, reconstruye los caminos óptimos mediante traceback con Numba JIT. Ha sido desarrollado para aplicaciones arqueológicas, con el objetivo de modelar movimientos en el territorio a partir de un ráster de fricción o coste.

La herramienta calcula el conjunto completo de rutas entre todos los pares de puntos de una red (análisis N×N), almacenando únicamente el triángulo superior de la matriz OD para evitar duplicar rutas simétricas.

## Características principales

- Implementa `MCP_Geometric` (scikit-image) como motor de propagación de costes, con soporte de conectividad completa (8 vecinos).
- Reconstrucción del traceback acelerada con Numba JIT (`@njit`).
- Análisis N×N deduplicado: para N puntos se calculan N×(N-1)/2 rutas únicas (triángulo superior de la matriz OD), evitando recalcular pares simétricos.
- **Ejecución paralela en dos fases con joblib** (`loky` backend): el análisis se divide en Fase 1 (calcular y guardar superficies de coste por origen) y Fase 2 (reconstruir rutas desde disco). El número de workers se auto-detecta según la RAM disponible para cada fase. El ráster se comparte mediante un archivo `memmap` en disco.
- Permite restringir el área de búsqueda a un polígono vectorial (shapefile o GeoPackage) para excluir zonas no accesibles.
- Acumula todos los resultados en memoria y escribe un único GeoPackage al final, con atributos enriquecidos por ruta.
- Verbosidad configurable mediante el parámetro `VERBOSE`.
- Incluye celdas de validación de datos de entrada y visualización diagnóstica.
- **Post-procesamiento**: generación de puntos densificados sobre rutas (Celda 8) y mapas de calor tipo QGIS (Celda 9).

## Estructura del proyecto

### Carpetas principales

- **`data/`**: Archivos de entrada (ráster de coste, shapefile/GeoPackage de puntos, máscara poligonal).
- **`output/`**: Resultados de cada sesión, organizados por marca de tiempo. Contiene el GeoPackage unificado de rutas.
- **`lcp/`**: Módulo principal con la lógica del proyecto:
  - `data_loader.py`: Carga el ráster, los puntos y la máscara vectorial.
  - `processing.py`: Procesamiento raster, creación de máscaras y conversión de coordenadas mundo↔píxel.
  - `pathfinder.py`: Motor de coste (`MCP_Geometric`), reconstrucción de traceback con Numba, y worker de Fase 1 (`compute_and_save_cost_surface`).
  - `postprocessing.py`: Extrae todas las rutas desde un origen dado, filtrando pares ya calculados (`dest_id > origin_id`), y worker de Fase 2 (`reconstruct_routes_from_disk`).
  - `utils.py`: Construcción del registro de ruta (`build_route_record`) y guardado legacy a shapefile.
  - `__init__.py`: Inicialización del paquete.

### Archivos principales

- **`LCP_MCP_Geometric.ipynb`**: Notebook principal de ejecución paso a paso, visualización y exportación de resultados (9 celdas).
- **`LCP_MCP_Geometric-retake.ipynb`**: Notebook alternativo para retomar resultados ya calculados y ejecutar solo celdas 8 y 9 (post-procesamiento).
- **`generar_heatmap.py`**: Script CLI standalone para generar heatmap KDE desde cualquier GPKG de puntos densificados. Acepta `--res`, `--bw`, `--kernel`.
- **`_diag_global.py`**: Script de diagnóstico para verificar que joblib loky funcione en Windows.
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

La carpeta de sesión tiene la siguiente estructura:

```
output/session_<timestamp>_Refactored/
├── _raster_shared.dat          # memmap temporal del raster; se elimina al finalizar
├── phase1_tracebacks/          # tracebacks .npz por origen (se elimina si KEEP_PHASE1_FILES=False)
│   ├── traceback_<id>.npz
│   └── ...
├── red_completa_unificada.gpkg # rutas calculadas, layer='rutas_unificadas'
├── puntos_rutas_25m.gpkg       # puntos densificados sobre rutas (Celda 8)
├── heatmap_30m_10km_quartic.tif # mapa de calor GeoTIFF (Celda 9)
└── heatmap_30m_10km_quartic.png # mapa de calor PNG (Celda 9)
```

## Método de trabajo

El flujo implementado en `LCP_MCP_Geometric.ipynb` sigue estos pasos:

### Celda 1 — Importaciones
Carga de librerías: `numpy`, `rasterio`, `fiona`, `geopandas`, `scikit-image`, `numba`, `tqdm`, y los módulos del paquete `lcp`.

### Celda 2 — Definición de funciones
Define `_calculate_n_jobs_for_phase` (auto-detección de workers por fase según RAM disponible) y `warm_up_numba` (primer llamado a la función Numba JIT para compilarla antes del bucle principal).

### Celda 3 — Configuración central
Parámetros editables por el usuario:

| Parámetro              | Descripción                                                                  |
|------------------------|------------------------------------------------------------------------------|
| `COST_RASTER_PATH`     | Ruta al ráster de coste/fricción (`.tif`)                                   |
| `ALL_POINTS_SHAPEFILE` | Shapefile o GeoPackage con todos los puntos de la red                        |
| `MASK_SHAPEFILE_PATH`  | Polígono de restricción espacial (`None` para desactivar)                   |
| `ID_FIELD_NAME`        | Nombre del campo ID en la capa de puntos                                     |
| `OUTPUT_DIR`           | Carpeta de salida (generada automáticamente con marca de tiempo)             |
| `VERBOSE`              | `True` imprime detalles por ruta; `False` suprime el ruido                  |
| `KEEP_PHASE1_FILES`    | `False` elimina `phase1_tracebacks/` al finalizar; `True` los conserva      |

### Celda 4 — Verificación de coherencia de datos
Valida que todos los puntos de entrada se encuentren dentro del área de búsqueda definida por la máscara. Si algún punto queda fuera, la ejecución se detiene con un mensaje de error.

### Celda 5 — Visualización diagnóstica
Genera un mapa del ráster de coste con los puntos superpuestos para verificar visualmente la configuración antes de lanzar el análisis.

### Celda 6 — Ejecución principal (dos fases paralelas)
1. Crea la carpeta de sesión y escribe el memmap compartido del raster.
2. Compila la función Numba JIT con `warm_up_numba()`.
3. **Fase 1**: ejecuta en paralelo `compute_and_save_cost_surface` por cada origen. Cada worker construye `MCP_Geometric`, extrae costes escalares por destino y guarda el traceback comprimido en `phase1_tracebacks/traceback_<id>.npz`.
4. **Fase 2**: ejecuta en paralelo `reconstruct_routes_from_disk` por cada origen. Cada worker carga el traceback desde disco y reconstruye todos los caminos con Numba.
5. Consolida todos los registros en un `GeoDataFrame` y escribe `red_completa_unificada.gpkg`. Elimina archivos intermedios si `KEEP_PHASE1_FILES = False`.

### Celda 7 — Visualización y exportación final
Carga el GeoPackage desde memoria (o disco si es necesario) y genera un mapa final de la red de rutas calculadas, con escala y norte. Incluye armonización automática de CRS para evitar desplazamientos entre el ráster y las capas vectoriales.

### Celda 8 — Puntos densificados sobre rutas
Genera puntos cada `SPACING_M` metros (por defecto 25 m) sobre todas las rutas calculadas. Utiliza vectorización con Shapely 2.0 para procesar ~81M puntos en segundos. Guarda `puntos_rutas_25m.gpkg` con columna `route_idx` para trazabilidad.

### Celda 9 — Mapa de calor (Kernel Density Estimation)
Implementa KDE estilo QGIS con kernel Quartic sobre los puntos densificados. Parámetros configurables: `HEATMAP_RES_M` (resolución, 30 m), `HEATMAP_BANDWIDTH_M` (radio, 10 km), `HEATMAP_KERNEL` (quartic/gaussian/triangular/uniform). Usa numpy binning vectorizado + FFT (`scipy.fft`) para procesar 81M puntos en ~20–30 s. Salidas: GeoTIFF con metadatos + PNG inline.

**Bugs corregidos en heatmap:**
1. **Kernel con pixel_size real** (commit `dbd1595`): El kernel se construye con `pixel_size` medio en vez de `HEATMAP_RES_M`. Usar `HEATMAP_RES_M` para el kernel pero `pixel_size` para el binning genera un desplazamiento sistemático que crece de oeste a este.
2. **FFT crop offset** (commit `be6161f`): La convolución circular via FFT desplaza el resultado. El crop correcto es `heatmap[2*radius_px:2*radius_px+height, 2*radius_px:2*radius_px+width]`. El offset incorrecto hacia que el pico KDE apareciera en `(2R-1, 2R-1)` en vez de `(R-1, R-1)`.

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
- `scipy`
- `shapely`
- `tqdm`
- `joblib`
- `datashader` (opcional, para heatmaps alternativos)

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

4. Ejecuta las celdas en orden (1 → 9). Las celdas 8 y 9 son post-procesamiento opcional.

5. Los resultados se guardan en `output/session_<timestamp>/`:
   - `red_completa_unificada.gpkg`: rutas calculadas
   - `puntos_rutas_25m.gpkg`: puntos densificados (Celda 8)
   - `heatmap_30m_10km_quartic.tif`: mapa de calor GeoTIFF (Celda 9)
   - `heatmap_30m_10km_quartic.png`: mapa de calor PNG (Celda 9)

**Para retomar resultados ya calculados**: usa `LCP_MCP_Geometric-retake.ipynb` y ejecuta directamente las celdas 8 y 9 (ya apuntan al directorio de sesión existente).

## Notas técnicas

- El número de rutas calculadas para N puntos es N×(N-1)/2 gracias a la deduplicación. Para N=169 puntos: 14 196 rutas.
- `MCP_Geometric` emplea un modelo de coste simétrico: el coste de A→B es idéntico al de B→A, por lo que almacenar ambas direcciones sería redundante.
- La función `reconstruct_path_from_traceback` se compila con `@njit` de Numba la primera vez que se ejecuta. `warm_up_numba()` fuerza esta compilación antes del bucle principal para que no penalice el primer origen real.
- El buffer del path en Numba está acotado a `traceback.shape[0] + traceback.shape[1]` (filas + columnas del raster). Usar `traceback.size` como tope generaría arrays de ~1.2 GB por llamada, agotando la RAM de los workers silenciosamente.
- En Fase 1 solo se guarda el traceback comprimido (int8, ~30–80 MB/origen). El array de costes completo (float64, ~1.27 GB/origen) no se persiste; solo se extrae el escalar de coste por destino.
- La variable `save_path_to_shapefile` en `utils.py` se mantiene por compatibilidad con código anterior; el flujo principal utiliza `build_route_record`.
- El ráster de coste debe estar en un CRS proyectado (p. ej. UTM) para que `length_m` sea significativo.
- **Celda 8 (optimización)**: Usa `shapely.length` + `np.repeat` + `shapely.line_interpolate_point` vectorizado. Speedup: 20–50× sobre bucle Python (~3 min → ~5–10 s).
- **Celda 9 (optimización)**: Reemplaza `rasterio.features.rasterize` por `shapely.get_coordinates` + `np.add.at` binning vectorizado, y `np.fft` por `scipy.fft` (MKL backend). Speedup: 50–100× sobre rasterize (~20 min → ~10–20 s).
- **generar_heatmap.py**: Script CLI standalone. Uso: `uv run python generar_heatmap.py <path.gpkg> [--res 30] [--bw 10000] [--kernel quartic]`. El GeoTIFF se guarda en el mismo directorio que el GPKG de entrada.
- **CRS en visualización**: Celda 6 captura `raster_crs` y lo usa para el GeoPackage. Celda 7 armoniza automáticamente CRS de rutas y puntos al CRS del raster, y usa `ax.imshow` con `plotting_extent` para evitar desplazamientos del basemap.
- **_diag_global.py**: Script de diagnóstico para verificar que joblib + loky funcione correctamente en Windows.
- **Datashader**: Instalado como alternativa profesional para heatmaps masivos. No se usa en el flujo principal pero está disponible.

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
