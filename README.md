# LCP_MCP_Geometric

## Descripción

LCP_MCP_Geometric es una herramienta de análisis espacial para el cálculo de rutas de menor coste entre múltiples puntos sobre superficies raster. Utiliza el algoritmo `MCP_Geometric` de scikit-image para propagar una superficie de coste acumulado desde cada origen y, a continuación, reconstruye los caminos óptimos mediante traceback con Numba JIT. Ha sido desarrollado para aplicaciones arqueológicas, con el objetivo de modelar movimientos en el territorio a partir de un ráster de fricción o coste.

La herramienta calcula el conjunto completo de rutas entre todos los pares de puntos de una red (análisis N×N), almacenando únicamente el triángulo superior de la matriz OD para evitar duplicar rutas simétricas.

## Características principales

- Implementa `MCP_Geometric` (scikit-image) como motor de propagación de costes, con soporte de conectividad completa (8 vecinos).
- Reconstrucción del traceback acelerada con Numba JIT (`@njit`).
- Análisis N×N deduplicado: para N puntos se calculan N×(N-1)/2 rutas únicas (triángulo superior de la matriz OD), evitando recalcular pares simétricos.
- **Modo `single_origin` (1 contra todos)**: con `ANALYSIS_MODE = "single_origin"` y un `ORIGIN_ID`, solo se calculan las N−1 rutas desde ese origen hacia el resto (estilo SSSP / `source=` de NetworkX), sin pares entre los destinos. Se elige editando la Celda 3 o desde CLI con `papermill -p` (ver sección propia).
- **Ejecución paralela en dos fases con joblib** (`loky` backend): el análisis se divide en Fase 1 (calcular y guardar superficies de coste por origen) y Fase 2 (reconstruir rutas desde disco). El número de workers se auto-detecta según la RAM disponible para cada fase. El ráster se comparte mediante un archivo `memmap` en disco.
- Permite restringir el área de búsqueda a un polígono vectorial (shapefile o GeoPackage) para excluir zonas no accesibles.
- Escribe los resultados de forma **incremental**: cada lote de la Fase 2 va directo a `red_completa_unificada.gpkg.tmp` y se renombra de forma atómica al final — nunca acumula todas las rutas en RAM ni deja un GPKG parcial — con atributos enriquecidos por ruta.
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
- **`tests/`**: Suite de pytest (`uv run pytest tests/`): unitarias de `lcp/` + roundtrip Fase 1 → Fase 2.
- **`run_notebook.py`**: Ejecuta el notebook de forma no interactiva con nbconvert (timeout 30 min/celda).
- **`LCP_MCP_Geometric.py`**: Export del notebook; se regenera con `uv run jupyter nbconvert --to script LCP_MCP_Geometric.ipynb` (no editar a mano).
- **`pyproject.toml`**: Configuración y dependencias del proyecto.
- **`uv.lock`**: Bloqueo de versiones de dependencias.
- **`requirements.txt`**: Lista de dependencias en formato pip.
- **`LICENSE`**: Licencia de uso (CC BY-NC 4.0).
- **`CITATION.cff`**: Metadatos de citación.
- **`CHANGELOG.md`**: Historial de cambios por hito.
- **`AGENTS.md`**: Instrucciones para agentes de IA que operen el repositorio.
- **`README.md`**: Este archivo.

## Formato de salida

Cada ruta calculada se almacena como un registro con los siguientes atributos:

| Campo        | Tipo    | Descripción                                              |
|--------------|---------|----------------------------------------------------------|
| `geometry`   | LineString | Geometría de la ruta en el CRS del proyecto           |
| `origin_id`  | entero  | ID del origen (en `all_pairs`, el menor del par; en `single_origin`, siempre `ORIGIN_ID`) | 
| `dest_id`    | entero  | ID del destino (en `all_pairs`, el mayor del par)        |
| `cost_total` | real    | Coste acumulado MCP en el píxel de destino               |
| `length_m`   | real    | Longitud de la ruta en metros (CRS proyectado)           |
| `n_vertices` | entero  | Número de vértices del camino reconstruido               |

Todos los registros se exportan en un único archivo **`red_completa_unificada.gpkg`** dentro de la carpeta de sesión — con la capa `rutas_unificadas` y, cuando hay destinos con coste infinito, la capa **`destinos_inalcanzables`** (`origin_id`, `dest_id`, `cost_total=inf`; se reescribe en cada corrida para no dejar filas viejas). El `.gpkg.tmp` que aparece durante la escritura incremental es transitorio: se renombra al final o se elimina si la corrida falla.

La carpeta de sesión tiene la siguiente estructura:

```
output/session_<timestamp>_Refactored/
├── _raster_shared.dat          # memmap temporal del raster; se elimina al finalizar
├── phase1_tracebacks/          # tracebacks .npz por origen (se elimina si KEEP_PHASE1_FILES=False)
│   ├── traceback_<id>.npz
│   └── ...
├── red_completa_unificada.gpkg # rutas calculadas, layer='rutas_unificadas'
├── puntos_rutas_25m.gpkg       # puntos densificados sobre rutas (Celda 8)
├── heatmap_30m_1km_quartic.tif # mapa de calor GeoTIFF (Celda 9)
└── heatmap_30m_1km_quartic.png # mapa de calor PNG (Celda 9)
```

## Método de trabajo

El flujo implementado en `LCP_MCP_Geometric.ipynb` sigue estos pasos:

### Celda 1 — Importaciones
Carga de librerías: `numpy`, `rasterio`, `fiona`, `geopandas`, `scikit-image`, `numba`, y los módulos del paquete `lcp`. (La barra de progreso con `tqdm.auto` se importa en la Celda 6.)

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
| `ANALYSIS_MODE`        | `"all_pairs"` (default, N×N) o `"single_origin"` (un origen → todos los demás) |
| `ORIGIN_ID`            | ID del punto origen; requerido y validado en modo `single_origin`            |
| `OUTPUT_DIR`           | Carpeta de salida (generada automáticamente con marca de tiempo)             |
| `VERBOSE`              | `True` imprime detalles por ruta; `False` suprime el ruido                  |
| `KEEP_PHASE1_FILES`    | `False` elimina `phase1_tracebacks/` al finalizar; `True` los conserva      |
| `RESUME`               | `True` reutiliza los `.npz` de Fase 1 si la huella (puntos+ráster+modo) coincide |

### Celda 4 — Verificación de coherencia de datos
Valida CRS idénticos (ráster/puntos/máscara), solape espacial y que **todos** los puntos caigan en píxeles válidos (fuera de máscara o `nodata` — los IDs problemáticos se listan). En modo `single_origin` comprueba además que el modo sea válido, que haya ≥ 2 puntos y que `ORIGIN_ID` exista. Cualquier problema **detiene la ejecución** con un mensaje concreto antes de calcular nada.

### Celda 5 — Visualización diagnóstica
Genera un mapa del ráster de coste con los puntos superpuestos para verificar visualmente la configuración antes de lanzar el análisis. En modo `single_origin`, resalta el origen con una estrella.

### Celda 6 — Ejecución principal (dos fases paralelas)
1. Crea la carpeta de sesión y escribe el memmap compartido del raster.
2. Compila la función Numba JIT con `warm_up_numba()`.
3. **Fase 1**: ejecuta en paralelo `compute_and_save_cost_surface` por cada origen. Cada worker construye `MCP_Geometric`, extrae costes escalares por destino y guarda el traceback en `phase1_tracebacks/traceback_<id>.npz` (comprimido si `KEEP_PHASE1_FILES=True`; **plano y rápido** si no). Con `RESUME = True`, los `.npz` con huella coincidente se reutilizan y solo se computan los pendientes.
4. **Fase 2**: ejecuta en paralelo `reconstruct_routes_from_disk` por cada origen. Cada worker carga el traceback desde disco y reconstruye todos los caminos con Numba.
5. Escribe cada lote directo (generador de joblib sin orden) a `red_completa_unificada.gpkg.tmp` y lo renombra con `os.replace` al final; imprime `MODO: ... | rutas esperadas: N` y compara reales vs esperadas. Elimina `phase1_tracebacks/` si `KEEP_PHASE1_FILES = False`.

### Celda 7 — Visualización y exportación final
Carga el GeoPackage desde memoria (o disco si es necesario) y genera un mapa final de la red de rutas calculadas, con escala y norte. Incluye armonización automática de CRS para evitar desplazamientos entre el ráster y las capas vectoriales.

### Celda 8 — Puntos densificados sobre rutas
Genera puntos cada `SPACING_M` metros (por defecto 25 m) sobre todas las rutas calculadas. Utiliza vectorización con Shapely 2.0 para procesar ~81M puntos en segundos. Guarda `puntos_rutas_25m.gpkg` con columna `route_idx` para trazabilidad.

### Celda 9 — Mapa de calor (Kernel Density Estimation)
Implementa KDE estilo QGIS con kernel Quartic sobre los puntos densificados. Parámetros configurables: `HEATMAP_RES_M` (resolución, 30 m), `HEATMAP_BANDWIDTH_M` (radio, 1 km), `HEATMAP_KERNEL` (quartic/gaussian/triangular/uniform). Usa numpy binning vectorizado + FFT (`scipy.fft`) para procesar 81M puntos en ~20–30 s — la lógica vive en `_compute_heatmap` (con tests de regresión: pico, georreferenciación y masa del kernel). Salidas: GeoTIFF con metadatos + PNG inline.

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
- **Herramientas dev** (en `pyproject.toml`): `pytest` (tests), `ruff` (lint/format), `papermill` (ejecución parametrizada sin abrir Jupyter), `nbstripout` (limpieza de outputs al hacer `git add`).

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
   - `ANALYSIS_MODE` + `ORIGIN_ID`: modo de análisis (ver sección «Modo single_origin»).

4. Ejecuta las celdas de código en orden (1 → 9); cada una está precedida de su celda markdown explicativa en español. Las celdas 8 y 9 son post-procesamiento opcional.

5. Los resultados se guardan en `output/session_<timestamp>/`:
   - `red_completa_unificada.gpkg`: rutas calculadas
   - `puntos_rutas_25m.gpkg`: puntos densificados (Celda 8)
   - `heatmap_30m_1km_quartic.tif`: mapa de calor GeoTIFF (Celda 9)
   - `heatmap_30m_1km_quartic.png`: mapa de calor PNG (Celda 9)

**Para retomar resultados ya calculados**: corre solo las celdas 8 y 9 — autodetectan la última sesión de `output/` (buscan `red_completa_unificada.gpkg` y `puntos_rutas_<espaciado>m.gpkg`) cuando `OUTPUT_DIR` no apunta a una sesión válida.

## Modos de análisis: all_pairs vs single_origin

La **Celda 3** (precedida de una celda markdown que explica ambos modos **dentro del propio notebook**) permite elegir entre dos modos:

| | `all_pairs` (default) | `single_origin` (estrella / SSSP) |
|---|---|---|
| Qué calcula | Todas las rutas entre todos los pares | Solo las rutas desde `ORIGIN_ID` hacia todos los demás; ninguna entre los otros puntos |
| Rutas con N=14 | N·(N−1)/2 = **91** | N−1 = **13** |
| MCPs en Fase 1 | N−1 = 13 (uno por origen) | **1** (solo el origen) |
| Dedup | filtro `did > origin_id` (triángulo superior OD) | `did != ORIGIN_ID` (sin triángulo) |
| Cuándo usarlo | Red completa: matrices OD, corredores, densificación (Celda 8), heatmaps (Celda 9) | Un sitio concreto: accesibilidad de un punto aislado, coste de acceso a un rasgo puntual, comparar un origen frente al resto |
| Mapas (C5/C7) | todos los puntos en rojo | origen resaltado con estrella lime |

**Concepto**: equivale a *single-source shortest paths* (SSSP) frente a *all-pairs* (APSP) — el mismo flujo con un parámetro que acota el alcance, al estilo de `source=` en NetworkX. Como `MCP_Geometric` es simétrico, la ruta O→P calculada es idéntica a P→O: en modo estrella no se pierde información para ese origen.

**Selección** — editando la Celda 3 (etiquetada `parameters`) o sin abrir el notebook:

```sh
# modo estrella
uv run papermill LCP_MCP_Geometric.ipynb out.ipynb -p ANALYSIS_MODE single_origin -p ORIGIN_ID 7
# modo completo (equivalente al default)
uv run papermill LCP_MCP_Geometric.ipynb out.ipynb -p ANALYSIS_MODE all_pairs
```

**Validación (Celda 4)**: modo desconocido, `single_origin` sin `ORIGIN_ID` o ID inexistente detienen la ejecución listando los valores válidos / IDs disponibles.

**Salida**: el mismo `red_completa_unificada.gpkg` con los mismos campos (`origin_id`, `dest_id`, `cost_total`, `length_m`, `n_vertices`) — con N−1 filas en modo estrella. La Celda 6 imprime `MODO: ... | rutas esperadas: N` al arrancar y compara reales vs esperadas al final (si difieren: destinos con coste infinito, informados con sus IDs).

**Rendimiento**: el modo estrella ejecuta 1 MCP en vez de N−1 (con los datos de ejemplo, Fase 1 de ~1,9 s a ~0,2 s y un solo worker).

**Casos límite**: `N = 1` en modo estrella genera un task degenerado (fuera del dominio previsto); los destinos inalcanzables no bloquean pero se informan con sus IDs.

## Notas técnicas

- El número de rutas calculadas para N puntos es N×(N-1)/2 gracias a la deduplicación. Para N=169 puntos: 14 196 rutas.
- `MCP_Geometric` emplea un modelo de coste simétrico: el coste de A→B es idéntico al de B→A, por lo que almacenar ambas direcciones sería redundante.
- La función `reconstruct_path_from_traceback` se compila con `@njit` de Numba la primera vez que se ejecuta. `warm_up_numba()` fuerza esta compilación antes del bucle principal para que no penalice el primer origen real.
- El buffer del path en Numba está acotado a `traceback.shape[0] + traceback.shape[1]` (filas + columnas del raster). Usar `traceback.size` como tope generaría arrays de ~1.2 GB por llamada, agotando la RAM de los workers silenciosamente.
- En Fase 1 solo se guarda el traceback (int16, 2 B/celda; comprimido solo si `KEEP_PHASE1_FILES=True` — si no, plano y rápido). El array de costes completo (float64, ~1.27 GB/origen) no se persiste; solo se extrae el escalar de coste por destino.
- La variable `save_path_to_shapefile` en `utils.py` se mantiene por compatibilidad con código anterior; el flujo principal utiliza `build_route_record`.
- El ráster de coste debe estar en un CRS proyectado (p. ej. UTM) para que `length_m` sea significativo.
- **Celda 8 (optimización)**: Usa `shapely.length` + `np.repeat` + `shapely.line_interpolate_point` vectorizado. Speedup: 20–50× sobre bucle Python (~3 min → ~5–10 s).
- **Celda 9 (optimización)**: Reemplaza `rasterio.features.rasterize` por `shapely.get_coordinates` + `np.add.at` binning vectorizado, y `np.fft` por `scipy.fft` (MKL backend). Speedup: 50–100× sobre rasterize (~20 min → ~10–20 s).
- **Tests**: `uv run pytest tests/` — suite en `tests/` (unitarias de `lcp/` + roundtrip Fase 1 → Fase 2).
- **Outputs del notebook**: no se commitean — `nbstripout` (dependencia dev) está instalado como filtro **local** de git (`nbstripout --install`), así un `git add` limpia los outputs automáticamente. La evidencia de cada corrida queda en los `_exec_*.ipynb` (ignorados) o en la salida de `run_notebook.py`.
- **Robustez**: si un worker de la Fase 1 falla, la ejecución se detiene antes de escribir el GPKG listando los IDs fallidos (nunca una red parcial en silencio); los destinos inalcanzables (coste `inf`) se informan. `load_points_as_dict` lanza `ValueError` con los IDs duplicados.
- **CRS en visualización**: Celda 6 captura `raster_crs` y lo usa para el GeoPackage. Celda 7 armoniza automáticamente CRS de rutas y puntos al CRS del raster, y usa `ax.imshow` con `plotting_extent` para evitar desplazamientos del basemap.
- **Datashader**: Instalado como alternativa profesional para heatmaps masivos. No se usa en el flujo principal pero está disponible.
- **Escalabilidad (N grande)**: la Fase 2 y el densificado de la Celda 8 se escriben de forma **incremental** — la memoria solo sostiene un lote/origen o un chunk, nunca todos los registros (antes: todos los `LineString` en RAM + concatenación global de ~1,6 GB a 81M puntos). Ambas fases muestran progreso con `tqdm`. El orden de filas de los GPKG es el orden de finalización; `route_idx` da la trazabilidad por ruta. *Límite conocido*: la Celda 7 (mapa final) materializa todas las rutas para graficar — con N muy grande usa el PNG ya guardado o evita esa celda.
- **Rendimiento (E/S)**: con `KEEP_PHASE1_FILES=False` (lo normal) los `.npz` de Fase 1 se guardan **planos** (~29 s de zlib por origen ahorrados en rasters grandes); el worker usa la **vista del memmap** sin copiar el ráster (+~20–25% de workers por la menor RAM); los puntos se escriben con **pyogrio** (2,9× vs el bucle de fiona). La evaluación completa con benchmarks está en `odd/tasks/perf-fast-io.md`.

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
