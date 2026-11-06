# LCP_MCP_Geometric

## Descripción

LCP_MCP_Geometric es una herramienta de análisis espacial para el cálculo de rutas de menor coste entre múltiples puntos sobre superficies raster. Utiliza el algoritmo `MCP_Geometric` de scikit-image para propagar una superficie de coste acumulado desde cada origen y, a continuación, reconstruye los caminos óptimos mediante traceback con Numba JIT. Ha sido desarrollado para aplicaciones arqueológicas, con el objetivo de modelar movimientos en el territorio a partir de un ráster de fricción o coste.

La herramienta calcula el conjunto completo de rutas entre todos los pares de puntos de una red (análisis N×N), almacenando únicamente el triángulo superior de la matriz OD para evitar duplicar rutas simétricas.

## Características principales

- Implementa `MCP_Geometric` (scikit-image) como motor de propagación de costes, con soporte de conectividad completa (8 vecinos).
- Reconstrucción del traceback acelerada con Numba JIT (`@njit`).
- Análisis N×N deduplicado: para N puntos se calculan N×(N-1)/2 rutas únicas (triángulo superior de la matriz OD), evitando recalcular pares simétricos.
- **Modo `single_origin` (1 contra todos)**: con `ANALYSIS_MODE = "single_origin"` y un `ORIGIN_ID`, solo se calculan las N−1 rutas desde ese origen hacia el resto (estilo SSSP / `source=` de NetworkX), sin pares entre los destinos. Se elige editando la Celda 3 o desde CLI con `papermill -p` (ver sección propia).
- **Modo `cross_layers` (una capa de orígenes contra una capa de destinos)**: con `ANALYSIS_MODE = "cross_layers"` se calculan las rutas de **cada** punto de `ORIGIN_POINTS_PATH` hacia **todos** los de `DEST_POINTS_PATH` (M×N), con **una corrida MCP por raíz** y salidas **no unificadas**: un GPKG, un mapa, un densificado y un KDE por raíz, más un manifiesto CSV. Admite invertir la dirección cuando las fuentes son menos que los sitios (ver «Modo cross_layers»).
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
  - `postprocessing.py`: Extrae todas las rutas desde un origen dado, filtrando pares ya calculados (`dest_id > origin_id`), y worker de Fase 2 (`reconstruct_routes_from_disk`; también la variante invertida del modo de dos capas, `reconstruct_inverted_routes_from_disk`).
  - `planner.py`: Planificador puro (sin E/S) de pares por raíz MCP para los tres modos (`build_pair_plan`) y detección de pares que caen en el mismo píxel (`identical_pixel_pairs`).
  - `outputs.py`: Nombres canónicos de los artefactos por raíz, manifiesto CSV y escritura atómica por archivo.
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

En `ANALYSIS_MODE = "cross_layers"` **no hay archivo unificado**: cada raíz MCP escribe su propio GeoPackage con las columnas `origin_id`, `dest_id`, `origin_layer`, `dest_layer`, `cost_total`, `length_m`, `n_vertices` y `geometry` (ver «Modo cross_layers»). Los modos `all_pairs` y `single_origin` siguen escribiendo `red_completa_unificada.gpkg` sin cambios.

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
| `ANALYSIS_MODE`        | `"all_pairs"` (default, N×N), `"single_origin"` (un origen → todos los demás) o `"cross_layers"` (capa de orígenes → capa de destinos, M×N) |
| `ORIGIN_ID`            | ID del punto origen; requerido y validado en modo `single_origin`            |
| `ORIGIN_POINTS_PATH`, `DEST_POINTS_PATH`, `ORIGIN_ID_FIELD`, `DEST_ID_FIELD`, `INVERT_DIRECTION` | Solo en `cross_layers`: capa de sitios, capa de fuentes, campo id de cada capa y política de inversión de dirección (ver «Modo cross_layers») |
| `HEATMAP_ORIGIN_IDS`   | Solo en `cross_layers`: `None` = un KDE por cada raíz; lista de ids = solo esas raíces |
| `OUTPUT_DIR`           | Carpeta de salida (generada automáticamente con marca de tiempo)             |
| `VERBOSE`              | `True` imprime detalles por ruta; `False` suprime el ruido                  |
| `KEEP_PHASE1_FILES`    | `False` elimina `phase1_tracebacks/` al finalizar; `True` los conserva      |
| `RESUME`               | `True` reutiliza los `.npz` de Fase 1 si la huella (puntos+ráster+modo) coincide |
| `MAX_WORKERS_PHASE1`   | `None` = automático. La Celda 2 detecta el hardware (RAM disponible, núcleos físicos/lógicos, disco) y **sugiere** los workers con el modelo de memoria medido; un entero actúa como tope y, si excede lo que permite la RAM, se avisa y se recorta |
| `EDGE_BUFFER_PX`       | `0` = no excluye nada. `N` excluye las `N` filas/columnas exteriores del ráster: corta el corredor perimetral cuando el modelo de coste es poco fiable en los límites (la máscara del polígono es siempre **estricta**, no configurable) |

### Celda 4 — Verificación de coherencia de datos
Valida CRS idénticos (ráster/puntos/máscara), solape espacial y que **todos** los puntos caigan en píxeles válidos (fuera de máscara o `nodata` — los IDs problemáticos se listan). En modo `single_origin` comprueba además que el modo sea válido, que haya ≥ 2 puntos y que `ORIGIN_ID` exista. Cualquier problema **detiene la ejecución** con un mensaje concreto antes de calcular nada. En `cross_layers` valida además la política `INVERT_DIRECTION`, el CRS y los píxeles válidos de **ambas** capas, y aborta si un sitio y una fuente caen en el **mismo píxel** (lista los pares `(sitio, fuente)`: una ruta de longitud cero no se puede representar).

### Celda 5 — Visualización diagnóstica
Genera un mapa del ráster de coste con los puntos superpuestos para verificar visualmente la configuración antes de lanzar el análisis. En modo `single_origin`, resalta el origen con una estrella.

### Celda 6 — Ejecución principal (dos fases paralelas)
1. Crea la carpeta de sesión y escribe el memmap compartido del raster.
2. Compila la función Numba JIT con `warm_up_numba()`.
3. **Fase 1**: ejecuta en paralelo `compute_and_save_cost_surface` por cada origen. Cada worker construye `MCP_Geometric`, extrae costes escalares por destino y guarda el traceback en `phase1_tracebacks/traceback_<id>.npz` (comprimido si `KEEP_PHASE1_FILES=True`; **plano y rápido** si no). Con `RESUME = True`, los `.npz` con huella coincidente se reutilizan y solo se computan los pendientes.
4. **Fase 2**: ejecuta en paralelo `reconstruct_routes_from_disk` por cada origen. Cada worker carga el traceback desde disco y reconstruye todos los caminos con Numba.
5. Escribe cada lote directo (generador de joblib sin orden) a `red_completa_unificada.gpkg.tmp` y lo renombra con `os.replace` al final; imprime `MODO: ... | rutas esperadas: N` y compara reales vs esperadas. Elimina `phase1_tracebacks/` si `KEEP_PHASE1_FILES = False`.

**En `cross_layers`** la Fase 2 cambia de forma: el plan de `lcp.planner.build_pair_plan` fija **una raíz MCP por sitio** (o por fuente si la dirección se invierte) y cada raíz escribe su propio `rutas_por_origen/rutas_origen_<raiz>.gpkg` — con `.tmp` + `os.replace` **por archivo** — más una fila en `manifest_rutas_por_origen.csv`, que se reescribe tras cada raíz para que un corte deje un estado parcial legible. El GPKG unificado no se crea en este modo; al final imprime un reporte por raíz con `esperados`, `inalcanzables`, `escritos` y `estado`.

### Celda 7 — Visualización y exportación final
Carga el GeoPackage desde memoria (o disco si es necesario) y genera un mapa final de la red de rutas calculadas, con escala y norte. Incluye armonización automática de CRS para evitar desplazamientos entre el ráster y las capas vectoriales. En `cross_layers` recorre el manifiesto y genera **un PNG por raíz** (`mapas_por_origen/mapa_origen_<raiz>.png`) y no deja figuras abiertas.

### Celda 8 — Puntos densificados sobre rutas
Genera puntos cada `SPACING_M` metros (por defecto 25 m) sobre todas las rutas calculadas. Utiliza vectorización con Shapely 2.0 para procesar ~81M puntos en segundos. Guarda `puntos_rutas_25m.gpkg` con columna `route_idx` para trazabilidad. En `cross_layers` guarda **un GPKG por raíz** (`puntos_por_origen/puntos_origen_<raiz>.gpkg`, capa `puntos_origen_<raiz>`, misma columna `route_idx`).

### Celda 9 — Mapa de calor (Kernel Density Estimation)
Implementa KDE estilo QGIS con kernel Quartic sobre los puntos densificados. Parámetros configurables: `HEATMAP_RES_M` (resolución, 30 m), `HEATMAP_BANDWIDTH_M` (radio, 1 km), `HEATMAP_KERNEL` (quartic/gaussian/triangular/uniform). Usa numpy binning vectorizado + FFT (`scipy.fft`) para procesar 81M puntos en ~20–30 s — la lógica vive en `_compute_heatmap` (con tests de regresión: pico, georreferenciación y masa del kernel). Salidas: GeoTIFF con metadatos + PNG inline. En `cross_layers` genera **un GeoTIFF por raíz** (`heatmaps_por_origen/heatmap_origen_<raiz>_<res>m_<bw>km_<kernel>.tif`) sobre **una sola grilla por sesión** (unión de extensiones + margen de banda), para que la FFT del kernel se calcule una vez y se reutilice; el KDE recorre las raíces en secuencia (1 worker) y `HEATMAP_ORIGIN_IDS` permite limitarlo a un subconjunto de raíces.

**Bugs corregidos en heatmap:**
1. **Kernel con pixel_size real** (commit `dbd1595`): El kernel se construye con `pixel_size` medio en vez de `HEATMAP_RES_M`. Usar `HEATMAP_RES_M` para el kernel pero `pixel_size` para el binning genera un desplazamiento sistemático que crece de oeste a este.
2. **FFT crop offset** (commit `be6161f`): La convolución circular via FFT desplaza el resultado. El crop correcto es `heatmap[2*radius_px:2*radius_px+height, 2*radius_px:2*radius_px+width]`. El offset incorrecto hacia que el pico KDE apareciera en `(2R-1, 2R-1)` en vez de `(R-1, R-1)`.

## Dependencias

El proyecto utiliza las siguientes dependencias principales (ver `pyproject.toml`):

- `fiona`
- `geopandas`
- `jupyterlab`
- `ipywidgets`
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

## Instalación desde cero

Estos pasos llevan de un equipo limpio a una ejecución funcional del notebook en
Windows, macOS y Linux. Los ejemplos suponen que el repositorio queda en una
carpeta llamada `Least-Cost-Path`, creada por `git clone`.

### Requisitos previos

| Componente | Mínimo | Notas |
|---|---|---|
| Sistema operativo | Windows 10/11, macOS 12+, Linux x86_64 | Las tres plataformas están cubiertas con wheels binarios en x86_64 |
| Git | 2.30 | Necesario para clonar el repositorio y para que `uv sync` obtenga `joblib` desde git |
| `uv` | Última versión | Vía recomendada: gestiona el entorno e instala Python 3.12 si falta (`requires-python = ">=3.12"`; `.python-version` fija `3.12`) |
| Python | 3.12 | Solo si se usa la ruta alternativa con `pip` (Paso 7) |
| RAM | 8 GB (recomendado 16 GB) | Los rásteres grandes pueden necesitar bajar `MAX_WORKERS_PHASE1` en la Celda 3 (ver «Problemas frecuentes») |
| Espacio en disco | ~2 GB libres | Entorno virtual y dependencias; los datos de entrada y `output/` van aparte |
| GDAL / PROJ | Ninguno | `rasterio`, `fiona`, `pyogrio`, `geopandas` y `shapely` se instalan como wheels binarios: no hay que compilar GDAL ni usar conda. Una plataforma sin wheels (p. ej. Linux arm64) sí requiere un entorno conda con GDAL del sistema |

### Paso 1 — Instalar Git

Windows (PowerShell):

```sh
winget install --id Git.Git -e
```

Instalador alternativo para Windows: <https://git-scm.com/download/win>

macOS:

```sh
xcode-select --install
# o, con Homebrew
brew install git
```

Linux:

```sh
sudo apt install git      # Debian / Ubuntu
sudo dnf install git      # Fedora / RHEL
sudo pacman -S git        # Arch
```

### Paso 2 — Instalar uv

Windows (PowerShell):

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# alternativa con winget
winget install --id=astral-sh.uv -e
```

macOS y Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternativas si ya hay un Python disponible:

```sh
pipx install uv
# o
pip install uv
```

Comprobar la instalación:

```sh
uv --version
```

En Windows hay que **cerrar y reabrir la terminal** después de instalar para que
el `PATH` se actualice; hasta entonces `uv` responde `uv: command not found`.

### Paso 3 — Clonar el repositorio

```sh
git clone -b LCP_MCP_Geometric https://github.com/arqueomendez/Least-Cost-Path.git
cd Least-Cost-Path
```

El flag `-b LCP_MCP_Geometric` no es decorativo: la rama por defecto del
repositorio remoto (`main`) es un puntero antiguo y heredado, y todo el trabajo
actual vive en `LCP_MCP_Geometric`. Un `git clone` sin `-b` descarga código
obsoleto.

Verificar que se ha clonado la rama correcta:

```sh
git log -1 --oneline
# esperado: 92b598f ... (o un commit posterior de la misma rama)
```

Actualizar el repositorio más adelante:

```sh
git pull --ff-only
```

En Windows, clonar en una ruta local corta y sin sincronización (por ejemplo
`C:\repos\Least-Cost-Path`), **nunca** dentro de OneDrive ni de una carpeta de
red, y activar las rutas largas antes de clonar:

```sh
git config --global core.longpaths true
```

### Paso 4 — Crear el entorno e instalar dependencias

Desde la raíz del repositorio:

```sh
uv sync
```

`uv sync` crea `.venv` con Python 3.12, resuelve las versiones exactas de
`uv.lock` e instala todos los grupos de dependencias, incluidas las herramientas
de desarrollo (`nbstripout`, `papermill`, `pytest`, `ruff`, definidas en
`[dependency-groups]`). La primera vez también descarga `joblib` desde su
repositorio git, porque `pyproject.toml` lo fija en `[tool.uv.sources]` mientras
no haya versión estable publicada.

**Activar el entorno es opcional**: `uv run` antepone el `.venv` del proyecto a
cualquier comando, así que los pasos siguientes funcionan sin activar nada. Si
prefieres activarlo de forma explícita:

Windows (PowerShell o CMD):

```sh
.venv\Scripts\activate
```

Si PowerShell bloquea el script por política de ejecución:

```sh
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

macOS y Linux:

```sh
source .venv/bin/activate
```

### Paso 5 — Verificar la instalación

```sh
uv run pytest tests/ -q
```

La salida esperada es `96 passed`.

```sh
uv run ruff check .
```

Ambos comandos deben pasar antes de usar datos reales: si la suite está en verde
y `ruff` no reporta errores, el entorno está sano.

### Paso 6 — Abrir JupyterLab

```sh
uv run jupyter lab
```

Abrir `LCP_MCP_Geometric.ipynb` y ejecutar las celdas de código en orden (1 → 9).
Las celdas 8 (puntos densificados) y 9 (mapa de calor) son post-procesamiento
opcional y pueden omitirse sin afectar al resultado principal de la Celda 6. Las
barras de progreso (`tqdm`) y los widgets se renderizan dentro de JupyterLab; en
ejecuciones sin interfaz (`run_notebook.py`, `papermill`) caen a texto.

### Paso 7 — Alternativa sin uv (pip + venv)

Si no puedes o no quieres instalar `uv`, `requirements.txt` contiene el mismo
conjunto de dependencias bloqueadas:

Windows (Python launcher):

```sh
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

macOS y Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` es un volcado automático de `uv.lock` (un test de la suite
comprueba que sigue siendo fiel) y no se edita a mano. Para regenerarlo, con `uv`
instalado:

```sh
uv export --format requirements-txt --output-file requirements.txt
```

En esta ruta no hay `uv run`, así que los comandos se lanzan con el entorno
activado (`pytest tests/ -q`, `ruff check .`, `jupyter lab`) y siempre desde la
raíz del repositorio.

### Problemas frecuentes de instalación

| Síntoma | Causa y solución |
|---|---|
| `uv: command not found` | El `PATH` todavía no incluye el instalador. Cerrar y reabrir la terminal; si persiste, comprobar que `%USERPROFILE%\.local\bin` (Windows) o `~/.local/bin` (macOS/Linux) está en el `PATH` |
| Al activar el venv: no se puede cargar el archivo porque la ejecución de scripts está deshabilitada | Política de PowerShell. Ejecutar `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y reintentar |
| `UnicodeEncodeError: 'charmap' codec can't encode character ...` al imprimir acentos o símbolos desde Python | La consola de Windows usa cp1252. Definir `$env:PYTHONIOENCODING = "utf-8"` en PowerShell (`set PYTHONIOENCODING=utf-8` en CMD) antes de ejecutar |
| `ModuleNotFoundError: No module named 'lcp'` | El comando no se lanzó desde la raíz del repositorio. El paquete se importa como `lcp.<modulo>` y no está instalado en el entorno: hay que hacer `cd` a la carpeta clonada |
| `uv sync` lento o con errores de TLS detrás de un proxy | Reapuntar la caché con `$env:UV_CACHE_DIR` (`export UV_CACHE_DIR=...` en macOS/Linux) o forzar el almacén de certificados del sistema con `uv sync --native-tls` |
| `ArrayMemoryError: Unable to allocate ... _offset_edge_map` durante una corrida | No es un fallo de instalación: bajar `MAX_WORKERS_PHASE1` en la Celda 3 (por ejemplo a `4`) y reejecutar con `RESUME = True` |
| El primer `uv sync` parece colgado sin mensajes | Está resolviendo y construyendo `joblib` desde git. Dejar que termine; las sincronizaciones siguientes usan la caché |

### Siguiente paso

Con el entorno verificado, continúa en «Uso básico» para preparar los datos de
entrada. Los parámetros de ejecución se ajustan en la Celda 3 (ver «Celda 3 —
Configuración central»).

## Uso básico

1. Prepara los archivos de entrada:
   - Un ráster de coste/fricción (`*.tif`).
   - Un shapefile o GeoPackage con todos los puntos de la red (`*.shp` / `*.gpkg`). Acepta `Point` y `MultiPoint`: un MultiPoint de k coordenadas se **explota en k nodos** con ids derivados `id*1000+j` (la carga imprime un resumen de la explosión).
   - **Opcional**: Un shapefile con el polígono de restricción espacial (`*.shp`).

2. Abre `LCP_MCP_Geometric.ipynb` en JupyterLab.

3. Ajusta los parámetros en **Celda 3**:
   - `COST_RASTER_PATH`: ruta a tu ráster de coste.
   - `ALL_POINTS_SHAPEFILE`: ruta a tus puntos.
   - `MASK_SHAPEFILE_PATH`: ruta al polígono de máscara, o `None`.
   - `ID_FIELD_NAME`: nombre del campo ID en la capa de puntos.
   - `ANALYSIS_MODE`: modo de análisis — `all_pairs`, `single_origin` (+ `ORIGIN_ID`) o `cross_layers` (+ `ORIGIN_POINTS_PATH`, `DEST_POINTS_PATH`, `INVERT_DIRECTION`); ver «Modos de análisis» y «Modo cross_layers».

4. Ejecuta las celdas de código en orden (1 → 9); cada una está precedida de su celda markdown explicativa en español. Las celdas 8 y 9 son post-procesamiento opcional.

5. Los resultados se guardan en `output/session_<timestamp>/`:
   - `red_completa_unificada.gpkg`: rutas calculadas (`all_pairs` / `single_origin`)
   - `puntos_rutas_25m.gpkg`: puntos densificados (Celda 8)
   - `heatmap_30m_1km_quartic.tif`: mapa de calor GeoTIFF (Celda 9)
   - `heatmap_30m_1km_quartic.png`: mapa de calor PNG (Celda 9)
   - en `cross_layers`: `rutas_por_origen/`, `mapas_por_origen/`, `puntos_por_origen/`, `heatmaps_por_origen/` y `manifest_rutas_por_origen.csv` (un juego de artefactos por raíz MCP)

**Para retomar resultados ya calculados**: corre solo las celdas 8 y 9 — autodetectan la última sesión de `output/` (buscan `red_completa_unificada.gpkg` y `puntos_rutas_<espaciado>m.gpkg`) cuando `OUTPUT_DIR` no apunta a una sesión válida.

## Modos de análisis: all_pairs, single_origin y cross_layers

La **Celda 3** (precedida de celdas markdown que explican los tres modos **dentro del propio notebook**) permite elegir entre tres modos de análisis:

| | `all_pairs` (default) | `single_origin` (estrella / SSSP) | `cross_layers` (dos capas) |
|---|---|---|---|
| Qué calcula | Todas las rutas entre todos los pares | Solo las rutas desde `ORIGIN_ID` hacia todos los demás; ninguna entre los otros puntos | **Cada** punto de la capa de orígenes hacia **todos** los de la capa de destinos; ningún par dentro de la misma capa |
| Alcance | 1 capa, todos los pares | 1 capa, 1 origen | 2 capas, M×N |
| Rutas (ejemplo) | N=14: N·(N−1)/2 = **91** | N=14: N−1 = **13** | 3 sitios × 2 fuentes = **6** |
| MCPs en Fase 1 | N−1 = 13 (uno por origen) | **1** (solo el origen) | **M** (uno por sitio) o **N** (uno por fuente si se invierte) |
| Dedup | filtro `did > origin_id` (triángulo superior OD) | `did != ORIGIN_ID` (sin triángulo) | **ninguna**: los ids se repiten entre capas y es legítimo |
| Salidas | `red_completa_unificada.gpkg` (+ capa `destinos_inalcanzables`) | el mismo GPKG unificado | un GPKG + un PNG + un densificado + un GeoTIFF **por raíz** y `manifest_rutas_por_origen.csv` |
| Cuándo usarlo | Red completa: matrices OD, corredores, densificación (Celda 8), heatmaps (Celda 9) | Un sitio concreto: accesibilidad de un punto aislado, coste de acceso a un rasgo puntual, comparar un origen frente al resto | Origen y destino son **fenómenos distintos** en capas distintas (sitios → fuentes, casos → hospitales) y cada corrida se revisa o se comparte por separado |
| Mapas (C5/C7) | todos los puntos en rojo | origen resaltado con estrella lime | un mapa por raíz (orígenes en cian, destinos en amarillo) |

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

### Modo cross_layers

Modo de **dos capas**: las rutas van de **cada** punto de una capa de orígenes (los sitios) hacia **todos** los puntos de una capa de destinos (las fuentes). A diferencia de los modos de una sola capa, las dos capas se cargan en **diccionarios separados y nunca se fusionan**: los ids se repiten entre capas y eso es legítimo (el sitio `1` y la fuente `1` son puntos distintos, con coordenadas distintas).

**Concepto**: **M×N pares** — todos los sitios contra todas las fuentes, sin deduplicación por id y sin pares dentro de una misma capa. Con 3 sitios y 2 fuentes se calculan 6 rutas, y cada corrida MCP (una por **raíz**) produce su propio juego de artefactos.

**Parámetros nuevos (Celda 3)**:

| Parámetro | Default | Descripción |
|---|---|---|
| `ANALYSIS_MODE` | `"all_pairs"` | Poner `"cross_layers"` para activar el modo. En este modo `ALL_POINTS_SHAPEFILE`, `ID_FIELD_NAME` y `ORIGIN_ID` no se usan |
| `ORIGIN_POINTS_PATH` | `None` | Capa de orígenes / sitios (obligatoria en este modo) |
| `DEST_POINTS_PATH` | `None` | Capa de destinos / fuentes (obligatoria en este modo) |
| `ORIGIN_ID_FIELD` | `"id"` | Campo identificador de la capa de orígenes |
| `DEST_ID_FIELD` | `"id"` | Campo identificador de la capa de destinos |
| `INVERT_DIRECTION` | `"auto"` | Política de inversión: `"auto"`, `"never"` o `"always"` |
| `HEATMAP_ORIGIN_IDS` | `None` | `None` = KDE de todas las raíces; lista de ids = solo esas raíces |

**Ejemplo de configuración con las dos capas** (Celda 3):

```python
ANALYSIS_MODE = "cross_layers"

ORIGIN_POINTS_PATH = r"D:\Sitios y fuentes\Sitios.gpkg"  # capa de sitios
DEST_POINTS_PATH = r"D:\Sitios y fuentes\Fuentes.gpkg"  # capa de fuentes
ORIGIN_ID_FIELD = "id"
DEST_ID_FIELD = "id"
INVERT_DIRECTION = "auto"

# El resto de la configuracion (raster de coste, mascara, EDGE_BUFFER_PX,
# KEEP_PHASE1_FILES, RESUME) se mantiene igual que en los otros modos.
COST_RASTER_PATH = os.path.join(DATA_DIR, "cost.tif")
MASK_SHAPEFILE_PATH = os.path.join(DATA_DIR, "area-mask.shp")
```

**Ejecución sin abrir el notebook**:

```sh
uv run papermill LCP_MCP_Geometric.ipynb out.ipynb -p ANALYSIS_MODE cross_layers -p ORIGIN_POINTS_PATH "D:\Sitios y fuentes\Sitios.gpkg" -p DEST_POINTS_PATH "D:\Sitios y fuentes\Fuentes.gpkg"
```

**Inversión de dirección** (`INVERT_DIRECTION`, solo en este modo):

| Valor | Comportamiento |
|---|---|
| `"auto"` | Invierte solo si hay **menos destinos que orígenes** (N < M) |
| `"never"` | Nunca invierte: M corridas MCP, una por sitio |
| `"always"` | Invierte siempre: N corridas MCP, una por fuente |

- **Cuándo conviene**: `MCP_Geometric` explora el área necesaria para alcanzar sus destinos, así que el coste lo manda el área barrida. Si las fuentes son pocas y los sitios muchos, invertir cambia M corridas por N.
- **Las rutas se registran igual**: `origin_id` = sitio y `dest_id` = fuente en ambas direcciones (la polilínea se invierte si la corrida partió de la fuente).
- **El coste acumulado es idéntico**: el MCP es simétrico, así que `cost_total` del par sitio-fuente no cambia al invertir (medido: 34.1421 en ambas direcciones).
- **La geometría puede diferir levemente**: con empates de coste el traceback elige una de las rutas óptimas, de modo que `n_vertices` y `length_m` pueden cambiar (medido: mismo número de vértices, trazado no bit-idéntico). La diferencia es de trazado, no de optimalidad.

**Layout de salidas** — una raíz MCP = un juego de artefactos (nunca unificados):

```
output/session_<timestamp>_Refactored/
├── _raster_shared.dat                      # memmap temporal; se elimina al finalizar
├── phase1_tracebacks/                      # un .npz por raíz MCP
│   └── traceback_<raiz>.npz
├── rutas_por_origen/
│   └── rutas_origen_<raiz>.gpkg            # layer 'rutas_origen_<raiz>'
├── mapas_por_origen/
│   └── mapa_origen_<raiz>.png              # Celda 7
├── puntos_por_origen/
│   └── puntos_origen_<raiz>.gpkg           # Celda 8, layer 'puntos_origen_<raiz>'
├── heatmaps_por_origen/
│   └── heatmap_origen_<raiz>_30m_1km_quartic.tif   # Celda 9
└── manifest_rutas_por_origen.csv           # una fila por raíz
```

`<raiz>` es el id de la raíz MCP de esa corrida: el **sitio** si la dirección es directa, la **fuente** si se invirtió. No se escribe `red_completa_unificada.gpkg`: en este modo no hay red unificada.

**Columnas de cada GPKG de rutas**:

| Columna | Significado |
|---|---|
| `origin_id` | Id del sitio (extremo de origen de la ruta) |
| `dest_id` | Id de la fuente (extremo de destino) |
| `origin_layer` | Nombre de la capa del origen (las capas nunca se fusionan) |
| `dest_layer` | Nombre de la capa del destino |
| `length_m` | **Distancia de la ruta** entregada, en unidades del CRS (metros si el CRS es proyectado) |
| `cost_total` | **Coste acumulado mínimo** del par según el modelo de coste: la suma de costes de píxel del camino óptimo, no una distancia |
| `n_vertices` | Número de vértices de la polilínea |

`length_m` y `cost_total` miden cosas distintas y no son intercambiables: dos rutas con la misma distancia pueden tener costes muy distintos si cruzan píxeles caros.

**El manifiesto** (`manifest_rutas_por_origen.csv`) declara el estado de cada raíz y se reescribe tras cada una para que una interrupción deje un estado parcial legible:

| Columna | Significado |
|---|---|
| `root_id` | Id de la raíz MCP |
| `capa` | Capa de la raíz: `origin` (directo) o `dest` (invertido) |
| `agrupacion` | Agrupación de los artefactos (`root`) |
| `esperados` | Pares planificados para esa raíz |
| `inalcanzables` | Pares con coste infinito |
| `escritos` | Rutas realmente escritas en el GPKG de la raíz |
| `estado` | `ok`, `incompleto`, `sin_rutas` o `error` |
| `gpkg`, `mapa`, `densificado`, `heatmap` | Rutas de los artefactos; las tres últimas se completan al correr las Celdas 7, 8 y 9 |

Una raíz en estado distinto de `ok` no bloquea a las demás: la Celda 6 termina con un reporte por raíz que lista las que no quedaron completas.

**Heatmap global a partir de los densificados por raíz** (opcional): la Celda 9 de este modo produce un KDE **por raíz**. Si quieres un único heatmap de toda la red, concatena los `puntos_origen_*.gpkg` en el archivo que espera el camino legacy (`puntos_rutas_<SPACING_M>m.gpkg`) y vuelve a ejecutar la Celda 9 con `ANALYSIS_MODE` distinto de `"cross_layers"` (por ejemplo `"all_pairs"`), que usa una sola grilla para todos los puntos:

```python
import glob
import os

import geopandas as gpd
import pandas as pd

partes = [
    gpd.read_file(p)
    for p in sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "puntos_por_origen", "puntos_origen_*.gpkg"))
    )
]
global_gdf = gpd.GeoDataFrame(pd.concat(partes, ignore_index=True), crs=partes[0].crs)
global_gdf.to_file(
    os.path.join(OUTPUT_DIR, "puntos_rutas_25m.gpkg"),
    layer="puntos_rutas_25m",
    driver="GPKG",
)
```

La columna `route_idx` se reinicia en cada raíz, así que en ese archivo concatenado identifica la ruta **dentro de su GPKG de origen**: si necesitas trazabilidad por raíz, añade una columna con el id de la raíz antes de escribir (p. ej. `global_gdf["root_id"] = ...`).

**Rendimiento**:

- **Una pasada de MCP por raíz**: M corridas en dirección directa, N si se invierte (más la Fase 2 correspondiente).
- **El coste lo manda el área explorada y la cantidad de raíces, no la cantidad de destinos**: un MCP explora hasta el destino más lejano. Medido, un solo destino cercano supuso 363 celdas y ~1 ms, mientras que 180 destinos exploraron ~360 000 celdas y tardaron ~110 ms.
- **GPKG por archivo**: cada GeoPackage arrastra ~96 KB de overhead propio (metadatos/esquema de SQLite), así que M raíces implican M archivos — frente a uno solo en los modos legacy.
- **KDE**: una sola grilla por sesión permite calcular la FFT del kernel una vez; el bucle por raíz reutiliza esa FFT y corre secuencial (1 worker), de modo el tiempo total crece con el número de raíces.

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
- **Capas MultiPoint**: se aceptan en la capa de puntos; cada feature de k > 1 coordenadas genera k nodos con ids derivados `id*1000+j` (j = 1..k, orden del archivo; colisión de ids o geometría no puntual → `ValueError`). Los ids derivados aparecen como `origin_id`/`dest_id` en las salidas.
- **CRS en visualización**: Celda 6 captura `raster_crs` y lo usa para el GeoPackage. Celda 7 armoniza automáticamente CRS de rutas y puntos al CRS del raster, y usa `ax.imshow` con `plotting_extent` para evitar desplazamientos del basemap.
- **Datashader**: Instalado como alternativa profesional para heatmaps masivos. No se usa en el flujo principal pero está disponible.
- **Escalabilidad (N grande)**: la Fase 2 y el densificado de la Celda 8 se escriben de forma **incremental** — la memoria solo sostiene un lote/origen o un chunk, nunca todos los registros (antes: todos los `LineString` en RAM + concatenación global de ~1,6 GB a 81M puntos). Ambas fases muestran progreso con `tqdm`. El orden de filas de los GPKG es el orden de finalización; `route_idx` da la trazabilidad por ruta. *Límite conocido*: la Celda 7 (mapa final) materializa todas las rutas para graficar — con N muy grande usa el PNG ya guardado o evita esa celda.
- **Memoria por worker (modelo medido)**: el pico de la Fase 1 no lo domina el
  ráster sino las estructuras de `skimage.graph.MCP_Geometric` (mapa de offsets
  `(2, H, W)` int64 + costes float64 + traceback int16). Medido con `fully_connected=True`:
  **76 B por píxel y worker** más ~30 MB fijos (12,2 GB por worker en el ráster de
  171,5 Mpx). La Fase 2 cuesta **2 B por píxel** (solo carga el traceback).
  `lcp/resources.py` implementa el modelo y la sugerencia; un factor del tipo
  «ráster × 4,5» subestima ×2 en rásteres grandes y es la causa del
  `ArrayMemoryError` en `_offset_edge_map`. Disco de Fase 1: `orígenes × H × W × 2`
  bytes (~19 GB con 60 orígenes en ese ráster).
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
