# AGENTS.md — LCP_MCP_Geometric

**Workspace:** `G:\LCP_MCP_Geometric` (único directorio válido)

Instrucciones para agentes de IA que operen en este repositorio.
Este archivo tiene precedencia sobre el comportamiento por defecto del agente.

---

## 1. Entorno y gestor de paquetes

El proyecto usa **uv** como gestor de entorno y dependencias (Python ≥ 3.12).

```sh
# Instalar dependencias y crear .venv
uv sync

# Activar el entorno (Windows)
.venv\Scripts\activate

# Activar el entorno (macOS / Linux)
source .venv/bin/activate
```

### 1.1 Windows — codificación de caracteres

En Windows, la consola usa `cp1252` que NO soporta caracteres Unicode como `→`, `Δ`, `√`, etc.
**No uses caracteres Unicode en prints ni f-strings.** Prefiere ASCII puro:

```python
# MAL: print(f"Δx={dx:.1f}")  # UnicodeEncodeError en Windows
# BIEN: print(f"dx={dx:.1f}")
```

Nunca edites `uv.lock` a mano. Añade dependencias con `uv add <paquete>` y confirma
que `pyproject.toml` y `uv.lock` quedan actualizados.

---

## 2. Linting y formato

El proyecto usa **ruff** (v0.12.5). No hay un `[tool.ruff]` explícito en
`pyproject.toml`, por lo que rigen los defaults de ruff.

```sh
# Verificar estilo (sin modificar archivos)
uv run ruff check lcp/

# Corregir automáticamente lo que ruff pueda
uv run ruff check --fix lcp/

# Aplicar formato
uv run ruff format lcp/

# Verificar un único archivo
uv run ruff check lcp/pathfinder.py
uv run ruff format --check lcp/pathfinder.py
```

Antes de proponer un cambio, ejecuta `ruff check` y `ruff format --check` sobre
los archivos modificados. No introduzcas errores de lint nuevos.

> **Nota**: Si `uv run ruff check` falla con `program not found`, usa
> `uv tool run ruff check ...` o instala ruff globalmente:
> `uv tool install ruff`

---

## 3. Tests

**No existe una suite de tests formal** en este repositorio. No hay directorio
`tests/` ni archivos `test_*.py`.

### Scripts de diagnóstico

```sh
# Verifica que el patrón global + loky funcione en Windows
uv run python _diag_global.py
```

### Ejecutar el notebook de forma no interactiva

```sh
# Ejecuta LCP_MCP_Geometric.ipynb con nbconvert (timeout 30 min por celda)
uv run python run_notebook.py
```

### Si añades tests (recomendado)

- Crea el directorio `tests/` en la raíz del proyecto.
- Nombra los archivos `test_<modulo>.py` (p. ej. `test_pathfinder.py`).
- Usa **pytest**: `uv run pytest tests/`
- Para un único test: `uv run pytest tests/test_pathfinder.py::test_mi_funcion -v`
- Las funciones Numba (`@njit`) deben testearse con arrays `numpy` reales;
  el primer llamado incurre en compilación JIT (~2–5 s): es normal.

---

## 4. Ejecución interactiva

```sh
# Abrir JupyterLab con el entorno del proyecto
uv run jupyter lab
```

El notebook principal es `LCP_MCP_Geometric.ipynb`. Ejecuta las celdas en orden
(1 → 9). Los parámetros de usuario están en **Celda 3** (pipeline principal) y en
**Celdas 8 y 9** (post-procesamiento: puntos densificados y heatmap).

Para retomar resultados ya calculados sin reejecutar el pipeline completo, usa
`LCP_MCP_Geometric-retake.ipynb` y corre directamente las celdas 8 y 9 (ya
apuntan al directorio de sesión existente).

---

## 5. Arquitectura del módulo `lcp/`

| Archivo | Responsabilidad |
|---|---|
| `data_loader.py` | Carga raster (`rasterio`), puntos (`fiona`) y máscara vectorial (`shapely`). |
| `processing.py` | Conversión mundo↔píxel, creación de máscaras raster desde vectores, corredor de búsqueda. |
| `pathfinder.py` | Motor MCP (`MCP_Geometric`), reconstrucción de traceback con Numba `@njit`, y `compute_and_save_cost_surface()` (worker Fase 1: calcula superficie de coste y guarda traceback comprimido en `.npz`). |
| `postprocessing.py` | Extrae todas las rutas desde un origen dado (`extract_routes`); aplica deduplicación `dest_id > origin_id`; y `reconstruct_routes_from_disk()` (worker Fase 2: carga traceback desde `.npz` y reconstruye rutas). |
| `utils.py` | Construye el registro de ruta (`build_route_record`) y guardado legacy a shapefile. |
| `__init__.py` | Vacío; el paquete se importa con `import lcp.<modulo>`. |

Los módulos del paquete se importan siempre con la ruta completa:

```python
import lcp.pathfinder as pf
import lcp.utils as utils
import lcp.processing as proc
```

---

## 6. Estilo de código

### 6.1 Imports

Orden obligatorio (una línea en blanco entre grupos):

```python
# 1. Stdlib
import os
import numpy as np

# 2. Terceros
import rasterio
import fiona
from shapely.geometry import shape

# 3. Módulos internos lcp
import lcp.pathfinder as pf
import lcp.processing as proc
```

- Preferir `import modulo as alias` sobre `from modulo import *`.
- No usar imports relativos (`.`); siempre `lcp.<modulo>`.

### 6.2 Nombres

| Elemento | Convención | Ejemplo |
|---|---|---|
| Funciones y variables | `snake_case` | `cost_surface_calculator`, `origin_id` |
| Constantes de configuración | `UPPER_CASE` | `COST_RASTER_PATH`, `N_JOBS` |
| Clases (si se añaden) | `PascalCase` | `RouteRecord` |
| Archivos de módulo | `snake_case` | `data_loader.py` |

### 6.3 Type annotations

El código existente **no usa type annotations**. No añadas anotaciones de tipo
a funciones ya existentes salvo que se solicite explícitamente; mantén coherencia
con el estilo del archivo que estés modificando.

### 6.4 Docstrings

- Docstring de una sola línea para funciones simples.
- Párrafo libre para funciones complejas; sin secciones `Args:` / `Returns:` formales.
- Idioma: docstrings cortos en inglés, comentarios inline en español (refleja el
  estado actual de la base de código).

```python
def world_to_pixel(transform, x, y):
    """
    Converts world coordinates to pixel (row, column) using the raster transform.
    """
    ...
```

### 6.5 Manejo de errores

- Usa excepciones estándar de Python: `FileNotFoundError`, `ValueError`, `KeyError`.
- Los mensajes de error van en español y son descriptivos:

```python
raise FileNotFoundError(f"El archivo raster no se encontró en la ruta: {path}")
raise ValueError(f"Discrepancia de CRS. Raster: {raster_src.crs}, Vector: {vf.crs}")
```

- Para advertencias no fatales usa `print(f"  ADVERTENCIA: ...")` (el proyecto
  no usa el módulo `logging`).
- Devuelve `None` como centinela explícito cuando una operación opcional no produce
  resultado (p. ej. `build_route_record` devuelve `None` para rutas inválidas).

### 6.6 Logging y verbosidad

No se usa el módulo `logging`. El patrón establecido es:

```python
# Mensajes informativos de progreso
print(f"Cargando puntos desde: {os.path.basename(path)}")

# Mensajes de detalle controlados por un flag verbose
if verbose:
    print(f"  [postp] Reconstruyendo rutas desde origen {origin_id}")
```

Las funciones públicas que pueden ser ruidosas aceptan un parámetro `verbose=False`.

### 6.7 Numba (`@njit`)

- Las funciones decoradas con `@njit` solo pueden contener operaciones sobre
  arrays NumPy y tipos primitivos. No pases objetos Python (listas, dicts, etc.).
- Declara arrays con `dtype` explícito: `np.zeros((n, 2), dtype=np.int32)`.
- La primera llamada compila la función (JIT warm-up). Usa `warm_up_numba()` antes
  del bucle principal para evitar que el primer origen real sea penalizado.
- **El buffer del path debe estar acotado a `traceback.shape[0] + traceback.shape[1]`
  (filas + columnas del raster), no a `traceback.size`.** Usar `traceback.size`
  como tope genera un array de ~1.2 GB por llamada en rasters grandes (p. ej.
  12 913 × 12 238), lo que agota la RAM silenciosamente en workers joblib y hace
  que devuelvan listas vacías.

---

## 7. Datos de entrada — restricciones

- El raster de coste debe estar en un **CRS proyectado** (p. ej. UTM) para que
  `length_m` sea significativo.
- El raster se debe pasar como `float64` a `MCP_Geometric`. El cast se hace una
  sola vez fuera del bucle:
  ```python
  cost_array = src.read(1).astype(np.float64)
  ```
- Las celdas fuera de la máscara vectorial se asignan a `np.inf` antes de
  construir el objeto MCP.
- El CRS del vector de puntos y el CRS del raster deben coincidir; la herramienta
  no reprojecta automáticamente.

---

## 8. Salida

- Formato de salida principal: **GeoPackage** (`*.gpkg`).
- Archivo de salida: `output/session_<timestamp>/red_completa_unificada.gpkg`.
- No escribas shapefiles en código nuevo; usa `build_route_record` + `GeoDataFrame`.
  `save_path_to_shapefile` en `utils.py` se mantiene solo por compatibilidad.

### Estructura de la carpeta de sesión

```
output/session_<timestamp>_Refactored/
├── _raster_shared.dat          # memmap temporal del raster (float64); se elimina al finalizar
├── phase1_tracebacks/          # tracebacks comprimidos por origen (un .npz por origen)
│   ├── traceback_<id>.npz
│   └── ...
└── red_completa_unificada.gpkg # GeoPackage de salida; layer='rutas_unificadas'
```

`phase1_tracebacks/` se elimina automáticamente al finalizar si `KEEP_PHASE1_FILES = False`
(parámetro en Celda 3 del notebook).

---

## 9. Arquitectura de dos fases (Celda 6)

El análisis principal se divide en dos fases paralelas independientes.

### Fase 1 — Calcular y guardar superficies de coste

Worker: `lcp.pathfinder.compute_and_save_cost_surface(args)`

- Un worker por origen.
- Construye `MCP_Geometric` sobre el raster enmascarado.
- Ejecuta `find_costs` desde el origen.
- **Guarda solo el traceback** (int8, ~158 MB sin comprimir → ~30–80 MB comprimido)
  en `phase1_tracebacks/traceback_<origin_id>.npz`.
- **No guarda el array de costes completo** (float64, ~1.27 GB/origen). En su lugar,
  extrae el escalar `costs[row, col]` por cada destino y lo retorna como dict ligero.
- Estimación de RAM por worker: `raster_bytes × 2.125`
  (raster float64 + costs float64 + traceback int8).

### Fase 2 — Reconstruir rutas desde disco

Worker: `lcp.postprocessing.reconstruct_routes_from_disk(args)`

- Un worker por origen.
- Carga el traceback desde `phase1_tracebacks/traceback_<origin_id>.npz`.
- Reconstruye todos los paths de ese origen usando Numba `@njit`.
- Aplica deduplicación `dest_id > origin_id`.
- Retorna lista de `RouteRecord`.
- Estimación de RAM por worker: ~80 MB (solo traceback descomprimido).
  Permite hasta `n_cpu × 2` workers.

### Auto-detección de workers

`_calculate_n_jobs_for_phase(raster_shape, raster_dtype, phase)` (definida en Celda 2)
calcula cuántos workers caben en la RAM disponible para cada fase, respetando un margen
de seguridad del 20%.

---

## 10. Celdas 8 y 9 — Post-procesamiento

### Celda 8: Puntos densificados
- Parámetro configurable: `SPACING_M` (metros entre puntos, por defecto 25.0).
- Implementación vectorizada con Shapely 2.0 (`shapely.length`, `np.repeat`, `shapely.line_interpolate_point`).
- Salida: `puntos_rutas_<SPACING_M>m.gpkg` con columna `route_idx`.
- No usar bucles Python `for geom in geometries: geom.interpolate(d)`; eso es código legacy.

### Celda 9: Mapa de calor (KDE estilo QGIS)
- Parámetros configurables: `HEATMAP_RES_M`, `HEATMAP_BANDWIDTH_M`, `HEATMAP_KERNEL`.
- Kernel por defecto: Quartic (QGIS default). Opciones: quartic, gaussian, triangular, uniform.
- Implementación: `shapely.get_coordinates` + `np.add.at` binning vectorizado en vez de `rasterio.features.rasterize`.
- FFT via `scipy.fft` (MKL backend) en vez de `np.fft`.
- **Georreferenciación**: usar `rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)`.
  El pixel real del raster es `pixel_size_x = (maxx-minx)/width` y `pixel_size_y = (maxy-miny)/height`,
  que NO es exactamente `HEATMAP_RES_M`. El binning numpy DEBE usar este mismo pixel_size, no
  HEATMAP_RES_M, para que el grid de datos y el grid del raster coincidan exactamente.
  Si se usa `HEATMAP_RES_M` distinto al pixel_size real, se genera un desplazamiento sistemático
  que crece de oeste a este (y de norte a sur). **Bug corregido en commit `dbd1595`.**
- **FFT crop offset**: La convolución circular via FFT desplaza el resultado en `radius_px`
  posiciones (`circular[n] = full_linear[(n-R) mod P]`). Para extraer la porción "same" se debe
  usar `heatmap[2*radius_px:2*radius_px+height, 2*radius_px:2*radius_px+width]`, no
  `heatmap[radius_px:...]`. El offset incorrecto causaba que el pico del KDE apareciera en
  `(2*R-1, 2*R-1)` en vez de `(R-1, R-1)`. **Bug corregido en commit `be6161f`.**
- Salidas: GeoTIFF (`heatmap_<res>m_<bw>km_<kernel>.tif`) + PNG inline.
- La convolución FFT sobre histograma rasterizado es matemáticamente idéntica al KDE punto-a-punto.
- Script standalone: `generar_heatmap.py` — acepta argumentos CLI (`--res`, `--bw`, `--kernel`)
  para procesar cualquier GPKG de puntos sin editar código.

---

## 11. Lo que NO debes hacer

- No ejecutes `jupyter nbconvert --execute` directamente sobre el notebook si
  `run_notebook.py` ya lo hace; usa ese script.
- No subas archivos de la carpeta `output/` ni archivos `*.dat` (están en `.gitignore`).
- No modifiques `uv.lock` a mano.
- No añadas `from lcp import *` ni imports con comodín.
- No uses rutas absolutas hardcodeadas fuera del notebook; usa `os.path.join` o
  `pathlib.Path`.
- No guardes el array de costes completo (float64, ~1.27 GB/origen) en Fase 1;
  solo el traceback comprimido (`.npz`) más un dict ligero con el escalar de coste
  por destino.
- No acotas el buffer del path en Numba con `traceback.size`; usa siempre
  `traceback.shape[0] + traceback.shape[1]` como tope máximo de pasos.
