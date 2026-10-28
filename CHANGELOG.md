# Changelog

Cambios destacados, agrupados por hito. El detalle vive en los commits
(Conventional) y en los documentos `odd/tasks/` (specs S#, log L#).

## Sin versión — publicado en `origin/LCP_MCP_Geometric` (rama principal)

### notebook-hardening (robustez del pipeline)

- **Fase 1 fail-hard**: si un worker falla, la ejecución lista los IDs de orígenes
  afectados y se detiene **antes** de escribir el GPKG (nunca una red parcial en
  silencio); los destinos con coste infinito se informan con sus IDs.
- **Celda 9**: path dinámico `puntos_rutas_<SPACING_M>m.gpkg` (fin del hardcode
  `25m`), `globals()` en vez de `dir()`, guards para rutas degeneradas y CRS `None`.
- **Celda 3**: `ID_FIELD_NAME` real de los datos de ejemplo (`id`).
- **`load_points_as_dict`**: `ValueError` con los IDs duplicados (antes se
  sobrescribían en silencio).
- Primeras suites pytest (RED→GREEN), `[tool.ruff]` fijado en `pyproject.toml`,
  README/AGENTS sincronizados, artefactos (`__pycache__`, checkpoints, logs) fuera
  del índice git, export `.py` regenerado sin drift.

### star-origin-mode (modo 1 contra todos)

- **`ANALYSIS_MODE`**: `"all_pairs"` (N×(N−1)/2, default intacto) o
  `"single_origin"` con `ORIGIN_ID` → N−1 rutas desde un origen, sin pares entre
  los destinos (SSSP vs APSP, al estilo `source=` de NetworkX).
- Validación temprana (modo desconocido, ID faltante/inexistente) con listado de
  IDs disponibles; echo `MODO: ... | rutas esperadas: N` y comparación
  reales vs esperadas al final; resaltado del origen en los mapas (Celdas 5 y 7).
- **papermill** como dependencia dev: la Celda 3 está etiquetada `parameters`, y
  el modo se elige sin abrir Jupyter (`papermill ... -p ANALYSIS_MODE single_origin
  -p ORIGIN_ID 7`).
- Celda markdown con la explicación completa de ambos modos; sección dedicada en
  README y AGENTS.

### quality-corrections (backlog de calidad)

- **Tests**: loaders y máscara (`load_raster`, `load_mask_geometry`,
  `create_mask_from_vector` con CRS mismatch y reparación de geometrías),
  estimación de workers, **regresión KDE/FFT** (pico en la celda del punto,
  `pixel_size` desde bounds, masa = 1/ps² — cubre los bugs históricos `dbd1595` y
  `be6161f`) y un **harness de mini-notebook** que ejecuta celdas con fixtures
  negativas (CRS desalineado, píxel nodata).
- **Robustez**: `N ≥ 2` obligatorio en modo estrella; capa GPKG aditiva
  `destinos_inalcanzables` (destinos con coste infinito visibles en la salida);
  **reanudación de Fase 1** con huella (`RESUME`, `phase1_meta.json`, escalares
  persistidos en los `.npz`) — probada con doble corrida.
- **Notebook auto-documentado**: celda markdown en español antes de cada una de
  las 9 celdas de código.
- **Higiene**: `ruff` 0.17.2 como dependencia dev (sin fallback), docstring real
  de `run_notebook.py`, `description` real en `pyproject.toml`, **nbstripout**
  (los outputs dejan de commitearse; filtro local, sin CI),
  `joblib` anclado a **git main `1.7.dev0`** (loky 3.7.0) que **resolvió el
  KeyError** del resource_tracker — 0 apariciones vs 5 por corrida antes
  (commit `878c9e05` fijado en uv.lock; volver a PyPI con `uv add joblib`
  cuando salga 1.7.0 estable), este CHANGELOG y `CITATION.cff` actualizados.
- **Integración y publicación**: tres merges `--no-ff` en la rama principal
  (`notebook-hardening` → `star-origin-mode` → `quality-corrections`) y push a
  `origin/LCP_MCP_Geometric` (`b65cb74..10689df`); las ramas `feature/*` también
  quedan subidas al remoto.
- **Verificación independiente**: REJECT del verificador por la capa
  `destinos_inalcanzables` no idempotente (`mode="a"` duplicaba filas al
  re-ejecutar) → reparado con `mode="w"` y **recheck VERIFY**; después, el
  advisory de filas stale (re-corrida con 0 pares) se cerró escribiendo la capa
  siempre (vacía cuando no hay pares). Tests endurecidos a 3 corridas con
  `listlayers` + esquema.
- **Housekeeping**: `nul` eliminado, stashes antiguos revisados y borrados.

### scale-large-n (escala para N muy grandes)

- **Fase 2 incremental**: `return_as="generator_unordered"` — cada lote de
  origen se escribe a `red_completa_unificada.gpkg.tmp` y se renombra con
  `os.replace` al final: la Celda 6 **nunca acumula las rutas en RAM** y el
  GPKG final nunca queda parcial (el `finally` limpia el temporal si falla).
- **Celda 8 incremental**: los chunks se escriben directos al GPKG de puntos
  (sin la concatenación global `xs/ys/idx` — ~1,6 GB a 81M puntos) con aviso
  honesto si `escritos != esperados`.
- **Progreso**: `tqdm.auto` sobre resultados en Fase 1 y Fase 2 (joblib
  `verbose=0`; `disable=None` mantiene limpias las salidas headless).
- **Paridad verificada** contra el baseline pre-cambio: rutas idénticas
  (coste, longitud, vértices — orden-agnóstico), 82 288 puntos con
  distribución idéntica (`sorted(bincount)` sha256 igual), campos iguales,
  0 errores; 74 tests + ruff verdes.
- *Conocido → **resuelto***: el `TqdmWarning: IProgress not found` se corrigió añadiendo **`ipywidgets` como dependencia del proyecto** (8.1.9 + jupyterlab-widgets 3.0.17): verificado con probe y corrida completa (0 warnings, **2 salidas widget** — las barras de las fases renderizan como widgets reales). *Conocido*: la Celda 7 materializa todas las rutas para graficar (límite
  documentado en README para N muy grandes).

### perf-fast-io (rendimiento de E/S)

- **Compresión híbrida de los npz de Fase 1**: con `KEEP_PHASE1_FILES=False`
  (lo normal) se guardan **planos** con `np.savez` — se ahorran ~29 s de zlib
  por origen en raster real (bench: int16 316 MB → 29,0 s comprimido vs 0,3 s
  plano) y el directorio se borra al final igual; con `True` se comprimen
  para que los archivos persistentes de la reanudación sean chicos.
  `np.load` lee ambos (Fase 2 sobre npz plano testeada).
- **Vista del memmap sin copia**: el worker entrega la vista directa a
  `MCP_Geometric` (verificado que no muta la entrada) — ahorra 0,82 s y
  1,27 GB de RAM por worker y origen; el estimador de workers baja de ×5.5 a
  ×4.5 (~+20–25% de paralelismo en máquinas limitadas por RAM).
- **Escritura de la Celda 8 con pyogrio**: 0,71 s vs 2,04 s por 100k puntos
  (**2,9×**) — `gdf.to_file` por chunk en vez del bucle de records de fiona.
- Paridad verificada tras los3 cambios: 91 rutas, 82 288 puntos,
  `sorted(bincount)` idéntico, 0 errores. Evaluación completa con benchmarks:
  `odd/tasks/perf-fast-io.md`. Sin cambios en FFT (Celda 9), Fase 2/densify
  ni overhead de imports (decisiones explícitas).

### multipoint-support

- **Capas MultiPoint en los puntos**: `Point` y MultiPoint de 1 parte
  conservan el id; un MultiPoint de **k ≥ 2 coordenadas se explota en k
  nodos** con ids derivados `id*1000+j` (j = 1..k, orden del archivo, solo
  x/y) con una única línea de resumen. Colisiones de ids derivados y
  geometrías no puntuales → `ValueError` (los ids siguen siendo enteros
  únicos: lo exigen npz/GPKG y `did > origin_id`).
- Las Celdas 5 y 7 grafican desde el diccionario ya cargado — fin del
  `AttributeError: 'MultiPoint' object has no attribute 'x'`.
- Verificación: RED 8 → GREEN 9 (tests multipoint), harness de celdas 1–4 con
  fixture MultiPoint (`Se cargaron 5 puntos`), suite **87 passed**, corrida de
  muestra 91/91 con 0 errores y los 3 PNG (C5/C7 desde el dict).
