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

### corrección del sesgo de borde de la máscara

- Síntoma reportado: algunas rutas se dirigían al borde del polígono de máscara y
  lo recorrían para alcanzar otros puntos — un corredor de bajo coste que el
  polígono no cubría. Causa: `create_mask_from_vector` rasterizaba con
  `all_touched=True`, de modo que la máscara **se expandía** a todo píxel que el
  polígono apenas rozaba. Ese anillo exterior conserva el valor del modelo de
  coste; si el modelo es barato fuera del área de estudio, el anillo forma un
  camino continuo de bajo coste que rodea el polígono, y el LCP lo prefiere
  frente a cruzar el interior.
- **`MASK_ALL_TOUCHED`** (Celda 3, default `False` = máscara estricta): solo son
  navegables los píxeles cuyo **centro** cae dentro del polígono. `True`
  restaura el comportamiento anterior para quien tenga puntos legítimos justo
  sobre el límite; la Celda 4 avisa entonces del sesgo de borde.
- **`EDGE_BUFFER_PX`** (Celda 3, default `0`): excluye las N filas/columnas
  exteriores del ráster. Con la máscara estricta, el anillo lo aporta el propio
  polígono, pero si la máscara cubre todo el ráster el corredor barato está en el
  marco del modelo de coste (verificado en los datos reales: medianas de coste
  ~5 y ~12 en los márgenes izquierdo/derecho frente a ~33 en el centro).
- Diagnóstico nuevo en la Celda 4: si el polígono de máscara cubre el 100 % del
  ráster se avisa de que **no restringe** el área de búsqueda; si algún punto
  cae inválido por quedar en el anillo rozado, el error lo explica y propone las
  dos salidas (activar `MASK_ALL_TOUCHED` o corregir el polígono/punto), en vez
  de listar un ID sin contexto.
- Verificación: RED `TypeError: unexpected keyword 'all_touched'` → GREEN
  (4 pins nuevos en `test_processing.py` + 2 pins de cableado en
  `test_notebook_hardening.py`), y ejecución de las Celdas 1-4 con los datos
  reales: máscara estricta sin puntos inválidos + aviso de máscara total; con
  `EDGE_BUFFER_PX=200` se excluyen 10 360 000 píxeles del marco.

### retiro del interruptor de máscara permisiva (`MASK_ALL_TOUCHED`)

- La máscara del área de estudio es **siempre estricta**: solo son navegables los
  píxeles cuyo centro cae dentro del polígono. El parámetro `MASK_ALL_TOUCHED`
  desaparece de la Celda 3 y `create_mask_from_vector(..., all_touched=True)`
  queda solo para diagnóstico y pruebas.
- Motivo (evaluación medida, `tests/test_mask_all_touched_effect.py` y
  `scripts/mask_all_touched_eval.py`): la rasterización permisiva agrega un anillo
  exterior de 1 px con el coste del modelo. Su peso relativo cae al crecer el
  polígono (21,0 % → 1,0 % del área navegable de 20 a 400 px de lado) pero **el
  ahorro no**: la ruta sigue siendo ~20× más barata (ratio 0,096 → 0,043). El
  anillo se vuelve atractivo desde una razón de coste interior/exterior de **2,5**
  y es irrelevante con ≤ 2.
- Efectos nulos comprobados: polígono que contiene el ráster (máscaras y rutas
  idénticas), bordes alineados al píxel (máscaras iguales), y en diagonal 45° el
  anillo es de 1 celda y solo conecta en 8 (funciona porque MCP usa
  `fully_connected=True`).
- Su único uso legítimo (admitir un punto cuyo centro queda fuera del polígono) se
  resuelve ahora con un mensaje accionable en la Celda 4: mover el punto o
  corregir/erosionar el polígono, nunca abrir el anillo.
- Verificación: pin de notebook reescrito (no queda el knob y la máscara navegable
  usa el default) + 8 pruebas de caracterización del parámetro; suite **109
  passed**, `ruff` limpio.

### endurecimiento de la reconstrucción de rutas (tracebacks corruptos)

- **Problema**: `reconstruct_path_from_traceback` es `@njit` y solo comprobaba
  `code < 0`. Un traceback corrupto (todo ceros, un código de offset inventado o
  un camino que sale del ráster) hacía que el código compilado leyera fuera del
  array y el proceso muriera con **violación de acceso** (`0xC0000005`), tumbando
  el worker entero en lugar de devolver `None`. Verificado: 4 de 6 escenarios
  corruptos mataban el proceso hijo.
- **Corrección**: se validan los extremos (`start`/`end` dentro del ráster), el
  código de offset (`0 <= code < len(offsets)`) y la celda destino de cada paso,
  en las dos pasadas. Cualquier anomalía devuelve `None`.
- **Coste**: medido sobre 575 050 vértices, el resultado es idéntico y el tiempo
  no empeora (17,9 ms con controles frente a 19,6 ms sin ellos; dentro del ruido).
  Con la carga real (11,77 M vértices) la reconstrucción total toma ~0,43 s, muy
  por debajo de los 39,8 s que dura la Fase 2.
- **Pruebas**: `tests/test_pathfinder_corruption.py` (7) — los seis casos corruptos
  se ejecutan en un **subproceso**, de modo que una regresión falla por el código
  de salida del hijo en vez de matar pytest; más un control positivo con un
  traceback real de `MCP_Geometric`.

### modelo de memoria por worker medido y sugerencia de workers

- **Problema**: el estimador de workers de la Celda 2 usaba `raster_bytes × 4.5`
  para la Fase 1 y 80 MB fijos para la Fase 2. En el ráster real (14 355 × 11 945
  = 171,5 Mpx) predecía 6,2 GB por worker cuando el pico medido es **12,2 GB**, y
  con 6 workers el consumó real era 73 GB frente a 54 GB disponibles: de ahí el
  `ArrayMemoryError: Unable to allocate 2.56 GiB … _offset_edge_map`.
- **Causa**: el pico no lo domina el ráster sino las estructuras de
  `MCP_Geometric` — mapa de offsets `(2, H, W)` int64 (16 B/px, además **contiguo**),
  costes acumulados float64 (8 B/px) y traceback int16 (2 B/px).
- **Medición** (pico RSS de un proceso que solo ejecuta un MCP, `fully_connected=True`):
  83,9 B/px en 4 Mpx, 76,9 en 36 Mpx y 76,2 en 171,5 Mpx → se adopta
  **76 B/px + 30 MB** por worker; la Fase 2 mide **2 B/px** (solo carga el traceback).
- **`lcp/resources.py`**: `phase_peak_bytes`, `suggest_n_jobs`, `advice_for_phase`,
  `detect_hardware`, `disk_bytes_for_phase1` y `format_hardware`. La sugerencia es
  `min(RAM_disponible × 0,8 / pico, núcleos físicos [Fase 1] o 2 × lógicos [Fase 2],
  tareas)`, con mínimo 1.
- **Cableado**: la Celda 2 imprime hardware, pico por worker y sugerencia, y acepta
  `n_tasks`; la Celda 6 le pasa el número real de tareas y, si `MAX_WORKERS_PHASE1`
  excede el límite seguro, avisa con los GB implicados y recorta.
- **Verificación**: 17 pruebas nuevas en `tests/test_resources.py` (pines de las tres
  mediciones y de la sugerencia, con chequeo de mutación: con la constante vieja
  fallan 7), 5 pines reescritos en `tests/test_quality_backlog.py` al seam nuevo y
  el caso real pinneado (171,5 Mpx + 40 GB → 2 workers); suite **129 passed**.

### control de workers de Fase 1 (OOM verificado en dos máquinas)
- **`MAX_WORKERS_PHASE1`** (Celda 3, default `None` = automático): tope manual
  de workers de la Fase 1. Motivo: con el ráster real (14 355 × 11 945 ≈ 171 Mpx,
  float64 = 1,37 GB) cada worker necesita además un bloque **contiguo** de
  2,74 GB (`_offset_edge_map` de skimage, (2,H,W) int64) — con la RAM
  comprometida por otras aplicaciones la asignación falla con
  `ArrayMemoryError: Unable to allocate 2.56 GiB` (reproducido en esta máquina
  con 6 workers y en el equipo remoto de 128 GB). Remedio: limitar a 4–6
  workers; con `RESUME=True` la re-ejecución reutiliza los `.npz` sanos y solo
  recalcula los orígenes fallidos.
