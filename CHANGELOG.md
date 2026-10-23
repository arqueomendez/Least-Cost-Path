# Changelog

Cambios destacados, agrupados por hito. El detalle vive en los commits
(Conventional) y en los documentos `odd/tasks/` (specs S#, log L#).

## No publicado — ramas locales (`feature/*` sobre `LCP_MCP_Geometric`)

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
  (los outputs dejan de commitearse; filtro local, sin CI), `joblib` 1.6.0
  (loky 3.6.0; el `KeyError` cosmético del resource_tracker **persiste** — fix
  upstream en joblib PR#1829 / loky ≥3.7.0 pendiente de release), este CHANGELOG
  y `CITATION.cff` actualizado.
