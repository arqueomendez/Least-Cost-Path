# Feature: od-pair-mode

Objetivo: nuevo modo de analisis **`od_pair`** — una sola ruta origen->destino elegida **por id o por cualquier otro atributo**, dentro de la misma capa de puntos. Motivacion del usuario (L1): "se puede seleccionar via id u otro atributo un unico punto de inicio y un uno punto de destino. Es importante que justamente este proceso no lance fallos o tenga una mejor forma de seleccionar esto, ya que ha causado problemas previamente con id vs fid".
Branch: `LCP_MCP_Geometric` (feature corto; commits por unidad de trabajo).
Runner: `uv run pytest tests/ -q` · `uv run ruff check .` · `uv run ruff format --check .`
Regresion obligatoria: `all_pairs` 14 puntos = 91 rutas; `single_origin` = 13 rutas.

## Specs

- **S1** — Modo `od_pair` en `ANALYSIS_MODE` ("all_pairs" | "single_origin" | "cross_layers" | "od_pair"). Una sola ruta entre un punto de inicio y un punto de destino de la **misma** capa (`ALL_POINTS_SHAPEFILE`). Reutiliza el camino legacy: GPKG unificado `red_completa_unificada.gpkg` y Celdas 7-8-9 intactas (mapa, densificado y KDE sobre esa ruta unica). Sin manifiesto por raiz ni salidas no unificadas.
- **S2** — Seleccion por campo configurable y valor: `ORIGIN_SELECT_FIELD` / `ORIGIN_SELECT_VALUE` y `DEST_SELECT_FIELD` / `DEST_SELECT_VALUE` (default `"id"`). El valor puede ser numerico o texto; la comparacion es tolerante a tipo (int/float/str normalizados). **Prohibido el fallback silencioso a fid**: si el campo no existe se aborta con error que lista los campos disponibles (nunca se usa el fid sin que el usuario lo pida explicitamente eligiendo `field="fid"`).
- **S3** — Errores estrictos con opciones: campo inexistente -> `ValueError` con la lista de campos de la capa; valor no encontrado -> `ValueError` con los valores disponibles de ese campo; `ORIGIN_SELECT_VALUE == DEST_SELECT_VALUE` cuando ambos usan el mismo campo y valor -> `ValueError` (origen = destino). Mensajes en espanol, descriptivos, con sugerencia practica (ej. "corre el preview de Celda 4").
- **S4** — Preview (selector interactivo): en modo `od_pair`, si falta `ORIGIN_SELECT_VALUE` o `DEST_SELECT_VALUE`, la Celda 4 imprime la tabla de candidatos de la capa — `id`, `fid`, coordenadas y píxel — y **detiene** la ejecucion con un mensaje claro; nunca adivina ni elige automaticamente. Si los valores estan definidos y son validos, imprime el eco de la seleccion en vez de la tabla completa.
- **S5** — `lcp/planner.py`: `build_pair_plan("od_pair", origins, origin_id=<id>, dest_id=<id>)` -> `{"mode": "od_pair", "inverted": False, "roots": {origin_id: {dest_id: (x, y)}}, "n_expected": 1}`. Validaciones: `origin_id` y `dest_id` existen (error lista los ids disponibles), `origin_id != dest_id`. Los modos existentes no cambian.
- **S6** — Eco de seleccion: al resolver cada punto se imprime `id` elegido, `fid` real del feature, campo/valor usado y píxel destino del raster. Columnas del GPKG identicas a legacy (`origin_id, dest_id, cost_total, length_m, n_vertices, geometry`); sin dedup `did > origin_id` ni filtro `did != origin_id`: exactamente el par pedido (incluye el caso `origin_id > dest_id`).
- **S7** — Notebook y docs: parametros en Celda 3 (etiqueta `parameters`), preview en Celda 4, un solo task de Fase 1 (raiz = origen, target = destino) y Fase 2 reconstruyendo la ruta unica en Celda 6; celda markdown en espanol documentando el modo (tabla comparativa + CLI papermill); README y AGENTS actualizados; `LCP_MCP_Geometric.py` regenerado con nbconvert (nunca a mano).
- **S8** — Verificacion: RED->GREEN de los tests nuevos (selector y planner); suite completa verde; ruff limpio; corrida funcional `od_pair` con fixture (1 ruta en el GPKG, columnas pobladas), incluyendo el caso `origin_id > dest_id` (sin dedup) y seleccion por campo no-id; regresion legacy `all_pairs` = 91 y `single_origin` = 13.

## API congelada

```python
# lcp/data_loader.py
def load_point_candidates(path, id_field):
    """Lista de {id, fid, coords, properties} de TODOS los puntos (para preview)."""

def select_point(candidates, field, value):
    """Devuelve el dict del punto elegido; estricto: sin fallback a fid."""


# lcp/planner.py
# build_pair_plan("od_pair", origins, dests=None, origin_id=..., dest_id=...)
# nuevo kwarg dest_id; mode "od_pair" en MODES.
```

## Tasks

- T1 — S2,S3,S5 — done (commit b5e4fe7) — selector estricto (`load_point_candidates` + `select_point`) en `lcp/data_loader.py` y modo `od_pair` en `lcp/planner.py`; tests RED primero (`tests/test_od_pair.py`).
- T2 — S1,S4,S6 — done (commit 20b2adc; 16 tests: 11 pins + 5 e2e) — Celdas 3/4/6 del notebook con `od_pair` (parametros, preview, un solo task F1/F2 sin dedup) + pins estructurales en `tests/test_od_pair_notebook.py`.
- T3 — S7 — done (commit 590443a) — docs (README, AGENTS, celda markdown, CHANGELOG) y export regenerado.
- T4 — S8 — done — verificacion: `pytest tests/` 238 passed (era 207; +31), `ruff check .` limpio, `ruff format --check` limpio. Corrida funcional od_pair sobre data/ real: 1 ruta (id 7 -> id 3, cost=2786.1, length=19800.6 m, 598 vertices), eco con fid real (id 7 -> fid 6: demuestra la confusion id/fid resuelta). Regresion legacy: all_pairs = 91 OK, single_origin = 13 OK. verificacion completa (suite, ruff, corrida funcional, regresion 91/13).

## Log

- L1 (peticion original, verbatim): "Lo que necesito es incorporar una nueva funcionalidad donde se puede seleccionar via id u otro atributo un unico punto de inicio y un uno punto de destino. Es importante que justamente este proceso no lance fallos o tenga una mejor forma de seleccionar esto, ya que ha causado problemas previamente con id vs fid"
- L2 (decision del usuario): modo nuevo "od_pair" (opcion 1) + "Selector interactivo con preview" (opcion 3 de seleccion).
- L3: hallazgo de exploracion: `load_points_as_dict` hoy hace fallback silencioso a `feature.id` (fid) cuando el campo no esta en las propiedades (solo con aviso impreso); en modos legacy se conserva (regresion), pero el selector nuevo de od_pair nunca cae en ese fallback.

- L4 (verificacion T4): corrida real evidencio la confusion historica: seleccionar id 7 resolvio al feature con fid 6 (ids de atributo desfasados respecto del fid del shapefile). El selector estricto + eco de ambos elimina la ambiguedad.
