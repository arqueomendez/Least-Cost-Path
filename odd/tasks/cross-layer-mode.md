# Feature: cross-layer-mode

Objetivo: nuevo modo de analisis **capa de origenes -> capa de destinos** (M x N, "uno a todos pero cada uno"), con **una corrida por raiz MCP** y salidas NO unificadas (GPKG + mapa + densificado + KDE por raiz), **inversion de direccion** cuando las fuentes son menos que los sitios, y columnas `length_m` (distancia) y `cost_total` (coste acumulado) en cada GPKG.
Branch: `feature/cross-layer-mode` (desde `LCP_MCP_Geometric`). Commits locales por unidad de trabajo, sin push.
Runner: `uv run pytest tests/` · `uv run ruff check .` · `uv run ruff format --check .`
Regresion obligatoria: `all_pairs` 14 puntos = 91 rutas; `single_origin` = 13 rutas.

## Specs

- **S1** — Modo `cross_layers` en `ANALYSIS_MODE`. Nuevos parametros en Celda 3: `ORIGIN_POINTS_PATH`, `DEST_POINTS_PATH`, `ORIGIN_ID_FIELD = "id"`, `DEST_ID_FIELD = "id"`. Cada capa se carga en un diccionario PROPIO; **nunca se fusionan** (los ids se repiten entre capas y es legitimo). Semantica: M x N pares, todos los sitios contra todas las fuentes, sin dedup por id.
- **S2** — `lcp/planner.py` (nuevo, puro, sin E/S): `build_pair_plan(mode, origins, dests=None, origin_id=None, invert="never")` devuelve `{"mode", "inverted", "roots": {root_id: {target_id: (x, y)}}, "n_expected"}`.
  Reglas: `all_pairs` = `did > oid` (N(N-1)/2); `single_origin` = todos menos `origin_id` (N-1); `cross_layers` = M*N. Modo invalido, `origin_id` ausente o capa vacia -> `ValueError` con los modos/IDs disponibles.
- **S3** — Inversion de direccion (solo `cross_layers`): `INVERT_DIRECTION = "auto" | "never" | "always"` (default `"auto"`). `auto` invierte si `len(dests) < len(origins)`. Invertido = un MCP por **fuente** (raiz) hacia todos los sitios; las rutas se registran igual semantica sitio->fuente (`origin_id` = sitio, `dest_id` = fuente) con la polilinea invertida. `cost_total` es el minimo acumulado (identico por simetria de MCP); `length_m` es la longitud de la polilinea entregada.
- **S4** — Validacion temprana en Celda 4: `_VALID_MODES` incluye `cross_layers`; valida CRS, extension y pixeles validos de **ambas** capas; valida `INVERT_DIRECTION`; y si un sitio y una fuente caen en el **mismo pixel** -> `ValueError` listando los pares `(sitio, fuente)`. Modos legacy conservan su comportamiento exacto.
- **S5** — Salidas por corrida, **no unificadas**. Una raiz MCP = un juego de artefactos: `rutas_por_origen/rutas_origen_<root>.gpkg` (una capa `rutas_origen_<root>`), `mapas_por_origen/mapa_origen_<root>.png`, `puntos_por_origen/puntos_origen_<root>.gpkg`, `heatmaps_por_origen/heatmap_origen_<root>_<res>m_<bw>km_<kernel>.tif`. Agrupacion = **raiz MCP** (`origin` directo, `dest` invertido); el manifiesto lo declara.
- **S6** — Columnas de cada GPKG: `origin_id, dest_id, origin_layer, dest_layer, cost_total, length_m, n_vertices, geometry`. `length_m` = distancia de la ruta en unidades del CRS (metrico) y `cost_total` = coste acumulado minimo. Ambas presentes en los tres modos de escritura del modo nuevo, con o sin inversion.
- **S7** — `manifest_rutas_por_origen.csv` (una fila por raiz): `root_id, capa, agrupacion, esperados, inalcanzables, escritos, estado, gpkg, mapa, densificado, heatmap`. `estado` in {`ok`, `incompleto`, `sin_rutas`, `error`}. Se reescribe tras cada raiz (atomico) para que un corte deje estado parcial legible. Reporte final por raiz en consola.
- **S8** — Escritura atomica **por archivo**: borrar `.tmp` previo, escribir siempre con `mode="w"`, `os.replace` al final. Un fallo deja archivos completos y a lo sumo un `.tmp` huerfano (que nunca se reutiliza). Prohibido `mode="a"` sobre un `.tmp` existente (duplicaria filas).
- **S9** — `RESUME`: la huella incluye ambas capas y la identidad de la mascara (ademas de modo + direccion + raster). Fase 2 descarta registros con coste no finito (clase de ruta falsa desde traceback en cero). Los modos legacy conservan su ruta de codigo.
- **S10** — Celdas 7/8/9 por raiz: mapa, puntos densificados y KDE uno por artefacto de rutas; el kernel del KDE se calcula **una sola vez** y se reutiliza; el KDE corre secuencial (1 worker).
- **S11** — Docs: README (modo nuevo, layout de salidas, columnas, inversion), AGENTS (§4/§9 y tabla de tests), **celda markdown en espanol** dentro del notebook junto a la de los otros modos, y `LCP_MCP_Geometric.py` regenerado con nbconvert (nunca a mano).
- **S12** — Verificacion: RED->GREEN de los pins nuevos; `pytest tests/` completo; `ruff check`/`format --check` limpios; corrida de regresion `all_pairs` (91) y `single_origin` (13); corrida `cross_layers` con fixtures de dos capas (3 x 2 = 6 rutas) verificando un GPKG por raiz, columnas `length_m`/`cost_total` pobladas y ausencia de pares sitio-sitio o fuente-fuente.

## API congelada (los workers deben implementar exactamente esto)

```python
# lcp/planner.py
MODES = ("all_pairs", "single_origin", "cross_layers")
INVERT_POLICIES = ("auto", "never", "always")
def build_pair_plan(mode, origins, dests=None, origin_id=None, invert="never"): ...
def identical_pixel_pairs(origin_pixels, dest_pixels): ...  # [(oid, did)] mismo (row, col)

# lcp/outputs.py
MANIFEST_NAME = "manifest_rutas_por_origen.csv"
MANIFEST_COLUMNS = ("root_id", "capa", "agrupacion", "esperados", "inalcanzables",
                    "escritos", "estado", "gpkg", "mapa", "densificado", "heatmap")
def session_paths(output_dir): ...            # crea subdirs, devuelve rutas absolutas
def route_artifact(paths, root_id): ...
def figure_artifact(paths, root_id): ...
def dense_artifact(paths, root_id): ...
def heatmap_artifact(paths, root_id, res_m, bw_km, kernel): ...
def write_origin_routes(records, paths, root_id, crs): ...  # -> (path, n_escritas)
def write_manifest(paths, rows): ...          # atomico, reescribe completo
def read_manifest(paths): ...                 # -> list[dict]

# lcp/utils.py
def build_route_record(pixel_path, transform, origin_id, dest_id, cost_total,
                       origin_layer=None, dest_layer=None): ...
    # las dos columnas de capa se agregan SOLO si no son None (esquema legacy intacto)

# lcp/postprocessing.py
def reconstruct_routes_from_disk(args): ...
    # args[:6] como hoy; args[6] opcional (origin_layer, dest_layer)
    # descarta registros con coste no finito
def reconstruct_inverted_routes_from_disk(args): ...
    # (root_id, root_coords, target_items, cost_scalars, transform, phase1_dir[, layers])
    # MCP desde la fuente: registra origin_id=t_target, dest_id=root, geometria invertida
```

`lcp/pathfinder.py` NO cambia: `compute_and_save_cost_surface` ya es agnostico a la raiz.

## Tasks

- T1 — S2 — `lcp/planner.py` + `tests/test_planner.py` (RED->GREEN).
- T2 — S5,S7,S8 — `lcp/outputs.py` + `tests/test_outputs.py`.
- T3 — S6,S9 — `lcp/utils.py` + `lcp/postprocessing.py` + `tests/test_postprocessing_direction.py`.
- T4 — S1,S3,S4 — Celdas 3 y 4 del notebook.
- T5 — S5,S7,S8,S9 — Celda 6 del notebook (plan, escritura por raiz, manifiesto, huella).
- T6 — S10 — Celdas 7, 8 y 9 del notebook.
- T7 — S11 — Export regenerado + README + AGENTS + celda markdown.
- T8 — S12 — Verificacion completa y commits por unidad.

## Log

- **L1** (peticion original, verbatim): "Necesito poder integrar un nuevo método donde se calculen las rutas óptimas desde cada uno de los puntos de una capa vectorial a todos los puntos de otra capa, es decir, un uno a todos, pero cada uno. Por ejemplo, desde cada uno de los puntos de "D:\Sitios y fuentes\Sitios.gpkg" por separado a todos los puntos de "D:\Sitios y fuentes\Fuentes.gpkg". Analiza el caso y plantea la forma de incorporarlo."
- **L2** (verbatim): "Antes, es necesario que se generen tantos mapas resultantes como corridas desde un origen a todos los destinos. Esto no se debe unificar."
- **L3** (verbatim): "Para verificar, simula la lógica de una corrida por cada uno de los modos y detecta posibles fallos o problemas de optimización."
- **L4** (verbatim): "Aplica la implementación y, detecta que, cuando las fuentes sean menos que los sitios, se pueda invertir la dirección. También, es importante que el archivo gpkg final tenga una columna que indique la distancia de la ruta y su costo total acumulado."
- **L5** (decisiones del usuario, verbatim): "Módulo en lcp/ + modo del notebook" · "Capa nueva + columnas de capa" · "Fallar en la Celda 4" · "GPKG por origen + figura por origen" · "También por origen" (Celdas 8 y 9).
- **L6** (simulacion por modo, evidencia medida): `all_pairs` 3 pares -> 1 escrita (2 inalcanzables); `single_origin` 2 -> 1; `cross_layers` 12 -> 6 (5 inalcanzables + 1 degenerado silencioso). Hallazgos: F1 shortfall indistinguible (start==end -> 1 vertice -> `build_route_record` None); F2 `dest_items` desde la capa equivocada (KeyError 4, ruta a 224 m del destino real); F3 traceback en cero -> ruta falsa de 6 vertices (origen SE del destino); F4 huella de RESUME sin mascara ni segunda capa (2 rutas de 12, sin error); F5 `.tmp` rancio + `mode="a"` duplica filas; GPKG por archivo = 96 KB de overhead. Optimizacion: inversion valida en coste (34.1421 == 34.1421) pero geometria no bit-identica (31 vs 31 vertices, empates); coste de Fase 1 lo manda el area explorada (1 destino cerca 363 celdas / 1.0 ms vs 180 destinos 359657 celdas / 109.9 ms); `world_to_pixel` == `rowcol` en 2000/2000.
