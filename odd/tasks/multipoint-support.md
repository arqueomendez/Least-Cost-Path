# Feature: multipoint-support

Objetivo: soporte para capas de puntos con geometrías **MultiPoint** (hoy crashean con `AttributeError: 'MultiPoint' object has no attribute 'x'` en la carga de la Celda 4 y en el acceso `.x` de la Celda 5), con la semántica elegida por el usuario: **explosión con IDs derivados**.
Branch: `feature/multipoint-support` (apilada sobre `LCP_MCP_Geometric`). Entrega: commits locales; push solo si el usuario lo pide.
Runner (tras cada tarea): `uv run pytest tests/ -q` + `uv run ruff check .` + `uv run ruff format --check .`; verificación funcional con harness/corrida.

## Specs

- **S1** (semántica — decisión del usuario verbatim L2: "Explosión con IDs derivados") — en `load_points_as_dict`: `Point` → id original sin cambio (regresión); `MultiPoint` con **1** parte → id original (equivale a un punto); `MultiPoint` con **k ≥ 2** partes → **k nodos** con ids derivados **`id_original * 1000 + j`** (j = parte 1..k, k ≤ 999; ej. id 7 → 7001, 7002, 7003). Regla determinista por orden del archivo. **Colisión** de un derivado con cualquier otro id emitido → `ValueError` listando la colisión (los ids siguen siendo enteros únicos: lo exigen npz `dest_ids` int64, columnas GPKG y los filtros `did > origin_id`). Geometrías que no sean Point/MultiPoint → `ValueError` con el tipo. Se imprime un resumen (`N features MultiPoint explotadas en M puntos`). Contrato aguas abajo intacto: dict `id → (x, y)`.
- **S2** (gráfica) — Celdas 5 y 7 dejan de usar `.x` sobre la geometría cruda releída del archivo y grafican desde el dict ya cargado (todos los nodos explotados, incluida la estrella de modo estrella sobre `all_points_for_validation`).
- **S3** (tests, RED primero) — fixture MultiPoint en tmp: regla exacta de ids derivados; k=1 conserva id; colisión → `ValueError`; `Point` simple intacto (regresión); geometría LineString → `ValueError`; pin de que Celdas 5/7 no usan `point.x`/`.x` sobre geometría cruda.
- **S4** (verificación funcional) — suite completa verde + corrida funcional con datos MultiPoint (harness celdas 1–4 sobre fixture multipoint: validación pasa con los ids derivados) + regresión de la corrida muestra (91 rutas, sin cambios).
- **S5** (docs) — README (entrada multipoints + regla de ids derivados), AGENTS (fila `data_loader` + regla), CHANGELOG.

## Tasks

- T1 — S1,S3 — pending — inline/delegado — Implementación data_loader + celdas 5/7 + tests RED→GREEN
- T2 — S4 — pending — inline — Verificación funcional (harness multipoint + corrida muestra + batería)
- T3 — S5 — pending — inline — Docs (README/AGENTS/CHANGELOG) + cierre (doc, espejo, memoria, commits)

## Log

- **L1** (petición del usuario, verbatim): "Necesito que se añada soporte para multipoints"
- **L2** (decisión del usuario, verbatim — questionnaire): "Explosión con IDs derivados" — cuando una feature MultiPoint trae varias coordenadas para el mismo id, cada coordenada se vuelve un nodo real con id derivado (fue la opción marcada como la más invasiva: N crece y la trazabilidad de ids cambia).
- **L3** (fallo reproducido, evidencia): fixture MultiPoint con `load_points_as_dict` → `AttributeError: 'MultiPoint' object has no attribute 'x'`; mismo error en el acceso `.x` de la Celda 5.
