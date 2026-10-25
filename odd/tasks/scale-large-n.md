# Feature: scale-large-n

Objetivo: aplicar las tres soluciones de escala evaluadas para N muy grandes (500–1000 pts, ráster ~12k×12k): (i) escritura incremental de rutas sin acumular en RAM, (ii) Celda 8 sin concat global, (iii) progreso con tqdm. La 4ª (tiling del FFT) queda diferida por diseño.
Branch: `feature/scale-large-n` (apilada sobre `LCP_MCP_Geometric`). Entrega: commits locales; push solo si el usuario lo pide.
Runner (tras cada tarea): `uv run pytest tests/ -q` + `uv run ruff check .` + `uv run ruff format --check .` + corrida funcional cuando toca celdas.

## Specs

- **S1** (solución i — memoria Fase 2/GPKG): la Celda 6 **no acumula `all_route_records`**. Fase 2 se consume como generador (`return_as="generator_unordered"` de joblib) y cada lote de origen se escribe **incrementalmente** a un GPKG temporal (`…gpkg.tmp`, `mode='a'` tras crear con `'w'`), con `os.replace` atómico al final → el GPKG final conserva la misma capa `rutas_unificadas`, mismos campos y CRS, y **nunca queda parcial** (el `finally` limpia el temporal si falla). Conteos: `n_rutas_escritas` para los echoes `Rutas escritas: X | esperadas: N` y el aviso `!= esperadas`. `routes_gdf = None` al final para que la Celda 7 lea desde disco (y no uses datos viejos en un kernel vivo). Orden de filas = orden de finalización (los tests y consumidores son orden-agnósticos; documentado).
- **S2** (solución ii — memoria Celda 8): el densificado **no concatena `xs/ys/idx` globales**: el `Parallel` se consume con `return_as="generator_unordered"` y cada chunk se escribe directo en el GPKG de puntos (batches fiona ≤200 000), con progreso por escrito/n_total. `route_idx` conserva la trazabilidad.
- **S3** (solución iii — progreso): `tqdm.auto` (tqdm ya es dependencia) sobre los tasks de Fase 1 y Fase 2; `verbose` de joblib bajado a 0 en esas dos fases para no duplicar ruido.
- **S4** (verificación de paridad — obligatoria): **baseline capturado ANTES del cambio** (última sesión post-merge): rutas=91 con sus campos; puntos=82 288 con hash del `bincount(route_idx)` (orden-agnóstico). Tras el cambio: misma corrida completa → 91 rutas con mismos campos, 82 288 puntos con **bincount idéntico**, 0 outputs de error; los 8 tests harness existentes (barrera/idempotencia/resume/validación) en verde; pytest+ruff/format limpios. Pins estructurales con RED→GREEN: `generator_unordered` en celdas 6 y 8, ausencia de `all_route_records`/concat global, `tqdm` presente.
- **S5** (docs): AGENTS (nota de escritura incremental + progreso), README (nota de escala: qué cambió y límite conocido de la Celda 7 con N enorme), CHANGELOG (entrada escala), este doc con Log.

## Tasks

- T1 — S1 — pending — inline — Celda 6: streaming Fase 2 + GPKG tmp→replace + conteos (pin RED primero)
- T2 — S2 — pending — inline — Celda 8: escritura por chunk sin concat global (pin RED)
- T3 — S3 — pending — inline — tqdm.auto en Fase 1/2 (verbose 0)
- T4 — S4 — pending — inline — Paridad funcional: baseline vs post (91 + 82 288 + bincount) + batería harness + corrida completa
- T5 — S5 — pending — inline — Docs (AGENTS/README/CHANGELOG) + cierre (doc/espejo/memoria)

## Log

- **L1** (petición del usuario, verbatim): "Aplica las soluciones de ### 8. Escala para N muy grandes" — refiriéndose a la evaluación entregada: (i) escritura incremental de rutas [único blocker real], (ii) densify sin concat global, (iii) tqdm progreso; FFT tiling quedaría diferido.
- **L2** (baseline de paridad capturado antes de tocar código): sesión post-merge `output/session_20261023_015752_Refactored` — rutas 91 (campos origin_id/dest_id/cost_total/length_m/n_vertices), puntos 82 288, `bincount(route_idx)` sha256 registrado en esta sesión (ver T4).
