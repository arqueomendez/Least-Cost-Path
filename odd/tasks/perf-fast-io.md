# Feature: perf-fast-io

Objetivo: aplicar en orden las mejoras de rendimiento aprobadas sobre la evaluación con benchmarks — (1) política híbrida de compresión del npz, (2) escritura de la Celda 8 con pyogrio, (3) vista del memmap sin copia — sin tocar FFT, Fase 2/densify ni imports.
Branch: `feature/perf-fast-io` (apilada sobre `feature/scale-large-n`). Entrega: commits locales.
Runner (tras cada tarea): `uv run pytest tests/ -q` + `uv run ruff check .` + `uv run ruff format --check .`; corrida funcional de paridad al cierre.

## Evaluación de rendimiento (evidencia — benchmarks en esta máquina)

| Etapa | Medición | Veredicto y decisión del usuario |
|---|---|---|
| 1. MCP Fase 1 | Cython (`_mcp*.pyd`), 306 ns/celda → ~48 s/origen en raster real (158 Mpx) | Sin cambio posible (ya en C; reemplazar = riesgo alto, techo bajo). Early-stop no aplica (todos los destinos). |
| 2. Compresión npz | int16 316 MB: `savez_compressed` **29,0 s → 81 MB** vs `savez` **0,3 s → 316 MB** (int8 200 MB: 14,2 vs 0,2 s) | **Aprobado con híbrido**: "no comprimir cuando KEEP_PHASE1_FILES=False pero permitir la reanudación" — comprimir solo si los npz persisten (KEEP=True); np.load lee ambos. |
| 3. Copia del raster | `np.array(memmap)` = **0,82 s + 1,27 GB/worker/origen**; MCP **no muta** la entrada (probe: False) | **Aprobado**: vista directa del memmap + estimador de RAM ×5.5 → ×4.5. |
| 4. Escritura Celda 8 | fiona bucle **2,04 s** vs pyogrio **0,71 s** /100k pts → **2,9×** | **Aprobado**: pyogrio por chunk. |
| 5. FFT Celda 9 | 0,18 s @4096²; `workers=-1` → 1,0× | "No modificar" (verbatim). |
| 6. Fase 2 y densificado | Numba µs–ms; densify 0,5 s/82k | "No modificar" (verbatim). |
| 7. Overhead fijo | kernel+imports ~5–8 s | "No modificar" (verbatim). |

## Specs

- **S1** — La evaluación anterior queda documentada en este documento (y resumida en CHANGELOG/AGENTS al cierre).
- **S2** (compresión híbrida) — El worker de Fase 1 comprime el npz **solo si `shared_data["compress_npz"]` es true**: la Celda 6 pasa `KEEP_PHASE1_FILES` (False → `np.savez` plano y rápido; True → `np.savez_compressed` para que los archivos persistentes de la reanudación sean chicos). Default `True` para llamadores sin la clave (compatibilidad). `np.load` debe leer ambos (reanudación funciona en los dos sentidos). Test RED primero: tipo de almacenamiento (ZIP_STORED vs ZIP_DEFLATED) + phase2 sobre npz plano.
- **S3** (vista sin copia) — El worker entrega el memmap directo a `MCP_Geometric` (ramas bbox y full), eliminando `np.array(...)`; el estimador `_calculate_n_jobs_for_phase` baja de ×5.5 a ×4.5 con comentario actualizado. Cambio de invariante (misma salida): verificado por los tests numéricos existentes (8√2) + corrida de paridad integral (T4).
- **S4** (pyogrio Celda 8) — La escritura de puntos pasa de bucle `fiona.writerecords` a `gdf.to_file` por chunk (`mode='w'` al crear, `'a'` después), sin `schema`/`BATCH`, con progreso por chunk; se elimina `import fiona` de la Celda 8 si queda sin uso. Pin RED: `writerecords` ausente y `to_file` presente.
- **S5** — **No modificar**: FFT de la Celda 9, matemática de Fase 2 y densificado, overhead de kernel/imports (verbatim del usuario).
- **S6** (verificación) — Batería completa verde + corrida funcional con paridad: 91 rutas, 82 288 puntos, `sorted(bincount)` sha idéntico (`e79f4274…`), 0 errores, y prueba de que la reanudación sigue funcionando con npz plano (test) y con comprimido (ya cubierto).

## Tasks

- T1 — S2 — pending — inline — Compresión híbrida (RED: ZIP_STORED/ZIP_DEFLATED + phase2 sobre plano)
- T2 — S4 — pending — inline — Pyogrio en Celda 8 (RED: pin writerecords/to_file)
- T3 — S3 — pending — inline — Vista del memmap sin copia + estimador ×4.5
- T4 — S6 — pending — inline — Verificación integral: batería + corrida de paridad (91/82288/sha)
- T5 — S1,S5 — pending — inline — Docs (CHANGELOG/AGENTS/README) + cierre (doc, espejo, memoria)

## Log

- **L1** (directivas del usuario, verbatim): "2. Compresión del npz: no comprimir cuando KEEP_PHASE1_FILES=False pero permitir la reanudación. 3. Copia del raster: aplica. 4. Escritura de la Celda 8: aplica. 5. FFT de la Celda 9: No modificar. 6. Fase 2 y densificado: no modificiar. 7. Overhead fijo (kernel + imports): No modificar. Documenta esta evaluación y eje ejecuta en el orden que propusiste."
- **L2** (orden de ejecución): compresión → pyogrio Celda 8 → vista sin copia (el orden propuesto en la evaluación). Nota: el dtype real del traceback medido es **int16** (2 B/celda → 316 MB en raster real; AGENTS decía int8 — corregir en T5).
