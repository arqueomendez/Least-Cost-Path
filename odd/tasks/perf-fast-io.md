# Feature: perf-fast-io

Objetivo: aplicar en orden las mejoras de rendimiento aprobadas sobre la evaluación con benchmarks — (1) política híbrida de compresión del npz, (2) escritura de la Celda 8 con pyogrio, (3) vista del memmap sin copia — sin tocar FFT, Fase 2/densify ni imports.
Branch: `feature/scale-large-n` — **corrección (L4)**: los commits de este feature (`d1e01ea..5ab9346`) quedaron en la rama `feature/scale-large-n` (no se creó `feature/perf-fast-io`; el checkout se omitió al iniciar). Los dos features (scale-large-n + perf-fast-io) viven juntos en esa rama, apilada sobre `LCP_MCP_Geometric`. Entrega: commits locales → **push de la rama en L4**.
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

- T1 — S2 — done — `d1e01ea` + `45860bc` — Compresión híbrida: RED (ZIP_STORED con `compress_npz=False`) → worker con `saver = np.savez_compressed if shared_data.get("compress_npz", True) else np.savez` y Celda 6 pasando `KEEP_PHASE1_FILES`. Test prueba política de almacenamiento, igualdad de contenido entre formatos y **Fase 2 sobre npz plano**. 75 passed + fixes RUF059/format.
- T2 — S4 — done — `2d599eb` — Pyogrio en Celda 8: RED (pin `writerecords`) → `gdf.to_file` por chunk (`mode w→a`, capa = stem del archivo, el mismo default de fiona), `schema`/`BATCH` fuera, `import fiona` eliminado, header actualizado. 76 passed.
- T3 — S3 — done — `1828d5d` — Vista del memmap sin copia (ramas bbox y full) + estimador ×5.5→×4.5 con docstring corregido a **int16** (medido). RED del estimador (550 MB/8 MB → 12 workers, no 10) → GREEN; numéricos con vista: 7 passed (8√2). 77 passed.
- T4 — S6 — done — **PARIDAD OK** (corrida completa exit 0): `MODO: all_pairs`, `Rutas escritas: 91 | esperadas: 91`, `Escritos 82,288 / 82,288`, campos idénticos, **sorted-bincount sha `e79f4274…` idéntico al baseline**, 0 errores; ejercitó los 3 cambios end-to-end (npz plano leídos por Fase 2, pyogrio, vista).
- T5 — S1,S5 — done — Docs (CHANGELOG/AGENTS/README) + cierre (espejo, memoria) — commit de cierre.

## Log

- **L1** (directivas del usuario, verbatim): "2. Compresión del npz: no comprimir cuando KEEP_PHASE1_FILES=False pero permitir la reanudación. 3. Copia del raster: aplica. 4. Escritura de la Celda 8: aplica. 5. FFT de la Celda 9: No modificar. 6. Fase 2 y densificado: no modificiar. 7. Overhead fijo (kernel + imports): No modificar. Documenta esta evaluación y eje ejecuta en el orden que propusiste."
- **L2** (orden de ejecución): compresión → pyogrio Celda 8 → vista sin copia (el orden propuesto en la evaluación). Nota: el dtype real del traceback medido es **int16** (2 B/celda → 316 MB en raster real; AGENTS decía int8 — corregir en T5).
- **L3** (T1-T4): commits `d1e01ea`/`45860bc`/`2d599eb`/`1828d5d` — RED→GREEN en los 3 cambios con sus pins; paridad final **OK**: sha `e79f4274…` idéntico al baseline de scale-large-n (91 rutas, 82 288 pts, campos iguales, 0 errores). Los benchmarks que sustentan cada veredicto están en la tabla «Evaluación de rendimiento» de este documento.
- **L4** (commit+push del usuario, verbatim: "realiza commit y push"): no había cambios pendientes de commit en el árbol; se corrigió la cabecera de branch de este documento (el checkout a `feature/perf-fast-io` nunca ocurrió — los10 commits sin publicar están en `feature/scale-large-n`) y se subió esa rama a origin con los10 commits (5 de scale-large-n + 6 de perf-fast-io contando los docs… total10: scale T1-T3+docs =4, perf =6). `origin/LCP_MCP_Geometric` queda a 0 commits de retraso; la rama principal NO recibe los merges de estos dos features (pendiente de decisión del usuario, como en iteraciones anteriores).
- **L5** (merge + push a la principal, verbatim: "mergiá a la principal y subilo"): merge `--no-ff` de `feature/scale-large-n` (con los 11 commits de escala+perf) en `LCP_MCP_Geometric` sin conflictos (el árbol quedó idéntico a la rama; el phantom-diff de nbstripout de la iteración anterior no se repitió porque ambos lados ya llevaban el notebook sin outputs). Verificación post-merge: **77 passed**, ruff/format verdes, corrida completa exit 0 con `Rutas escritas: 91 | esperadas: 91`, 0 errores. Push de la principal ejecutado.
