# Feature: fix-review-advisories

Corregir los 8 advisory no bloqueantes del review nativo del kernel
`corridor_astar` (lineage review-3c4a31074f90ca7a) hasta llegar a cero,
sin cambiar el comportamiento ya aprobado salvo donde el advisory lo pide.

## Specs

- S1 (verbatim, review finding R2-corridor-gating-scatter): "An early 'if _KERNEL_CORRIDOR: ... else: <mcp pipeline>' split (or a dedicated function) would confine the kernel switch to one place."
- S2 (verbatim, review finding R2-unused-closed-copy): "assigns '_closed = mask[300:302, 296:305].copy()' but never reads it; the very next line mutates the mask in place."
- S3 (verbatim, review finding R3-corridor-same-pixel-silent-loss): "if the selected origin and destination map to the same pixel, a_star_search immediately reports a one-pixel path with cost 0, ... the requested pair is silently lost and no destinos_inalcanzables entry is written."
- S4 (verbatim, review finding R3-corridor-unreachable-not-materialized): "When corridor_astar finds no route, the unreachable pair exists only on stdout: cost_scalars_index stays empty ... whereas the MCP path materializes unreachable pairs in that layer (S8)."
- S5 (verbatim, review finding R3-cost-consistency-tests-tautological): "test_cost_matches_direct_astar recomputes pf.accumulate_cost on the very path corridor_astar_route just returned ... so equality is trivially true."
- S6 (verbatim, review finding R3-heuristic-admissibility-no-runtime-guard): "heuristic_weight defaults to 1.0, which is admissible only when every traversable cell cost is >= 1.0 ... no runtime detection, warning, or user-facing knob."
- S7 (verbatim, review finding R3-plan-b-quadratic-openset): "a_star_search manages the open set by vstack-ing one row per expansion and doing a full argmin scan per pop, making the search quadratic in explored cells."
- S8 (verbatim, review finding R3-reconstruct-path-limit-oob): "the unconditional path[count] = ... afterwards then writes at index == size, an out-of-bounds write under numba without bounds checking."

## Tasks

- T1 (S1) [inline] [commit 1] Celda 6: split temprano `if _KERNEL_CORRIDOR` con rama propia; eliminar guards dispersos `if not _KERNEL_CORRIDOR`.
- T2 (S2) [inline] [commit 1] test_mask_forbids_area: usar o eliminar `_closed`.
- T3 (S3) [inline] [commit 2] corridor od_pair mismo-pixel: registro explicito (ruta degenerada o destinos_inalcanzables), nunca silencio.
- T4 (S4) [inline] [commit 2] corridor sin ruta: materializar par inalcanzable en destinos_inalcanzables y capa rutas siempre presente.
- T5 (S5) [inline] [commit 2] Tests de coste: comparar contra coste independiente (suma de costes de celdas del path), no re-llamada del mismo acumulador.
- T6 (S6) [inline] [commit 1] Admisibilidad: deteccion runtime de costes < 1.0 con aviso; knob user-facing `HEURISTIC_WEIGHT`.
- T7 (S7) [inline] [commit 1] Plan B: open set con heap (heapq), complejidad log, sin vstack/argmin.
- T8 (S8) [inline] [commit 1] reconstruct_path: guard del write final cuando count == limit.
- T9 (S3-S8) [inline] [commit 2] Suite completa verde + ruff + registro en este doc.

## Log

- L1 (verbatim): "Corrige los 8 advisory no bloqueantes del review hasta llegar a cero"
- L2: Base = tronco LCP_MCP_Geometric e5689e4 (post-merge PR #3). Rama fix/review-advisories. Los 8 findings y claims verbatim estan en el L8 de odd/tasks/corridor-astar-kernel.md de esta misma rama tronco.
- L3: Implementacion delegada a gentle-ai-worker (un solo escritor, superficies acotadas). Suite 279 passed (+10 tests nuevos: T6 x4 capsys, T7 x2 heap, T8 x1 ciclo, T3/T4 x2 e2e + 1 pin MCP); ruff check clean; export nbconvert regenerado e idempotente. Decisiones de pins: (1) pin del guard disperso reemplazado por count de gating (2x if _KERNEL_CORRIDOR, 1x if not); (2) e2e unreachable endurecido a presencia obligatoria de capa rutas_unificadas vacia; (3) ningun pin de comportamiento debilitado. Excepciones TDD documentadas: T5 sin RED posible (corrige tautologia), T1/T2 verificacion estructural. Heap: desempate de rutas de coste identico puede diferir del argmin anterior (documentado en docstring). ruff format --check: desviaciones preexistentes sin cambios netos (no se reformateo codigo no tocado).
- L4: Commits por unidades + assess RDD + review nativo sobre el slice.
