# Kernel A* jerárquico como opción de cálculo (od_pair)

El usuario pidió evaluar la unificación de técnicas de las ramas en un solo método.
Decisión final: `LCP_MCP_Geometric` es el tronco; el A* jerárquico de `main` se porta
como **kernel alternativo seleccionable por el usuario** para el modo `od_pair` (1→1),
sin reemplazar el pipeline MCP existente. `LCP-parallel-tile` NO se unifica (queda
como feature futuro).

## Specs

- S1 (verbatim, Log L2): "No unifiques parallel-tile ya que, si bien puede ser una idea, la implementación es compleja y ha fallado. Debe quedar como feature futuro."
- S2 (verbatim, Log L3): "Evalua que el kernel rápido de 1x1 se incorpore como opción de calculo y no reemplace completamente lo realizado por lcp, y ello sea una seleccion por parte del usuario, que debe estar documentado en readme y en el notebook."

## Tasks

- T1 (S2) [inline] [commit 1] Portar kernel A* jerárquico: `a_star_search`/`reconstruct_path` (main:pathfinder) → `lcp/pathfinder.py`; `create_low_res_data`/`create_search_corridor` (main:processing) → `lcp/processing.py`.
- T2 (S2) [inline] [commit 1] `KERNEL = "mcp_geometric"` en Celda 3 del notebook; validación en Celda 4: default MCP, `corridor_astar` solo válido en modo `od_pair` (error claro en otros modos).
- T3 (S2) [inline] [commit 1] Rama de ejecución en Celda 6: `od_pair` + `corridor_astar` calcula la ruta con el A* jerárquico y escribe el mismo registro de ruta (GPKG/artefactos) que consume la Celda 7; el pipeline MCP queda intacto cuando `KERNEL="mcp_geometric"`.
- T4 (S2) [inline] [commit 2] Tests: unitarios del kernel portado (rutas conocidas, máscara, corredor) + integración de la selección de kernel; suite completa en verde.
- T5 (S1,S2) [inline] [commit 2] README: tabla de kernels, cuándo usar cada uno, limitaciones; nota de parallel-tile como feature futuro.

## Log

- L1 (verbatim): "Lo que necesito en este proyecto es que evalues el archivo main y sus ramas. Tienen diverso nivel de avance cada una. Lo más importante es que entiendas qué hace cada una, y evalues la posible unificación de las técnicas y métodos usadas en las ramas a un solo método que las reuna, siempre que resuelvan problemas que las otras no. Por ejemplo, ¿el método LCP_MPC_Geométric es suficientemente potente cuando se busca una ruta 1 a 1 en comparación con main? Realiza un análisis profundo y realiza una propuesta de implementación."
- L2 (verbatim): "Incorpora que todas las ramas, si no están presentes en local, se encuentran en https://github.com/arqueomendez/Least-Cost-Path"
- L3 (verbatim): "No unifiques parallel-tile ya que, si bien puede ser una idea, la implementación es compleja y ha fallado. Debe quedar como feature futuro. Evalua que el kernel rápido de 1x1 se incorpore como opción de calculo y no reemplace completamente lo realizado por lcp, y ello sea una seleccion por parte del usuario, que debe estar documentado en readme y en el notebook."
- L4: Análisis entregado. Tronco = LCP_MCP_Geometric (superset). MCP exacto pero Dijkstra sin dirección al objetivo: para 1→1 en rasters grandes explora casi toda la superficie; A* jerárquico de main es dirigido (fase 1 LR → corredor → fase 2 HR). Decisión: kernel opcional para od_pair.
- L5: Implementación completada en rama feat/corridor-astar-kernel. Evidencia: suite 267 passed (tests/test_corridor_astar.py: 16 tests nuevos); ruff clean; export nbconvert sincronizado (verifier S11 ok). End-to-end papermill od_pair 5→3 con KERNEL=corridor_astar: 1 ruta, 421 px, coste 39669.49, mismo flujo GPKG; corrida default mcp_geometric sin cambios. Commits: 5a38f51 (feat(kernel)), f45eb39 (docs(readme)), 6073eb1 (style). Aprendizaje de tests: barreras de 1 fila no bloquean la conectividad 8 (paso diagonal); máscara que cierra el paso barato no hace inalcanzable el destino, obliga a cruzar celdas caras.
- L7: Revision nativa (lineage review-c261759b13faeb6f, tier high, lentes readability+reliability, refuter): 2 hallazgos CRITICAL corroborados (R2-corridor-lr-oob, R3-astar-lowres-oob: indices LR sin clamp al grid recortado -> OOB en numba). Correccion acotada aplicada y commiteada (fd4327e, 31 lineas diff, test de regresion en borde; suite 268 passed). LA RUTA DE CORRECCION DEL PROVIDER QUEDO BLOQUEADA: el capture-correction-plan rechaza selector-mismatch dos veces (antes y despues de commitear la correccion); orden correcto aparente = declarar plan ANTES de editar/commitear. Estado del lineage: correction_required; se reporta el defecto, no se resetea autoridad. Entrega (push/PR) queda en decision del usuario.
- L6: Frontera de slice de trabajo: 3 commits (~590 líneas de cambio propio) sobre el tronco LCP_MCP_Geometric; el acumulado desde origin/main (25.3k líneas) hereda la historia de la rama tronco y no es producción de este slice.
