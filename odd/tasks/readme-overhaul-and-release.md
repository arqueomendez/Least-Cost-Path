# Feature: readme-overhaul-and-release

Reescritura completa del README con diagrama Mermaid del flujo completo del
pipeline y de cada seccion, actualizado al estado actual del tronco
(343d5d4: 4 modos, 2 kernels, post-procesamiento), seguido de un release
versionado autorizado por el usuario.

## Specs

- S1 (verbatim, Log L1): "Quiero que actualices completamente el readme un un diagrama del flujo del funcionamiento completo y cada sección, para luego ejecutar un release"
- S2: El README debe quedar consistente con el tronco post-PR#3/PR#4 (kernel corridor_astar, HEURISTIC_WEIGHT, od_pair, cross_layers, salidas por raiz, 285 tests) — sin secciones desactualizadas ni referencias rotas.
- S3: Diagrama Mermaid del flujo completo (Celdas 1-9, dos fases, kernels, salidas) + secciones del README mapeadas a las etapas.
- S4 (Log L1): "para luego ejecutar un release" — release versionado en GitHub tras el merge del README.

## Tasks

- T1 (S2,S3) [inline] [commit 1] Reescritura completa del README: descripción, características, diagrama Mermaid del flujo, estructura, salidas, método de trabajo por celda, dependencias, instalación, modos, kernels, post-procesamiento, tests, troubleshooting.
- T2 (S2) [inline] [commit 1] CHANGELOG: promover la seccion "Sin version" a la version del release.
- T3 (S4) [inline] [commit 2] Release: tag versionado + GitHub Release con notas.
- T4 (S1-S4) [inline] [commit 1] Verificacion: consistencia README vs tronco, diagrama valido, suite verde.

## Log

- L1 (verbatim): "Quiero que actualices completamente el readme un un diagrama del flujo del funcionamiento completo y cada sección, para luego ejecutar un release"
- L2: Base = tronco LCP_MCP_Geometric 343d5d4 (PRs #3 y #4 mergeados). Ultimo tag: v0.0.2. CHANGELOG con seccion "Sin version" pendiente de versionar.
- L3: Implementacion delegada a gentle-ai-worker. README reescrito (703->689 lineas): 2 diagramas Mermaid (principal 18 nodos C1-C9 con ambas ramas del kernel; secundario 8 nodos del detalle corridor_astar), secciones reescritas, instalacion multi-SO conservada (solo 96->285 passed). Verificacion del parent: 2 bloques mermaid con ids unicos, 27 parametros citados todos existentes en el tronco, MASK_ALL_TOUCHED correctamente ausente, suite 285 passed. CHANGELOG promovido a 0.1.0 (coincide con pyproject.toml) con las entradas cross-layer-mode y corridor-astar-kernel anadidas para cubrir el delta completo v0.0.2..HEAD. Commit 184f656.
- L4: Release v0.1.0 (tag + GitHub Release), autorizado en L1.
