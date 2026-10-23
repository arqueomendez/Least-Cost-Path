# Feature: star-origin-mode

Objetivo: modo "estrella" en el notebook — un origen seleccionado → rutas de mínimo coste hacia todos los demás puntos, sin pares entre ellos; interfaz simple de selección/ejecución (celda de parámetros + papermill), sin selección interactiva.
Branch: `feature/star-origin-mode` (apilada sobre `feature/notebook-hardening`). Estrategia de entrega: `chain_strategy=feature-branch-chain` (heredada); commits locales, sin push.
Runner: `uv run pytest tests/` · Verificación funcional: `uv run jupyter nbconvert --to notebook --execute --output _exec_check.ipynb --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 LCP_MCP_Geometric.ipynb` (y `uv run papermill ...` para el modo parametrizado).

## Specs

- **S1** — Celda 3 = panel de parámetros: `ANALYSIS_MODE = "all_pairs"` (enum `"all_pairs"` | `"single_origin"`) y `ORIGIN_ID = None`. Default `"all_pairs"` = comportamiento N×(N−1)/2 actual intacto (regresión obligatoria: 91 rutas con los datos de ejemplo).
- **S2** — Semántica del modo estrella: con `single_origin`, exáctamente **N−1 rutas**, todas con `origin_id == ORIGIN_ID`, **ningún par entre los destinos** (por construcción: un solo MCP desde O, sin filtro `did > origin_id` — obligatorio porque los destinos con id menor sí corresponden). Simetría de `MCP_Geometric`: ruta O→P ≡ P→O (no se pierde nada).
- **S3** — Validación temprana en Celda 4: modo desconocido → `ValueError` listando los modos válidos; `single_origin` sin `ORIGIN_ID` o con ID inexistente → `ValueError` listando los IDs disponibles.
- **S4** — Echo de modo efectivo al arrancar la ejecución (Celda 6): `MODO: ...` con rutas **esperadas** según el modo (N−1 o N(N−1)/2) y las **reales** escritas.
- **S5** — Resaltado visual (no interactividad): en modo `single_origin`, los mapas de las Celdas 5 y 7 pintan el punto origen con color/tamaño distinto.
- **S6** — Ejecución parametrizada: `papermill` como dependencia dev (`uv add --dev papermill`) y comando documentado: `uv run papermill LCP_MCP_Geometric.ipynb out.ipynb -p ANALYSIS_MODE single_origin -p ORIGIN_ID <id>`.
- **S7** — Docs: README y AGENTS con la tabla de parámetros nueva, el comando papermill, y la equivalencia conceptual SSSP vs APSP (al estilo `source=` de NetworkX: mismo flujo, parámetro que acota el alcance).
- **S8** — Tests y verificación: pins estructurales del modo con RED→GREEN observado; corrida funcional en modo estrella (N−1 rutas, todas desde O, ninguna entre destinos) + corrida de regresión `all_pairs` (91); `uv run pytest tests/` y `ruff check`/`format --check` limpios.

## Tasks

- T1 — S6 — inline — done — c4d889c — `papermill 2.7.0` como dev dep + `requirements.txt` re-exportado (149 pins, anclas S15 pasan).
- T2 — S1,S2,S3,S4,S5 — done — d59dc50 — RED observado (5 pins fallaban) → parche celdas 3/4/6 + resaltado 5/7 + tag `parameters` en Celda 3 (requisito de S6: sin él papermill inyectaría al inicio y la Celda 3 pisaría los overrides; pin añadido con RED→GREEN) → GREEN 30 passed.
- T3 — S1 — done — d59dc50 — Export regenerado: 29 menciones de `ANALYSIS_MODE|single_origin|ORIGIN_ID`, 0 de "Proyecto Vannia", drift de contenido vs notebook = 0.
- T4 — S7 — done — 476732f — README (bullets, tabla Celda 3, sección «Modo single_origin» con comando papermill) y AGENTS (modos + papermill, fila de tests). Ledger corregido aquí tras advisory del verificador.
- T5 — S8 — done — c4d889c..12e6481 — Corrida estrella vía papermill (exit 0, 10/10 celdas): echo `MODO: single_origin (origen 7) | rutas esperadas: 13`, `Rutas escritas: 13 | esperadas: 13`, GPKG con exactamente 13 filas todas `origin_id=7` y destinos 1..14 sin 7, cero pares no-nulos, 0 outputs de error. Regresión all_pairs vía nbconvert (exit 0): `MODO: all_pairs (N x N) | rutas esperadas: 91`, `Rutas escritas: 91 | esperadas: 91`, 13/13 tracebacks, pares origin<dest íntegros, 0 errores. Batería final: pytest **30 passed**, `ruff check .` limpio, `format --check .` 19 archivos. Commits de pruebas/gitignore `12e6481`.
- T6 — S1-S8 — done — integración del verificador `gentle-ai-verify` (task mvkcs4jk-1-8b3j): **VERIFY** + advisory de ledger corregido + sus 5 probes adoptados y ejecutados (**35 passed**, ruff/format limpios).
- T7 — S1-S8 — done — cierre en este mismo commit: veredicto en L7, espejo Engram actualizado, árbol limpio.

## Log

- **L1** (petición original del usuario, verbatim): "Quiero calcular también una versión donde seleccione un punto y se calculen las rutas de minimo coste hacia todos los demás pero no entre ellos. Evalua cómo realizarlo."
- **L2** (respuesta custom del usuario al questionnaire, verbatim): "No me gusta la forma interactiva de selección. A nivel de interface humano-maquina, debería haber un modo simple de seleccionar o ejecutar este modo 1 contra todos. Evalua en línea también opciones recomendadas para casos como este"
- **L3** (decisión del usuario, verbatim): "Celda + papermill (recomendado)" — modalidad de ejecución CLI.
- **L4** (evidencia de investigación en línea): convención SSSP vs APSP en herramientas de grafos (NetworkX expone `source=` como parámetro, no herramienta distinta); celda etiquetada `parameters` + `papermill -p` es el estándar industry (papermill docs, SageMaker, Airflow, Kestra — nbconvert no admite inyectar parámetros: issues jupyter/nbconvert#681, #1281); enum validado + sentinel `None` (PEP 661, Kestra ENUM inputs, Fowler feature toggles).
- **L5** (T2+T3): commit `d59dc50`. RED 5 failed → GREEN 30 passed; tag `parameters` descubierto como requisito implícito de S6 y añadido con su pin.
- **L6** (compuerta RDD, solo lectura): ASSESS sobre `feature/notebook-hardening..HEAD` → **risk `high`, `reviewDue: true`** (señal `process_boundary/shell_process` en `LCP_MCP_Geometric.py`, 10 paths, 1605 líneas) con plan `writerSelfVerification + independentVerifier`. **No se inicia lineage nuevo** (directiva L16 del feature anterior: la herramienta de revisión no tiene solución en esta sesión); el verificador independiente exigido por el plan se lanzó como subagente (`gentle-ai-verify`, task `mvkcs4jk-1-8b3j`) — integración pendiente.
- **L7** (verificador `gentle-ai-verify`, task mvkcs4jk-1-8b3j): **VERDICTO VERIFY** — S1-S8 PASS con evidencia (30 tests, ruff/format limpios, export 149/149 pins reproducible byte-a-byte, 0 drift del export, GPKG estrella = 13 filas solo `origin_id=7` con 7 pares OD byte-idénticos a la corrida de regresión, validaciones y echoes observados en ambos artefactos con 0 errores). Advisories: (1) ledger T4 `pending` → corregido; (2) des-acentuado cosmético de comentarios en la rama `all_pairs` → **aceptado** (solo comentarios, sin cambio de salida; no re-tocamos el candidato). Adoptados sus 5 probes durables (cerraron la hueca real: el pin de S2 se satisfacía con el texto del comentario de la propia rama) → **35 passed**. No verificado por el subagente (conocido): ruff versión fijada (fallback `uv tool run` documentado en AGENTS) y re-ejecución del notebook (hecha por el padre en T5). Edge anotado: `N=1` en modo estrella genera un task degenerado (fuera del dominio real).
