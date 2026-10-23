r"""
Ejecuta LCP_MCP_Geometric.ipynb de forma no interactiva con nbconvert
(sin browser; timeout 30 min por celda). Documentado en README y AGENTS.md.

Uso:
    uv run python run_notebook.py
"""

import os
import subprocess
import sys

NOTEBOOK = "LCP_MCP_Geometric.ipynb"
VENV_PYTHON = os.path.join(os.getcwd(), ".venv", "Scripts", "python.exe")
# Timeout por celda en segundos (30 min para el bucle paralelo)
CELL_TIMEOUT = 30 * 60


def main():
    print(f"[run] Ejecutando notebook: {NOTEBOOK}")
    print(f"[run] Python: {VENV_PYTHON}")
    print(f"[run] Timeout por celda: {CELL_TIMEOUT}s\n")

    result = subprocess.run(
        [
            VENV_PYTHON,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}",
            "--ExecutePreprocessor.kernel_name=python3",
            NOTEBOOK,
        ],
        cwd=os.getcwd(),
        check=False,  # el returncode se inspecciona y propaga mas abajo
    )

    if result.returncode == 0:
        print("\n[run] ✓ Notebook ejecutado correctamente.")
        print(f"[run] Resultados guardados en {NOTEBOOK}")
    else:
        print(f"\n[run] ✗ Error durante la ejecucion (codigo {result.returncode}).")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
