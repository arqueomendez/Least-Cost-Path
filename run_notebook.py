"""
Script temporal de sesion - NO forma parte del proyecto.
Ejecuta LCP_MCP_Geometric.ipynb directamente con nbclient (sin browser).

Uso:
    python run_notebook.py          <- con Python 3.13 del sistema
  o bien:
    .venv\Scripts\python run_notebook.py
"""

import subprocess
import sys
import os

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
    )

    if result.returncode == 0:
        print("\n[run] ✓ Notebook ejecutado correctamente.")
        print(f"[run] Resultados guardados en {NOTEBOOK}")
    else:
        print(f"\n[run] ✗ Error durante la ejecucion (codigo {result.returncode}).")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
