"""Robustez de la reconstruccion de rutas frente a tracebacks corruptos.

``reconstruct_path_from_traceback`` es ``@njit`` y no verifica limites por si
mismo: con un codigo de offset fuera de rango, o con un traceback que apunta
fuera del raster, leia memoria fuera del array y el proceso moria con un
segfault (un .npz danado tumbaba el worker entero en lugar de devolver None).

Los casos corruptos se ejecutan en un SUBPROCESO a proposito: si la funcion
vuelve a corromper la memoria, el test falla por el codigo de salida del hijo en
lugar de matar la suite de pytest.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import lcp.pathfinder as pf

ROOT = Path(__file__).resolve().parents[1]
PREPARAR = "\n".join(
    [
        "import numpy as np",
        "import lcp.pathfinder as pf",
        "OFF = pf.DEFAULT_OFFSETS.copy()",
    ]
)


def _run_in_child(cuerpo):
    """Ejecuta ``cuerpo`` en un interprete limpio con el repo en PYTHONPATH."""
    entorno = dict(os.environ)
    entorno["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", PREPARAR + "\n" + cuerpo],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=entorno,
    )


CASOS_CORRUPTOS = {
    # traceback a cero: el codigo 0 (offset (-1,-1)) manda al predecesor fuera
    # de la esquina; antes del endurecimiento esto era un segfault
    "traceback_a_cero": """
tb = np.zeros((40, 50), dtype=np.int16)
res = pf.reconstruct_path_from_traceback(tb, (0, 0), (30, 40), OFF)
print('RESULTADO', res)
""",
    # codigo de offset que no existe en la tabla
    "codigo_fuera_de_rango": """
tb = np.zeros((20, 20), dtype=np.int16)
tb[10, 10] = 99
res = pf.reconstruct_path_from_traceback(tb, (0, 0), (10, 10), OFF)
print('RESULTADO', res)
""",
    # codigo negativo
    "codigo_negativo": """
tb = np.zeros((20, 20), dtype=np.int16)
tb[10, 10] = -3
res = pf.reconstruct_path_from_traceback(tb, (0, 0), (10, 10), OFF)
print('RESULTADO', res)
""",
    # todos los codigos empujan hacia (-1,-1): desde (1,1) el predecesor seria (0,0)
    # y desde (0,0) ya sale del raster
    "predecesor_fuera_del_raster": """
tb = np.full((6, 6), 7, dtype=np.int16)
res = pf.reconstruct_path_from_traceback(tb, (3, 3), (1, 1), OFF)
print('RESULTADO', res)
""",
    # extremos fuera del raster
    "extremos_fuera": """
tb = np.zeros((10, 10), dtype=np.int16)
print('RESULTADO', pf.reconstruct_path_from_traceback(tb, (0, 0), (10, 10), OFF))
""",
    "extremos_negativos": """
tb = np.zeros((10, 10), dtype=np.int16)
print('RESULTADO', pf.reconstruct_path_from_traceback(tb, (-1, 0), (5, 5), OFF))
""",
}


@pytest.mark.parametrize("nombre", sorted(CASOS_CORRUPTOS))
def test_corrupt_traceback_is_rejected_without_crashing(nombre):
    hecho = _run_in_child(CASOS_CORRUPTOS[nombre])
    assert hecho.returncode == 0, (
        f"[{nombre}] el proceso hijo murio con {hecho.returncode}: "
        f"{hecho.stderr[-600:]}"
    )
    assert "RESULTADO None" in hecho.stdout, (
        f"[{nombre}] debia devolver None y devolvio: {hecho.stdout[-300:]}"
    )


def test_valid_path_still_reconstructs():
    # Control positivo: un traceback valido sigue devolviendo la ruta completa.
    from skimage.graph import MCP_Geometric

    costes = np.ones((12, 12), dtype=np.float64)
    mcp = MCP_Geometric(costes, fully_connected=True)
    _, tb = mcp.find_costs([(0, 0)], ends=[(11, 11)], find_all_ends=True)
    camino = pf.reconstruct_path_from_traceback(
        tb, (0, 0), (11, 11), np.asarray(mcp.offsets, dtype=np.int32)
    )
    assert camino is not None
    assert tuple(camino[0]) == (0, 0)
    assert tuple(camino[-1]) == (11, 11)
    assert camino.shape[1] == 2
    # el camino es conexo: cada paso es un offset de la tabla
    pasos = np.diff(camino, axis=0)
    assert np.all(np.abs(pasos) <= 1)
