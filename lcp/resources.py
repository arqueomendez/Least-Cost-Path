# lcp/resources.py

"""
Modelo de memoria por worker y sugerencia de número de workers.

El pico de RAM de la Fase 1 no lo domina el ráster sino las estructuras internas
de ``skimage.graph.MCP_Geometric``: en ``__init__`` construye un mapa de offsets
de forma ``(2, H, W)`` con dtype ``int64`` y ``find_costs`` agrega los costes
acumulados (``float64``) y el traceback (``int16``). El resultado medido, con
``fully_connected=True``, es **76 bytes por píxel y worker** más un sobrecosto
fijo de ~30 MB.

Mediciones (pico RSS de un proceso que solo ejecuta un MCP, raster float64 en
memmap, ``find_costs`` con destinos):

    forma        píxeles      pico      bytes/píxel
    2000x2000     4.0 M     320 MB        83.9
    6000x6000    36.0 M    2640 MB        76.9
    14355x11945 171.5 M   12459 MB        76.2

La Fase 2 (solo carga el traceback ``int16`` y reconstruye rutas) cuesta
**2 bytes por píxel y worker** (~2.07 medido en 36 Mpx, 1.58 en 171 Mpx).

Por eso un estimador del tipo "raster x 4.5" subestima por un factor ~2 en
rásteres grandes: para 171 Mpx predice 6,2 GB cuando el pico real es 12,2 GB, y
con 6 workers son 73 GB en vez de 37 GB, que es exactamente el desborde que se
observó en la corrida real.
"""

import math
import os

import psutil

# Constantes medidas (ver docstring del módulo). No ajustar a ojo: si se
# recalibran hay que repetir la sonda y actualizar los pines de los tests.
PHASE1_BYTES_PER_PIXEL = 76
PHASE2_BYTES_PER_PIXEL = 2
TRACEBACK_BYTES_PER_PIXEL = 2  # int16 = 2 B por celda en los .npz de Fase 1
FIXED_OVERHEAD_BYTES = 30 * 2**20  # intérprete + heap del worker

# Fracción de la RAM disponible que se puede comprometer. Deja aire para el
# sistema, otras aplicaciones y la fragmentación (el mapa de offsets necesita
# un bloque contiguo de 2*H*W*8 bytes por worker).
SAFETY_FRACTION = 0.8


def phase_peak_bytes(phase, shape):
    """
    Pico de RAM estimado de un worker de la fase dada, en bytes.

    ``phase`` es 1 o 2; ``shape`` es ``(filas, columnas)`` del ráster.
    """
    píxeles = int(shape[0]) * int(shape[1])
    if phase == 1:
        return PHASE1_BYTES_PER_PIXEL * píxeles + FIXED_OVERHEAD_BYTES
    if phase == 2:
        return PHASE2_BYTES_PER_PIXEL * píxeles + FIXED_OVERHEAD_BYTES
    raise ValueError(f"Fase desconocida: {phase!r} (use 1 o 2)")


def disk_bytes_for_phase1(shape, n_origins):
    """
    Espacio en disco que ocupan los .npz de Fase 1 (un traceback int16 por origen).

    Los archivos se guardan sin comprimir cuando ``KEEP_PHASE1_FILES=False``, así
    que el tamaño es predecible: ``origenes * H * W * 2`` bytes.
    """
    píxeles = int(shape[0]) * int(shape[1])
    return int(n_origins) * píxeles * TRACEBACK_BYTES_PER_PIXEL


def cpu_limits():
    """
    (núcleos físicos, núcleos lógicos) según el sistema; 1 si no se puede saber.
    """
    físicos = psutil.cpu_count(logical=False) or 1
    lógicos = psutil.cpu_count(logical=True) or físicos
    return físicos, lógicos


def detect_hardware(disk_path=None):
    """
    Inventario del hardware relevante para dimensionar workers.

    Devuelve un dict con RAM total/disponible (bytes), núcleos físicos/lógicos,
    y disco libre en ``disk_path`` (o en el directorio actual si es ``None``).
    """
    memoria = psutil.virtual_memory()
    físicos, lógicos = cpu_limits()
    ruta = disk_path or os.getcwd()
    disco = None
    try:
        disco = psutil.disk_usage(ruta).free
    except OSError:
        disco = None
    return {
        "ram_total": int(memoria.total),
        "ram_disponible": int(memoria.available),
        "cpu_fisicos": físicos,
        "cpu_logicos": lógicos,
        "disco_libre": disco,
    }


def suggest_n_jobs(
    phase,
    shape,
    n_tasks,
    available_bytes,
    cpu_count=None,
    safety=SAFETY_FRACTION,
):
    """
    Número de workers sugerido para la fase: el mayor que cabe en RAM sin pasar
    de la fracción de seguridad, acotado por tareas y por CPU.

    Nunca devuelve 0: si un solo worker no entra en la RAM disponible, devuelve 1
    y el aviso queda en :func:`advice_for_phase` (un worker puede fallar, pero
    ejecutar sin workers no tiene sentido).

    ``cpu_count`` es el tope de CPU (por defecto los núcleos físicos en Fase 1 y
    el doble de lógicos en Fase 2, que es mucho más liviana en CPU).
    """
    pico = phase_peak_bytes(phase, shape)
    presupuesto = max(0.0, float(available_bytes) * float(safety))
    por_ram = int(presupuesto // pico) if pico > 0 else 0

    físicos, lógicos = cpu_limits()
    if cpu_count is None:
        cpu_count = físicos if phase == 1 else lógicos * 2

    n = min(por_ram, int(cpu_count), max(1, int(n_tasks)))
    return max(1, n)


def advice_for_phase(
    phase,
    shape,
    n_tasks,
    hardware=None,
    override=None,
    safety=SAFETY_FRACTION,
):
    """
    Diagnóstico legible de una fase: hardware, pico por worker y workers sugeridos.

    Devuelve un dict con las claves ``fase``, ``pico_mb``, ``sugerido``,
    ``aplicado``, ``aviso`` (texto o ``None``) y ``lineas`` (lista de textos listos
    para imprimir). Si ``override`` supera lo que permite la RAM, ``aviso`` lo dice
    con el número concreto al que conviene bajar.
    """
    hw = hardware or detect_hardware()
    pico = phase_peak_bytes(phase, shape)
    sugerido = suggest_n_jobs(
        phase,
        shape,
        n_tasks,
        hw["ram_disponible"],
        safety=safety,
    )
    aplicado = min(int(override), sugerido) if override else sugerido

    linea_pico = (
        f"  Fase {phase}: pico estimado por worker {pico / 2**30:.2f} GB "
        f"({pico / (int(shape[0]) * int(shape[1])):.0f} B/px)"
    )
    linea_sug = (
        f"  Fase {phase}: workers sugeridos {sugerido} "
        f"(RAM disponible {hw['ram_disponible'] / 2**30:.1f} GB, "
        f"tareas {int(n_tasks)}, "
        f"CPU {hw['cpu_fisicos']}f/{hw['cpu_logicos']}l)"
    )
    lineas = [linea_pico, linea_sug]

    aviso = None
    if override and int(override) > sugerido:
        aviso = (
            f"AVISO: MAX_WORKERS_PHASE1={int(override)} excede lo que permite la "
            f"RAM disponible; se usaran {sugerido}. Con {int(override)} workers el "
            f"pico seria {pico * int(override) / 2**30:.1f} GB frente a "
            f"{hw['ram_disponible'] / 2**30:.1f} GB disponibles "
            f"(ver ArrayMemoryError en _offset_edge_map)."
        )
        lineas.append(f"  {aviso}")
    elif override:
        lineas.append(
            f"  MAX_WORKERS_PHASE1={int(override)} aplicado "
            f"(por debajo del limite seguro {sugerido})"
        )

    return {
        "fase": phase,
        "pico_mb": pico / 2**20,
        "sugerido": sugerido,
        "aplicado": aplicado,
        "aviso": aviso,
        "lineas": lineas,
    }


def format_hardware(hardware=None):
    """
    Texto de una línea con el hardware detectado.
    """
    hw = hardware or detect_hardware()
    disco = hw.get("disco_libre")
    texto_disco = f", disco libre {disco / 2**30:.1f} GB" if disco else ""
    return (
        f"Hardware: RAM total {hw['ram_total'] / 2**30:.1f} GB, "
        f"disponible {hw['ram_disponible'] / 2**30:.1f} GB, "
        f"CPU {hw['cpu_fisicos']} fisicos / {hw['cpu_logicos']} logicos{texto_disco}"
    )


def workers_that_fit(available_bytes, peak_bytes, safety=SAFETY_FRACTION):
    """
    Cuántos workers de ``peak_bytes`` caben en ``available_bytes`` (sin tope de CPU).
    """
    if peak_bytes <= 0:
        return 0
    return int((max(0.0, float(available_bytes)) * float(safety)) // peak_bytes)


def _pixels(shape):
    return int(shape[0]) * int(shape[1])


def rounded_gb(n_bytes):
    """
    Bytes a GB con un decimal, para mensajes.
    """
    return math.floor(n_bytes / 2**30 * 10) / 10
