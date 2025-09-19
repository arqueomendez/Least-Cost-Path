# lcp_network/utils.py

import os
import logging
import numpy as np
import psutil
import time
from tqdm import tqdm

from . import pathfinder

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def warm_up_numba(log):
    log.info("Calentando funciones de Numba para evitar conflictos en paralelo...")
    dummy_cost = np.ones((3, 3), dtype=np.float32)
    dummy_mask = np.ones((3, 3), dtype=bool)
    start = (0, 0); end = (2, 2)
    _, came_from, _ = pathfinder.a_star_numba_compiled(dummy_cost, -9999, start, end, 1.0, 1.0, 1.0, dummy_mask)
    came_from[1, 1] = 4
    _ = pathfinder.reconstruct_path_pixels_numba(came_from, start, (1, 1))
    _ = pathfinder.heuristic_numba(0, 0, 1, 1, 1, 1)
    log.info("Calentamiento de Numba completado.")

def resource_monitor(stop_event, records_list):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        mem = sum(p.memory_info().rss for p in [process] + process.children(recursive=True)) / (1024 * 1024)
        records_list.append({'cpu': cpu, 'mem_mb': mem})
        time.sleep(1)