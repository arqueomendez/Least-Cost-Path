# lcp_parallel_tile/orchestrator.py

import os
import logging
import re
import threading
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed

# Importaciones relativas desde el mismo paquete
from . import tasks
from . import worker
from . import geoutils
from . import utils

def run_analysis(config):
    """
    Orquesta el análisis completo: gestiona sesiones, reanudación, ejecución
    paralela, logging y fusión de resultados.

    Args:
        config: Un objeto o módulo que contiene todos los parámetros de configuración.

    Returns:
        str: La ruta al directorio de la sesión que se ha procesado.
    """
    # --- Configuración de la sesión y reanudación ---
    os.makedirs(config.BASE_PROCESSING_FOLDER, exist_ok=True)
    main_log = logging.getLogger("MasterLogger")
    all_possible_tasks = tasks.prepare_tasks(config, main_log)
    
    session_to_resume = None
    completed_tile_ids = set()
    try:
        existing_sessions = sorted([d for d in os.listdir(config.BASE_PROCESSING_FOLDER) if d.startswith('flow_network_session_')])
    except FileNotFoundError:
        existing_sessions = []

    if existing_sessions:
        latest_session_name = existing_sessions[-1]
        latest_session_path = os.path.join(config.BASE_PROCESSING_FOLDER, latest_session_name)
        session_log_path = os.path.join(latest_session_path, "registro_de_procesamiento.log")
        if os.path.exists(session_log_path):
            with open(session_log_path, 'r') as f:
                for line in f:
                    if "ÉXITO en worker para tile" in line:
                        match = re.search(r'tile_\d+_\d+', line)
                        if match: completed_tile_ids.add(match.group(0))
        if len(completed_tile_ids) < len(all_possible_tasks):
            session_to_resume = latest_session_name

    if session_to_resume:
        SESSION_OUTPUT_DIR = os.path.join(config.BASE_PROCESSING_FOLDER, session_to_resume)
        main_log.info(f"MODO REANUDACIÓN: Reanudando sesión '{session_to_resume}'.")
        main_log.info(f"{len(completed_tile_ids)} tiles ya completados.")
    else:
        if existing_sessions: main_log.info(f"La última sesión '{existing_sessions[-1]}' está completa.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder_name = f'flow_network_session_{timestamp}'
        SESSION_OUTPUT_DIR = os.path.join(config.BASE_PROCESSING_FOLDER, session_folder_name)
        main_log.info(f"MODO NUEVO: Creando sesión: {session_folder_name}")
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)

    tasks_to_run = [task for task in all_possible_tasks if task['tile_id'] not in completed_tile_ids]
    main_log.info(f"Se procesarán {len(tasks_to_run)} tiles pendientes en esta ejecución.")
    
    monitoring_records = []
    
    if tasks_to_run:
        session_log_path = os.path.join(SESSION_OUTPUT_DIR, "registro_de_procesamiento.log")
        session_log = logging.getLogger("SessionLogger"); session_log.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if session_log.hasHandlers():
            session_log.handlers.clear()
        session_fh = logging.FileHandler(session_log_path, mode='a'); session_fh.setFormatter(formatter); session_log.addHandler(session_fh)

        # --- [MODIFICACIÓN] Añadir el nuevo parámetro a la configuración del worker ---
        worker_config = {
            'cost_raster_path': config.COST_RASTER_PATH,
            'mask_shapefile_path': config.MASK_SHAPEFILE_PATH,
            'session_output_dir': SESSION_OUTPUT_DIR,
            'downsampling_factors': config.DOWNSAMPLING_FACTORS,
            'corridor_buffer_pixels': config.CORRIDOR_BUFFER_PIXELS,
            'heuristic_weight': config.HEURISTIC_WEIGHT,
            'TILE_EDGE_BUFFER': config.TILE_EDGE_BUFFER # <-- AÑADIDO
        }
        full_tasks = [(task, worker_config) for task in tasks_to_run]
        
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=utils.resource_monitor, args=(stop_event, monitoring_records))
        start_time = time.time()
        monitor_thread.start()
        
        results = Parallel(n_jobs=config.N_JOBS)(delayed(worker.tile_worker)(task, cfg) for task, cfg in tqdm(full_tasks, desc="Procesando Tiles"))
        
        stop_event.set(); monitor_thread.join()
        total_duration = time.time() - start_time
        main_log.info(f"Proceso paralelo completado en {total_duration:.2f} segundos.")
        
        exitos = 0
        for status, tile_id, detail in results:
            if status == "Éxito":
                session_log.info(f"ÉXITO en worker para tile {tile_id}: Se calcularon {detail} rutas.")
                exitos += 1
            else:
                session_log.error(f"FALLO en worker para tile {tile_id}: {detail}")
        main_log.info(f"RESUMEN: {exitos} de {len(tasks_to_run)} tiles procesados con éxito.")
    else:
        main_log.info("No hay nuevas tareas que calcular.")
    
    geoutils.merge_results(SESSION_OUTPUT_DIR, os.path.join(SESSION_OUTPUT_DIR, "red_de_flujo_completa.gpkg"), main_log)

    if monitoring_records:
        avg_cpu = np.mean([r['cpu'] for r in monitoring_records])
        peak_mem = np.max([r['mem_mb'] for r in monitoring_records])
        main_log.info("\n--- Reporte de Rendimiento ---")
        main_log.info(f"Uso promedio de CPU: {avg_cpu:.2f}%")
        main_log.info(f"Pico de uso de Memoria: {peak_mem:.2f} MB")
        
    return SESSION_OUTPUT_DIR