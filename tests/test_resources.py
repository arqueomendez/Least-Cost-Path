"""Modelo de memoria por worker: pines de las mediciones y de la sugerencia.

Las constantes se calibraron midiendo el pico RSS de un proceso que solo ejecuta
un ``MCP_Geometric`` (ver el docstring de ``lcp/resources.py``). Estos tests
fallan si alguien cambia el modelo sin repetir la medición.
"""

import pytest

import lcp.resources as res

GB = 2**30
MB = 2**20
REAL = (14355, 11945)  # 171_470_475 px: el raster de tesis del usuario
PX_REAL = REAL[0] * REAL[1]
HARDWARE_REAL = {
    "ram_total": 64 * GB,
    "ram_disponible": 40 * GB,
    "cpu_fisicos": 8,
    "cpu_logicos": 16,
    "disco_libre": 125 * GB,
}


def test_phase1_peak_matches_the_three_measurements():
    # picos medidos: 320 MB (4 Mpx), 2640 MB (36 Mpx), 12459 MB (171,5 Mpx)
    for shape, medido_mb in (
        ((2000, 2000), 320.2),
        ((6000, 6000), 2639.5),
        (REAL, 12458.6),
    ):
        estimado_mb = res.phase_peak_bytes(1, shape) / MB
        assert estimado_mb == pytest.approx(medido_mb, rel=0.02), shape


def test_phase2_peak_matches_the_measurements():
    # la Fase 2 solo carga el traceback int16 (2 B/px)
    for shape, medido_mb in (((6000, 6000), 99.8), (REAL, 357.9)):
        estimado_mb = res.phase_peak_bytes(2, shape) / MB
        assert estimado_mb == pytest.approx(medido_mb, rel=0.08), shape


def test_real_raster_peak_is_about_12_gb_per_worker():
    # el numero que explica el desborde: el estimador viejo (raster x 4.5)
    # predecia 6,2 GB y el pico real es el doble
    pico_gb = res.phase_peak_bytes(1, REAL) / GB
    assert pico_gb == pytest.approx(12.17, abs=0.1)


def test_unknown_phase_raises():
    with pytest.raises(ValueError, match="Fase desconocida"):
        res.phase_peak_bytes(3, REAL)


def test_disk_budget_is_traceback_per_origin():
    # 60 origenes x 171_470_475 px x 2 B = 19,2 GB
    assert res.disk_bytes_for_phase1(REAL, 60) == 60 * PX_REAL * 2
    assert res.disk_bytes_for_phase1(REAL, 60) / GB == pytest.approx(19.2, abs=0.1)


def test_suggest_never_returns_zero():
    # ni con 1 GB disponible y un raster de 171 Mpx: 1 worker y aviso
    assert res.suggest_n_jobs(1, REAL, 60, 1 * GB, cpu_count=8) == 1


def test_suggest_is_capped_by_ram():
    # 40 GB disponibles x 0,8 = 32 GB; 32 / 12,17 = 2,6 -> 2
    assert res.suggest_n_jobs(1, REAL, 60, 40 * GB, cpu_count=8) == 2
    # 54 GB (la RAM libre del intento que fallo): 0,8 x 54 / 12,17 = 3,5 -> 3
    assert res.suggest_n_jobs(1, REAL, 60, 54 * GB, cpu_count=8) == 3


def test_suggest_is_capped_by_available_tasks():
    assert res.suggest_n_jobs(1, REAL, 1, 128 * GB, cpu_count=8) == 1
    assert res.suggest_n_jobs(1, REAL, 3, 128 * GB, cpu_count=8) == 3


def test_suggest_is_capped_by_cpu_count():
    # raster chico: la RAM no limita, manda el tope de CPU inyectado
    assert res.suggest_n_jobs(1, (1000, 1000), 60, 128 * GB, cpu_count=3) == 3


def test_phase2_allows_more_workers_than_phase1():
    # mismo hardware: la Fase 2 es ~38x mas liviana por worker
    n1 = res.suggest_n_jobs(1, REAL, 60, 40 * GB, cpu_count=8)
    n2 = res.suggest_n_jobs(2, REAL, 60, 40 * GB, cpu_count=8)
    assert n2 > n1
    assert n1 == 2


def test_safety_fraction_shrinks_the_suggestion():
    estricto = res.suggest_n_jobs(
        1, (4000, 4000), 60, 40 * GB, cpu_count=64, safety=0.9
    )
    conservador = res.suggest_n_jobs(
        1, (4000, 4000), 60, 40 * GB, cpu_count=64, safety=0.4
    )
    assert conservador < estricto


def test_workers_that_fit():
    assert res.workers_that_fit(40 * GB, res.phase_peak_bytes(1, REAL)) == 2
    assert res.workers_that_fit(0, res.phase_peak_bytes(1, REAL)) == 0
    assert res.workers_that_fit(40 * GB, 0) == 0


def test_advice_warns_when_override_exceeds_the_safe_limit():
    d = res.advice_for_phase(1, REAL, 60, hardware=HARDWARE_REAL, override=6)
    assert d["sugerido"] == 2
    assert d["aplicado"] == 2
    assert d["aviso"] is not None
    # el aviso nombra el valor pedido, el seguro y los GB reales
    assert "MAX_WORKERS_PHASE1=6" in d["aviso"]
    assert "se usaran 2" in d["aviso"]
    assert "73.0 GB" in d["aviso"]  # 6 x 12,17 GB
    assert any("AVISO" in linea for linea in d["lineas"])


def test_advice_without_override_reports_the_suggestion():
    d = res.advice_for_phase(1, REAL, 60, hardware=HARDWARE_REAL)
    assert d["sugerido"] == 2
    assert d["aplicado"] == 2
    assert d["aviso"] is None


def test_advice_with_safe_override_keeps_it_silent():
    d = res.advice_for_phase(1, REAL, 60, hardware=HARDWARE_REAL, override=1)
    assert d["aplicado"] == 1
    assert d["aviso"] is None
    assert any("MAX_WORKERS_PHASE1=1" in linea for linea in d["lineas"])


def test_format_hardware_reports_the_three_axes():
    texto = res.format_hardware(HARDWARE_REAL)
    assert "RAM total 64.0 GB" in texto
    assert "disponible 40.0 GB" in texto
    assert "8 fisicos / 16 logicos" in texto
    assert "disco libre 125.0 GB" in texto


def test_detect_hardware_reads_the_machine():
    hw = res.detect_hardware()
    assert hw["ram_total"] > 0
    assert hw["ram_disponible"] > 0
    assert hw["cpu_fisicos"] >= 1
    assert hw["cpu_logicos"] >= hw["cpu_fisicos"]
