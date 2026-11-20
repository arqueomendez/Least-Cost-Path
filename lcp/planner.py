# lcp/planner.py

"""
Planificador puro de pares origen-destino por modo de analisis.

No hace E/S: recibe diccionarios `{id: (x, y)}` de cada capa y devuelve, por
raiz MCP, el diccionario de destinos `{dest_id: (x, y)}`. Los modos de una sola
capa trabajan siempre sobre `origins`; `dests` solo tiene sentido en
`cross_layers`.
"""

MODES = ("all_pairs", "single_origin", "cross_layers", "od_pair")
INVERT_POLICIES = ("auto", "never", "always")


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(
            f"Modo de análisis no válido: {mode!r}. "
            f"Modos disponibles: {', '.join(MODES)}"
        )


def _check_invert(invert):
    if invert not in INVERT_POLICIES:
        raise ValueError(
            f"Política de inversión no válida: {invert!r}. "
            f"Políticas disponibles: {', '.join(INVERT_POLICIES)}"
        )


def _check_layers(origins, dests):
    if not origins:
        raise ValueError(
            "La capa de orígenes está vacía: no hay puntos para planificar rutas"
        )
    if dests is not None and not dests:
        raise ValueError(
            "La capa de destinos está vacía: no hay puntos para planificar rutas"
        )


def _single_layer_plan(mode, origins, origin_id):
    """
    Builds roots and n_expected for the single-layer modes (all_pairs, single_origin).
    """
    ids = sorted(origins)

    if mode == "all_pairs":
        roots = {}
        for oid in ids:
            # Dedup legacy: solo los ids estrictamente mayores evitan duplicar pares.
            targets = {did: tuple(origins[did]) for did in ids if did > oid}
            if targets:
                roots[oid] = targets
        return roots, len(ids) * (len(ids) - 1) // 2

    if origin_id is None:
        raise ValueError(
            f"El modo 'single_origin' requiere origin_id. IDs disponibles: {ids}"
        )
    if origin_id not in origins:
        raise ValueError(
            f"origin_id {origin_id!r} no existe en la capa de orígenes. "
            f"IDs disponibles: {ids}"
        )

    targets = {tid: tuple(origins[tid]) for tid in ids if tid != origin_id}
    roots = {origin_id: targets} if targets else {}
    return roots, len(ids) - 1


def _cross_layer_plan(origins, dests, invert):
    """
    Builds roots, inverted flag and n_expected for the two-layer mode.
    """
    origin_ids = sorted(origins)
    dest_ids = sorted(dests)

    if invert == "auto":
        inverted = len(dest_ids) < len(origin_ids)
    else:
        inverted = invert == "always"

    if inverted:
        # Una corrida MCP por fuente (raiz) hacia todos los sitios.
        roots = {
            did: {oid: tuple(origins[oid]) for oid in origin_ids} for did in dest_ids
        }
    else:
        roots = {
            oid: {did: tuple(dests[did]) for did in dest_ids} for oid in origin_ids
        }

    # M*N pares en ambos sentidos: la inversion no cambia el total esperado.
    return roots, inverted, len(origin_ids) * len(dest_ids)


def _od_pair_plan(origins, origin_id, dest_id):
    """
    Builds the single origin->dest plan for the od_pair mode.

    Sin dedup ni filtros: exactamente el par pedido, incluso con
    origin_id > dest_id (a diferencia del triangulo superior de all_pairs).
    """
    ids = sorted(origins)
    if origin_id is None:
        raise ValueError(
            "El modo 'od_pair' requiere origin_id (punto de inicio). "
            f"IDs disponibles: {ids}"
        )
    if dest_id is None:
        raise ValueError(
            "El modo 'od_pair' requiere dest_id (punto de destino). "
            f"IDs disponibles: {ids}"
        )
    if origin_id not in origins:
        raise ValueError(
            f"origin_id {origin_id!r} no existe en la capa de puntos. "
            f"IDs disponibles: {ids}"
        )
    if dest_id not in origins:
        raise ValueError(
            f"dest_id {dest_id!r} no existe en la capa de puntos. "
            f"IDs disponibles: {ids}"
        )
    if origin_id == dest_id:
        raise ValueError(
            f"El origen y el destino son el mismo punto (id {origin_id!r}): "
            "no hay ruta que calcular. Elegi dos puntos distintos (podes ver "
            "los candidatos con el preview de la Celda 4)."
        )
    roots = {origin_id: {dest_id: tuple(origins[dest_id])}}
    return roots, 1


def build_pair_plan(
    mode, origins, dests=None, origin_id=None, invert="never", dest_id=None
):
    """
    Builds the per-root plan for an analysis mode.

    `origins` (y `dests` en `cross_layers`) son `{id: (x, y)}` con ids enteros.
    Devuelve `{"mode", "inverted", "roots": {root_id: {target_id: (x, y)}},
    "n_expected"}` con los ids ordenados ascendentemente para que el plan sea
    reproducible.
    """
    _check_mode(mode)
    _check_invert(invert)
    _check_layers(origins, dests)

    if mode == "cross_layers":
        if dests is None:
            raise ValueError(
                "El modo 'cross_layers' requiere la capa de destinos "
                "(DEST_POINTS_PATH): no se planifican pares dentro de una misma capa"
            )
        roots, inverted, n_expected = _cross_layer_plan(origins, dests, invert)
    elif mode == "od_pair":
        # Una sola ruta dentro de la misma capa; `dests` no aplica.
        roots, n_expected = _od_pair_plan(origins, origin_id, dest_id)
        inverted = False
    else:
        # Los modos legacy no tienen segunda capa ni direccion invertible.
        roots, n_expected = _single_layer_plan(mode, origins, origin_id)
        inverted = False

    return {
        "mode": mode,
        "inverted": inverted,
        "roots": roots,
        "n_expected": n_expected,
    }


def identical_pixel_pairs(origin_pixels, dest_pixels):
    """
    Returns the sorted (origin_id, dest_id) pairs sharing the same pixel.
    """
    by_pixel = {}
    for oid, pixel in origin_pixels.items():
        by_pixel.setdefault(tuple(pixel), []).append(oid)

    pairs = []
    for did, pixel in dest_pixels.items():
        for oid in by_pixel.get(tuple(pixel), ()):
            pairs.append((oid, did))

    return sorted(pairs)


KERNELS = ("mcp_geometric", "corridor_astar")


def check_kernel(mode, kernel):
    """
    Valida la seleccion de kernel de calculo (notebook, Celda 3/4).

    'mcp_geometric' es el default en TODOS los modos (comportamiento
    historico intacto). 'corridor_astar' es el kernel dirigido 1->1 y
    solo esta habilitado para el modo 'od_pair': en los otros modos una
    sola corrida MCP por raiz sirve a TODOS los destinos, y el A* de
    corredor (una busqueda por par) no tiene ventaja, solo coste.
    """
    if kernel not in KERNELS:
        raise ValueError(
            f"KERNEL no valido: {kernel!r}. Kernels disponibles: "
            f"{', '.join(KERNELS)}"
        )
    if kernel == "corridor_astar" and mode != "od_pair":
        raise ValueError(
            "KERNEL='corridor_astar' solo esta habilitado para el modo "
            f"'od_pair' (modo actual: {mode!r}). El kernel A* jerarquico "
            "resuelve UN par por busqueda; para varios destinos por origen "
            "use KERNEL='mcp_geometric'."
        )
    return kernel
