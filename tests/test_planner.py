import pytest

import lcp.planner as planner

# Capa de una sola capa con ids 1..N (los ids del ráster de ejemplo son 1..14).
ORIGINS_4 = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0), 4: (30.0, 0.0)}


def _layer(n, offset=0.0):
    return {i: (offset + i, 0.0) for i in range(1, n + 1)}


def test_all_pairs_four_points():
    plan = planner.build_pair_plan("all_pairs", ORIGINS_4)
    assert plan["mode"] == "all_pairs"
    assert plan["inverted"] is False
    assert plan["n_expected"] == 6
    # Orden determinista de ids y sin raíces sin destinos.
    assert list(plan["roots"]) == [1, 2, 3]
    assert list(plan["roots"][1]) == [2, 3, 4]
    assert list(plan["roots"][2]) == [3, 4]
    assert list(plan["roots"][3]) == [4]
    # Las coordenadas del destino se conservan tal cual.
    assert plan["roots"][2][4] == ORIGINS_4[4]


def test_all_pairs_fourteen_points_regression():
    # Misma forma que la regresión legacy: 14 puntos == 91 rutas.
    origins = _layer(14)
    plan = planner.build_pair_plan("all_pairs", origins)
    assert plan["n_expected"] == 91
    assert sum(len(t) for t in plan["roots"].values()) == 91
    assert list(plan["roots"]) == list(range(1, 14))


def test_single_origin_five_points_includes_lower_ids():
    plan = planner.build_pair_plan("single_origin", _layer(5), origin_id=3)
    assert plan["mode"] == "single_origin"
    assert plan["inverted"] is False
    assert plan["n_expected"] == 4
    assert list(plan["roots"]) == [3]
    assert list(plan["roots"][3]) == [1, 2, 4, 5]


def test_cross_layers_three_origins_two_dests():
    origins = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (200.0, 0.0)}
    dests = {7: (0.0, 500.0), 8: (100.0, 500.0)}
    plan = planner.build_pair_plan("cross_layers", origins, dests)
    assert plan["mode"] == "cross_layers"
    assert plan["inverted"] is False
    assert plan["n_expected"] == 6
    assert list(plan["roots"]) == [1, 2, 3]
    for oid in (1, 2, 3):
        assert list(plan["roots"][oid]) == [7, 8]
        assert plan["roots"][oid][7] == dests[7]
        assert plan["roots"][oid][8] == dests[8]


def test_cross_layers_inversion_auto():
    origins = _layer(5, offset=1000.0)
    dests = {7: (0.0, 500.0), 8: (100.0, 500.0)}
    plan = planner.build_pair_plan("cross_layers", origins, dests, invert="auto")
    assert plan["inverted"] is True
    assert plan["n_expected"] == 10
    # Invertido: raíces = ids de destino, cada una apunta a TODOS los orígenes.
    assert list(plan["roots"]) == [7, 8]
    for did in (7, 8):
        assert list(plan["roots"][did]) == [1, 2, 3, 4, 5]
        assert plan["roots"][did][3] == origins[3]


def test_cross_layers_no_inversion_when_dests_outnumber_origins():
    origins = {1: (0.0, 0.0), 2: (10.0, 0.0)}
    dests = _layer(5, offset=500.0)
    plan = planner.build_pair_plan("cross_layers", origins, dests, invert="auto")
    assert plan["inverted"] is False
    assert plan["n_expected"] == 10
    assert list(plan["roots"]) == [1, 2]
    for oid in (1, 2):
        assert list(plan["roots"][oid]) == [1, 2, 3, 4, 5]


def test_invert_never_and_always_are_honoured():
    origins = _layer(5, offset=1000.0)
    dests = {7: (0.0, 500.0), 8: (100.0, 500.0)}

    never = planner.build_pair_plan("cross_layers", origins, dests, invert="never")
    assert never["inverted"] is False
    assert list(never["roots"]) == [1, 2, 3, 4, 5]

    always = planner.build_pair_plan("cross_layers", origins, dests, invert="always")
    assert always["inverted"] is True
    assert list(always["roots"]) == [7, 8]
    assert always["n_expected"] == 10


def test_legacy_modes_never_invert():
    origins = _layer(4)
    for invert in ("auto", "always"):
        all_pairs = planner.build_pair_plan("all_pairs", origins, invert=invert)
        assert all_pairs["inverted"] is False
        single = planner.build_pair_plan(
            "single_origin", origins, origin_id=2, invert=invert
        )
        assert single["inverted"] is False


def test_invalid_mode_lists_available_modes():
    with pytest.raises(ValueError) as exc:
        planner.build_pair_plan("todo_a_todos", ORIGINS_4)
    message = str(exc.value)
    assert "todo_a_todos" in message
    for mode in planner.MODES:
        assert mode in message


def test_invalid_invert_policy_lists_available_policies():
    with pytest.raises(ValueError) as exc:
        planner.build_pair_plan(
            "cross_layers", _layer(3), {7: (0.0, 0.0)}, invert="a_veces"
        )
    message = str(exc.value)
    assert "a_veces" in message
    for policy in planner.INVERT_POLICIES:
        assert policy in message


def test_single_origin_requires_existing_origin_id():
    with pytest.raises(ValueError) as exc:
        planner.build_pair_plan("single_origin", _layer(5))
    assert "[1, 2, 3, 4, 5]" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        planner.build_pair_plan("single_origin", _layer(5), origin_id=9)
    message = str(exc.value)
    assert "9" in message
    assert "[1, 2, 3, 4, 5]" in message


def test_empty_layers_raise():
    with pytest.raises(ValueError, match="orígenes"):
        planner.build_pair_plan("all_pairs", {})
    with pytest.raises(ValueError, match="destinos"):
        planner.build_pair_plan("cross_layers", _layer(3), {})
    # cross_layers sin segunda capa no puede planificar pares sitio-sitio.
    with pytest.raises(ValueError, match="destinos"):
        planner.build_pair_plan("cross_layers", _layer(3))


def test_identical_pixel_pairs_detects_only_shared_pixels():
    origin_pixels = {10: (5, 5), 11: (0, 0), 12: (7, 3)}
    dest_pixels = {20: (5, 5), 21: (1, 1), 22: (7, 3)}
    assert planner.identical_pixel_pairs(origin_pixels, dest_pixels) == [
        (10, 20),
        (12, 22),
    ]


def test_identical_pixel_pairs_is_order_stable():
    origin_pixels = {30: (4, 4), 10: (4, 4), 50: (9, 9)}
    dest_pixels = {7: (9, 9), 3: (4, 4), 5: (0, 1)}
    expected = [(10, 3), (30, 3), (50, 7)]
    assert planner.identical_pixel_pairs(origin_pixels, dest_pixels) == expected
    # El orden de inserción de los diccionarios no cambia el resultado.
    origin_rev = dict(reversed(list(origin_pixels.items())))
    dest_rev = dict(reversed(list(dest_pixels.items())))
    assert planner.identical_pixel_pairs(origin_rev, dest_rev) == expected


def test_identical_pixel_pairs_empty_when_all_distinct():
    assert planner.identical_pixel_pairs({1: (0, 0)}, {2: (0, 1)}) == []
