import pytest

import lcp.data_loader as dl
import lcp.planner as planner

# Capa de una sola capa con ids 1..N (los ids del ráster de ejemplo son 1..14).
ORIGINS_4 = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0), 4: (30.0, 0.0)}


# ---------------------------------------------------------------------------
# Planner: modo od_pair (S5)
# ---------------------------------------------------------------------------


def test_od_pair_plan_single_route():
    plan = planner.build_pair_plan("od_pair", ORIGINS_4, origin_id=2, dest_id=4)
    assert plan["mode"] == "od_pair"
    assert plan["inverted"] is False
    assert plan["n_expected"] == 1
    assert list(plan["roots"]) == [2]
    assert list(plan["roots"][2]) == [4]
    assert plan["roots"][2][4] == ORIGINS_4[4]


def test_od_pair_plan_accepts_origin_id_greater_than_dest_id():
    # Sin dedup `did > origin_id`: el par pedido se calcula tal cual (S6).
    plan = planner.build_pair_plan("od_pair", ORIGINS_4, origin_id=4, dest_id=1)
    assert plan["n_expected"] == 1
    assert list(plan["roots"]) == [4]
    assert list(plan["roots"][4]) == [1]


def test_od_pair_missing_origin_id_lists_available_ids():
    with pytest.raises(ValueError, match="IDs disponibles"):
        planner.build_pair_plan("od_pair", ORIGINS_4, origin_id=99, dest_id=2)


def test_od_pair_missing_dest_id_lists_available_ids():
    with pytest.raises(ValueError, match="IDs disponibles"):
        planner.build_pair_plan("od_pair", ORIGINS_4, origin_id=1, dest_id=99)


def test_od_pair_origin_equals_dest_rejected():
    with pytest.raises(ValueError):
        planner.build_pair_plan("od_pair", ORIGINS_4, origin_id=3, dest_id=3)


def test_od_pair_requires_both_ids():
    with pytest.raises(ValueError):
        planner.build_pair_plan("od_pair", ORIGINS_4, origin_id=1)
    with pytest.raises(ValueError):
        planner.build_pair_plan("od_pair", ORIGINS_4, dest_id=1)


def test_od_pair_mode_registered():
    assert "od_pair" in planner.MODES


# ---------------------------------------------------------------------------
# Selector estricto de puntos (S2, S3)
# ---------------------------------------------------------------------------

CANDIDATES = [
    {"id": 1, "fid": 0, "coords": (10.0, 20.0), "properties": {"id": 1, "nombre": "a"}},
    {"id": 2, "fid": 1, "coords": (30.0, 40.0), "properties": {"id": 2, "nombre": "b"}},
    {
        "id": 7001,
        "fid": 2,
        "coords": (50.0, 60.0),
        "properties": {"id": 7, "nombre": "c"},
    },
]


def test_select_point_by_id_field():
    point = dl.select_point(CANDIDATES, "id", 2)
    assert point["id"] == 2
    assert point["fid"] == 1


def test_select_point_by_other_field_string_value():
    point = dl.select_point(CANDIDATES, "nombre", "c")
    assert point["id"] == 7001


def test_select_point_value_tolerant_to_type():
    # El valor puede llegar como texto ("2") aunque el atributo sea entero.
    assert dl.select_point(CANDIDATES, "id", "2")["id"] == 2
    assert dl.select_point(CANDIDATES, "id", 2.0)["id"] == 2


def test_select_point_unknown_field_raises_with_available_fields():
    with pytest.raises(ValueError, match="nombre_no_existe"):
        dl.select_point(CANDIDATES, "nombre_no_existe", 1)
    # El mensaje lista los campos disponibles.
    with pytest.raises(ValueError, match="'id'"):
        dl.select_point(CANDIDATES, "nombre_no_existe", 1)


def test_select_point_unknown_value_raises_with_available_values():
    with pytest.raises(ValueError, match="valor_no_existe"):
        dl.select_point(CANDIDATES, "id", "valor_no_existe")


def test_select_point_never_falls_back_to_fid():
    # El fid nunca se usa como campo implicito: solo si se pide explicitamente.
    with pytest.raises(ValueError):
        dl.select_point(CANDIDATES, "fid_no_existe", 0)


def test_select_point_explicit_fid_field_allowed():
    point = dl.select_point(CANDIDATES, "fid", 1)
    assert point["id"] == 2


def test_load_point_candidates_lists_id_fid_and_properties(tmp_path):
    # Fixture shapefile minimo con dos puntos y atributo 'id'.
    fiona = pytest.importorskip("fiona")
    from shapely.geometry import Point, mapping

    schema = {"geometry": "Point", "properties": {"id": "int"}}
    path = str(tmp_path / "points.shp")
    with fiona.open(path, "w", driver="ESRI Shapefile", schema=schema) as dst:
        dst.write(
            {
                "geometry": mapping(Point(0.0, 0.0)),
                "properties": {"id": 1},
            }
        )
        dst.write(
            {
                "geometry": mapping(Point(10.0, 0.0)),
                "properties": {"id": 2},
            }
        )

    candidates = dl.load_point_candidates(path, "id")
    assert len(candidates) == 2
    by_id = {c["id"]: c for c in candidates}
    assert by_id[1]["fid"] == 0
    assert by_id[2]["fid"] == 1
    assert by_id[1]["coords"] == (0.0, 0.0)
    assert by_id[1]["properties"]["id"] == 1


def test_select_point_large_numeric_ids_do_not_collide():
    # R3-1: la normalizacion numerica debe conservar TODOS los digitos
    # significativos; ids de 7+ digitos no colisionan entre si.
    large = [
        {"id": 1234567, "fid": 0, "coords": (1.0, 1.0), "properties": {"id": 1234567}},
        {"id": 1234568, "fid": 1, "coords": (2.0, 2.0), "properties": {"id": 1234568}},
    ]
    assert dl.select_point(large, "id", 1234567)["id"] == 1234567
    assert dl.select_point(large, "id", 1234568)["id"] == 1234568
    # El valor pedido que NO existe sigue abortando (sin vecino cercano falso).
    with pytest.raises(ValueError):
        dl.select_point(large, "id", 1234569)


def test_select_point_int_and_string_forms_equivalent():
    # REG-1: int, float y str del mismo valor logico comparten clave de
    # normalizacion, incluidos ids de 7+ digitos (una sola rama numerica).
    large = [
        {
            "id": 1234567,
            "fid": 0,
            "coords": (1.0, 1.0),
            "properties": {"id": "1234567"},
        },
        {
            "id": 1234568,
            "fid": 1,
            "coords": (2.0, 2.0),
            "properties": {"id": "1234568"},
        },
    ]
    assert dl.select_point(large, "id", 1234567)["id"] == 1234567
    assert dl.select_point(large, "id", "1234568")["id"] == 1234568
    assert dl.select_point(large, "id", 1234568.0)["id"] == 1234568


def test_select_point_fid_ambiguity_rejected():
    # R2-1: una feature MultiPoint explota en varios nodos con el MISMO fid;
    # la rama fid debe exigir unicidad como la rama de atributos, nunca
    # devolver el primero en silencio.
    exploded = [
        {"id": 7, "fid": 3, "coords": (1.0, 1.0), "properties": {"id": 7}},
        {"id": 7001, "fid": 3, "coords": (2.0, 2.0), "properties": {"id": 7}},
        {"id": 7002, "fid": 3, "coords": (3.0, 3.0), "properties": {"id": 7}},
        {"id": 9, "fid": 4, "coords": (4.0, 4.0), "properties": {"id": 9}},
    ]
    with pytest.raises(ValueError, match="debe ser unica"):
        dl.select_point(exploded, "fid", 3)
    # Un fid con un solo nodo sigue funcionando.
    assert dl.select_point(exploded, "fid", 4)["id"] == 9


# ---------------------------------------------------------------------------
# Follow-ups del ciclo de revision (F1): normalizacion y rama fid
# ---------------------------------------------------------------------------


def test_normalize_bool_matches_string_form():
    # R2-4/R3-3: un booleano True y el texto 'True'/'true' comparten clave.
    assert dl._normalize_value(True) == dl._normalize_value("true")
    assert dl._normalize_value(True) == dl._normalize_value("True")
    assert dl._normalize_value(False) == dl._normalize_value("false")


def test_normalize_none_is_its_own_key():
    # R4-null-string-silent-match: un atributo NULL no debe hacer match con el
    # texto literal 'None'/'none' ni con ningun otro valor.
    assert dl._normalize_value(None) != dl._normalize_value("None")
    assert dl._normalize_value(None) != dl._normalize_value("none")


def test_normalize_exact_integers_beyond_float_precision():
    # R4-float-precision-ids: los enteros se rutean sin pasar por float();
    # ids por encima de 2^53 conservan todos sus digitos y no colisionan.
    a = 10**18
    b = 10**18 + 1
    assert dl._normalize_value(a) != dl._normalize_value(b)
    # Garantia: int y su forma textual comparten clave exacta (sin float()).
    assert dl._normalize_value(a) == dl._normalize_value(str(a))
    assert dl._normalize_value(b) == dl._normalize_value(str(b))
    # Borde documentado: por encima de 2^53 la forma float NO es exacta
    # (precision inherente de float) y no se garantiza equivalencia.


def test_select_point_fid_tolerant_to_type_and_actionable_errors():
    # R2-5/R3-2/R4-fid-int-conversion: la rama fid comparte la tolerancia de
    # tipos ('7', 7, 7.0 equivalen) y errores accionables; 7.9 no se trunca.
    candidates = [
        {"id": 1, "fid": 7, "coords": (1.0, 1.0), "properties": {"id": 1}},
        {"id": 2, "fid": 8, "coords": (2.0, 2.0), "properties": {"id": 2}},
    ]
    assert dl.select_point(candidates, "fid", "7")["id"] == 1
    assert dl.select_point(candidates, "fid", 7.0)["id"] == 1
    assert dl.select_point(candidates, "fid", "7.0")["id"] == 1
    # 7.9 no es un fid valido: no se trunca silenciosamente a 7 y el error
    # es accionable (lista los fids disponibles).
    with pytest.raises(ValueError, match="fids disponibles"):
        dl.select_point(candidates, "fid", 7.9)
