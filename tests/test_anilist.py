"""
Tests con red mockeada (respx) para anilist_buscar_titulo() -- busqueda de
anime por texto en AniList, pensada como fuente alternativa a Jikan.

Mismo patron que test_network_jikan.py / test_retry_flow.py.
"""
import json

import httpx
import pytest
import respx

from chapterizen import anilist as cz

ANILIST_GRAPHQL = "https://graphql.anilist.co"


def _media(anilist_id, romaji, english=None, native=None, synonyms=None, episodes=None):
    return {
        "id": anilist_id,
        "idMal": anilist_id + 100000,
        "title": {
            "romaji": romaji,
            "english": english,
            "native": native,
            "userPreferred": romaji,
        },
        "synonyms": synonyms or [],
        "format": "TV",
        "status": "FINISHED",
        "episodes": episodes,
        "startDate": {"year": 2024, "month": 1, "day": 1},
    }


def _mock_search(media_list):
    respx.post(ANILIST_GRAPHQL).mock(
        return_value=httpx.Response(
            200, json={"data": {"Page": {"media": media_list}}}
        )
    )


@respx.mock
def test_resultado_exacto_unico_es_confiable():
    _mock_search([_media(100, "Sousou no Frieren", english="Frieren: Beyond Journey's End")])

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("Sousou no Frieren")

    assert canon == "Sousou no Frieren"
    assert item["id"] == 100
    assert confiable is True
    assert ts1 == 1.0


@respx.mock
def test_resultado_ambiguo_con_ganador_claro_es_confiable():
    _mock_search([
        _media(200, "Chained Soldier"),
        _media(201, "Chainsaw Man"),
    ])

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("Chained Soldier")

    assert canon == "Chained Soldier"
    assert item["id"] == 200
    assert confiable is True


@respx.mock
def test_resultado_ambiguo_sin_ganador_claro_no_es_confiable():
    # Dos candidatos con nombres parecidos entre si y distintos a la consulta
    # -- ningun candidato deberia despegarse claramente.
    _mock_search([
        _media(300, "Some Unrelated Anime Title"),
        _media(301, "Another Unrelated Anime Title"),
    ])

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("Completely Different Query")

    assert confiable is False


@respx.mock
def test_sin_resultados_devuelve_consulta_original():
    _mock_search([])

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("Anime Que No Existe")

    assert canon == "Anime Que No Existe"
    assert item is None
    assert confiable is False
    assert ts1 == 0.0


def test_consulta_vacia_no_hace_red():
    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("")
    assert canon == ""
    assert item is None
    assert confiable is False
    assert ts1 == 0.0


@respx.mock
def test_503_seguido_de_200_se_recupera_via_reintento():
    route = respx.post(ANILIST_GRAPHQL).mock(
        side_effect=[
            httpx.Response(503, json={"error": "temporarily unavailable"}),
            httpx.Response(
                200,
                json={"data": {"Page": {"media": [_media(400, "Recovered Anime")]}}},
            ),
        ]
    )

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("Recovered Anime")

    assert canon == "Recovered Anime"
    assert item["id"] == 400
    assert route.call_count == 2


@respx.mock
def test_titulo_principal_usa_romaji_antes_que_english():
    _mock_search([_media(500, "Kimetsu no Yaiba", english="Demon Slayer")])

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("Kimetsu no Yaiba")

    assert canon == "Kimetsu no Yaiba"


@respx.mock
def test_titulo_principal_cae_a_english_si_no_hay_romaji():
    media = _media(600, None, english="English Only Title")
    _mock_search([media])

    canon, item, confiable, ts1 = cz.anilist_buscar_titulo("English Only Title")

    assert canon == "English Only Title"


def test_anilist_titulos_desde_item_deduplica_y_preserva_orden():
    item = _media(700, "Romaji Title", english="Romaji Title", synonyms=["Alt Name"])
    titulos = cz.anilist_titulos_desde_item(item)
    assert titulos == ["Romaji Title", "Alt Name"]


# ── anilist_avanzar_a_secuela / anilist_resolver_temporada_por_sequel /  ──
# ── anilist_navegar_por_episodio                                        ──
#
# Mismo endpoint (POST a ANILIST_GRAPHQL) que anilist_buscar_titulo, pero
# con la query de relations -- el side_effect enruta por el "id" de las
# variables del body (no por el contenido de la query en si), analogo a
# _mock_trace_por_marcador en test_network_trace.py.

def _edge(relation_type: str, node: dict) -> dict:
    return {"relationType": relation_type, "node": node}


def _mock_relations(por_id: dict):
    """por_id: {anilist_id: [edge, ...]}. IDs no listados devuelven relations vacio."""
    def _side_effect(request: httpx.Request) -> httpx.Response:
        body  = json.loads(request.content)
        vid   = (body.get("variables") or {}).get("id")
        edges = por_id.get(vid, [])
        return httpx.Response(200, json={"data": {"Media": {"relations": {"edges": edges}}}})
    respx.post(ANILIST_GRAPHQL).mock(side_effect=_side_effect)


@respx.mock
def test_anilist_avanzar_a_secuela_ignora_spin_off_y_side_story_toma_solo_sequel():
    """relationType ya distingue SEQUEL de SPIN_OFF/SIDE_STORY (confirmado
    contra datos reales de AniList) -- el primer SEQUEL debe ganar aunque
    aparezcan otras relaciones antes en la lista."""
    _mock_relations({
        100: [
            _edge("SPIN_OFF", _media(150, "Attack on Titan: Junior High", episodes=12)),
            _edge("SIDE_STORY", _media(160, "Attack on Titan OVA", episodes=3)),
            _edge("SEQUEL", _media(200, "Attack on Titan Season 2", episodes=12)),
        ],
    })

    secuela = cz.anilist_avanzar_a_secuela(_media(100, "Attack on Titan", episodes=25))

    assert secuela["id"] == 200
    assert secuela["title"]["romaji"] == "Attack on Titan Season 2"


@respx.mock
def test_anilist_resolver_temporada_por_sequel_cadena_completa():
    _mock_relations({
        100: [_edge("SEQUEL", _media(200, "Attack on Titan Season 2", episodes=12))],
        200: [_edge("SEQUEL", _media(300, "Attack on Titan Season 3", episodes=22))],
    })
    base = _media(100, "Attack on Titan", episodes=25)

    resultado = cz.anilist_resolver_temporada_por_sequel(base, 3)

    assert resultado["id"] == 300
    assert resultado["title"]["romaji"] == "Attack on Titan Season 3"


@respx.mock
def test_anilist_resolver_temporada_por_sequel_cadena_cortada_lanza_runtime_error():
    _mock_relations({100: []})  # sin relaciones -- cadena se corta de inmediato
    base = _media(100, "Attack on Titan", episodes=25)

    with pytest.raises(RuntimeError, match="sin secuela"):
        cz.anilist_resolver_temporada_por_sequel(base, 2)


@respx.mock
def test_anilist_resolver_temporada_por_sequel_no_valida_el_titulo_resultante():
    """anilist_resolver_temporada_por_sequel es puramente mecanica, igual
    que jikan_resolver_temporada_por_sequel -- no aplica
    _aceptar_canon_sin_perder_tokens internamente. Esa decision es
    responsabilidad del caller (gui/workers.py), en el mismo punto donde
    ya se aplica para Jikan, para que el mensaje de rechazo sea el mismo
    sin importar la fuente (ver test_resolver_worker_integration.py para
    la cobertura de ese gate a nivel de ResolverWorker)."""
    _mock_relations({
        100: [_edge("SEQUEL", _media(999, "Completely Different Show", episodes=12))],
    })
    base = _media(100, "Attack on Titan", episodes=25)

    resultado = cz.anilist_resolver_temporada_por_sequel(base, 2)

    assert resultado["id"] == 999
    assert resultado["title"]["romaji"] == "Completely Different Show"


@respx.mock
def test_anilist_navegar_por_episodio_dentro_de_temporada_actual_no_navega():
    _mock_relations({})  # no deberia necesitarse ninguna consulta de relations
    base = _media(100, "Attack on Titan", episodes=25)

    resultado, ep_relativo, temporada = cz.anilist_navegar_por_episodio(base, 10)

    assert resultado is base
    assert ep_relativo == 10
    assert temporada == 1


@respx.mock
def test_anilist_navegar_por_episodio_avanza_dos_temporadas():
    _mock_relations({
        100: [_edge("SEQUEL", _media(200, "Attack on Titan Season 2", episodes=12))],
        200: [_edge("SEQUEL", _media(300, "Attack on Titan Season 3", episodes=22))],
    })
    base = _media(100, "Attack on Titan", episodes=25)  # S1: 25 eps, S2: 12 eps

    # Episodio absoluto 42 = 25 (S1) + 12 (S2) + 5 -> cae en S3, relativo 5
    resultado, ep_relativo, temporada = cz.anilist_navegar_por_episodio(base, 42)

    assert resultado["id"] == 300
    assert ep_relativo == 5
    assert temporada == 3
