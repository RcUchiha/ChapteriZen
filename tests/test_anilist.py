"""
Tests con red mockeada (respx) para anilist_buscar_titulo() -- busqueda de
anime por texto en AniList, pensada como fuente alternativa a Jikan.

Mismo patron que test_network_jikan.py / test_retry_flow.py.
"""
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
