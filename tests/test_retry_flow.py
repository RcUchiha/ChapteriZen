"""
Test de flujo real de reintento: confirma que _reintento_http (tenacity)
efectivamente reintenta ante un 503 y se recupera con la siguiente
respuesta exitosa, en vez de fallar en el primer intento.
"""
import httpx
import respx

import chapterizen as cz

JIKAN_ANIME = "https://api.jikan.moe/v4/anime"


@respx.mock
def test_503_seguido_de_200_se_recupera_via_reintento():
    route = respx.get(JIKAN_ANIME).mock(
        side_effect=[
            httpx.Response(503, json={"error": "temporarily unavailable"}),
            httpx.Response(200, json={"data": [{"mal_id": 1, "title": "Anime Recuperado"}]}),
        ]
    )

    resultado = cz.jikan_buscar_anime("Cualquier Query")

    assert resultado == [{"mal_id": 1, "title": "Anime Recuperado"}]
    assert route.call_count == 2


@respx.mock
def test_404_no_reintenta_y_propaga_el_error():
    route = respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(404, json={"error": "not found"}))

    import pytest
    with pytest.raises(httpx.HTTPStatusError):
        cz.jikan_buscar_anime("Query Inexistente")

    assert route.call_count == 1


@respx.mock
def test_tres_503_consecutivos_agota_reintentos_y_falla():
    """stop_after_attempt(3): tres 503 seguidos deben agotar los reintentos
    y terminar propagando el error (no reintenta indefinidamente)."""
    route = respx.get(JIKAN_ANIME).mock(
        return_value=httpx.Response(503, json={"error": "down"})
    )

    import pytest
    with pytest.raises(httpx.HTTPStatusError):
        cz.jikan_buscar_anime("Query Que Siempre Falla")

    assert route.call_count == 3
