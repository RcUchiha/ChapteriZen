"""
Tests con red mockeada (respx) para jikan_navegar_por_episodio(), que
navega la cadena de secuelas de Jikan por conteo de episodios.
"""
import httpx
import pytest
import respx

import chapterizen as cz


@respx.mock
def test_cadena_completa_resuelve_temporada_y_episodio_relativo():
    """S1 tiene 12 episodios; se pide el episodio absoluto 15 -> debe
    saltar a temporada 2, episodio relativo 3."""
    respx.get("https://api.jikan.moe/v4/anime/1/relations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"relation": "Sequel", "entry": [{"mal_id": 2}]}]}
        )
    )
    respx.get("https://api.jikan.moe/v4/anime/2").mock(
        return_value=httpx.Response(
            200, json={"data": {"mal_id": 2, "title": "Anime Season 2", "episodes": 13}}
        )
    )

    base_entry = {"mal_id": 1, "title": "Anime Season 1", "episodes": 12}
    entry, ep_relativo, temporada = cz.jikan_navegar_por_episodio(base_entry, 15)

    assert temporada == 2
    assert ep_relativo == 3
    assert entry["title"] == "Anime Season 2"


@respx.mock
def test_episodio_dentro_de_la_primera_temporada_no_navega():
    """Si el episodio pedido cabe en S1, no debe hacer ninguna llamada de red."""
    base_entry = {"mal_id": 1, "title": "Anime Season 1", "episodes": 12}
    entry, ep_relativo, temporada = cz.jikan_navegar_por_episodio(base_entry, 8)

    assert temporada == 1
    assert ep_relativo == 8
    assert entry is base_entry


@respx.mock
def test_cadena_incompleta_lanza_runtime_error():
    """S1=12 episodios, S2=5 episodios, se pide episodio absoluto 20
    (20-12=8 > 5, necesita temporada 3) pero Jikan no tiene esa secuela
    -> debe lanzar RuntimeError."""
    respx.get("https://api.jikan.moe/v4/anime/10/relations").mock(
        return_value=httpx.Response(
            200, json={"data": [{"relation": "Sequel", "entry": [{"mal_id": 11}]}]}
        )
    )
    respx.get("https://api.jikan.moe/v4/anime/11").mock(
        return_value=httpx.Response(
            200, json={"data": {"mal_id": 11, "title": "Anime S2", "episodes": 5}}
        )
    )
    respx.get("https://api.jikan.moe/v4/anime/11/relations").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    base_entry = {"mal_id": 10, "title": "Anime S1", "episodes": 12}
    with pytest.raises(RuntimeError):
        cz.jikan_navegar_por_episodio(base_entry, 20)


@respx.mock
def test_entry_sin_conteo_de_episodios_lanza_runtime_error():
    base_entry = {"mal_id": 99, "title": "Anime sin conteo", "episodes": None}
    with pytest.raises(RuntimeError):
        cz.jikan_navegar_por_episodio(base_entry, 5)
