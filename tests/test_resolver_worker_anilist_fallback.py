"""
Tests para el fallback Jikan -> AniList en ResolverWorker.run(): se activa
cuando jikan_resolver_titulo agota sus reintentos (503/504 persistente) O
cuando Jikan responde 200 pero sin resultados (picked_base is None) --
en ninguno de los dos casos cuando Jikan responde con confiable=False
(candidatos ambiguos, cubierto por el picker de discrepancia en vez de
AniList).
"""
import json

import httpx
import respx

from chapterizen.modelos import ParametrosTrabajo
from chapterizen.gui.resolver_worker import ResolverWorker

JIKAN_ANIME        = "https://api.jikan.moe/v4/anime"
ANILIST_GRAPHQL    = "https://graphql.anilist.co"
ANIMETHEMES_SEARCH = "https://api.animethemes.moe/search"
ANIMETHEMES_ANIME  = "https://api.animethemes.moe/anime"


def _anilist_media(anilist_id, romaji, episodes=12):
    return {
        "id": anilist_id,
        "idMal": anilist_id + 900000,
        "title": {
            "romaji": romaji,
            "english": None,
            "native": None,
            "userPreferred": romaji,
        },
        "synonyms": [],
        "format": "TV",
        "status": "FINISHED",
        "episodes": episodes,
        "startDate": {"year": 2024, "month": 1, "day": 1},
    }


def _run_resolver(tmp_path, video_name="Test Anime - 05.mkv"):
    video = tmp_path / video_name
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida="",
        crear_subcarpeta=False,
        search_override="",
    )
    worker = ResolverWorker(None, params, interactivo=False)

    logs = []
    worker.log.connect(logs.append)

    resultado = {}
    worker.resolved.connect(lambda p: resultado.update(ok=True, params=p))
    worker.failed.connect(lambda msg: resultado.update(ok=False, error=msg))

    worker.run()  # llamada directa, sincronica -- mismo patron ya usado en pruebas manuales
    return resultado, logs


@respx.mock
def test_jikan_agota_reintentos_usa_anilist_como_respaldo(tmp_path):
    respx.get(JIKAN_ANIME).mock(
        return_value=httpx.Response(503, json={"error": "temporarily unavailable"})
    )
    respx.post(ANILIST_GRAPHQL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"Page": {"media": [_anilist_media(999, "Test Anime")]}}},
        )
    )
    # Mock minimo de AnimeThemes: resultado unico no ambiguo para que la
    # resolucion de slug (que ahora corre siempre, ya no hay usar_exacto que
    # la salte) no abra picker ni dependa de un respaldo via Jikan (que en
    # este test esta caido a proposito). El atajo por ID externo (id=999,
    # idMal=900999) tambien resuelve directo con el mismo resultado.
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}))
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}}))

    resultado, logs = _run_resolver(tmp_path)

    assert resultado.get("ok") is True
    assert resultado["params"].titulo_usado == "Test Anime"
    assert any("Jikan no disponible, usando AniList como respaldo" in s for s in logs)
    # El resultado normal de AniList debe verse igual que el de Jikan siempre se vio.
    assert any("Título del anime: 'Test Anime'" in s for s in logs)


@respx.mock
def test_jikan_y_anilist_fallan_ambos_propaga_error(tmp_path):
    respx.get(JIKAN_ANIME).mock(
        return_value=httpx.Response(503, json={"error": "down"})
    )
    respx.post(ANILIST_GRAPHQL).mock(
        return_value=httpx.Response(503, json={"error": "also down"})
    )

    resultado, logs = _run_resolver(tmp_path)

    assert resultado.get("ok") is False
    assert resultado.get("error")
    # Se intento el respaldo (no se omitio AniList silenciosamente).
    assert any("Jikan no disponible, usando AniList como respaldo" in s for s in logs)


def test_titulos_alternativos_de_item_anilist_son_strings_limpios(tmp_path, monkeypatch):
    """Cuando jikan_item viene de AniList (fallback de Jikan agotado -- shape
    con 'id'/'idMal', SIN 'mal_id'), _resolver_slug_con_picker debe extraer
    los títulos alternativos con anilist_titulos_desde_item (strings sueltos:
    romaji/english/native/userPreferred), no con jikan_titulos_desde_item.
    Antes de este fix, jikan_titulos_desde_item hacía str(item['title']) --
    pero item['title'] en AniList es un dict, no un string -- así que la
    consulta enviada a AnimeThemes terminaba siendo el repr completo del
    dict, ej. "{'romaji': 'Mato Seihei no Slave', 'english': None, ...}"."""
    from chapterizen.gui import resolver_worker as workers_mod

    anilist_item = {
        "id": 12345,
        "idMal": 999999,
        "title": {
            "romaji": "Mato Seihei no Slave",
            "english": "Slave of the Magic Capital's Guardian Fairy",
            "native": None,
            "userPreferred": "Mato Seihei no Slave",
        },
        "synonyms": ["Guardian Fairy Slave"],
        "format": "TV",
        "status": "FINISHED",
        "episodes": 12,
    }

    consultas_vistas = []

    def _fake_buscar_animethemes(q):
        consultas_vistas.append(q)
        return []  # nunca hay match -- fuerza a intentar TODOS los títulos alternativos

    def _fake_jikan_buscar_anime(consulta, limite=10):
        return []  # respaldo final tambien vacio -- solo nos interesan las consultas intentadas

    monkeypatch.setattr(workers_mod, "buscar_anime_en_animethemes", _fake_buscar_animethemes)
    monkeypatch.setattr(workers_mod, "jikan_buscar_anime", _fake_jikan_buscar_anime)

    video = tmp_path / "Mato Seihei no Slave - 01.mkv"
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida="",
        crear_subcarpeta=False,
        search_override="",
    )
    worker = ResolverWorker(None, params, interactivo=False)

    try:
        worker._resolver_slug_con_picker("Mato Seihei no Slave", 1, jikan_item=anilist_item)
    except RuntimeError:
        pass  # esperado -- no hay resultados en ningun lado, lo que nos interesa es que se intento

    assert consultas_vistas  # se intentaron consultas
    assert not any(q.startswith("{") for q in consultas_vistas), (
        f"consulta corrupta (repr de dict) enviada a AnimeThemes: {consultas_vistas!r}"
    )
    assert "Mato Seihei no Slave" in consultas_vistas
    assert "Slave of the Magic Capital's Guardian Fairy" in consultas_vistas
    assert "Guardian Fairy Slave" in consultas_vistas


@respx.mock
def test_jikan_confiable_false_no_dispara_fallback_de_anilist(tmp_path):
    """Jikan responde (sin agotar reintentos) pero con multiples candidatos
    ambiguos -- confiable=False. Este caso NO debe activar AniList; usa el
    mecanismo existente (picker de discrepancia), que en modo no interactivo
    simplemente deja confiable=False sin tocar AniList para nada."""
    respx.get(JIKAN_ANIME).mock(
        return_value=httpx.Response(
            200,
            json={"data": [
                {"mal_id": 1, "title": "Some Other Anime", "score": 5.0},
                {"mal_id": 2, "title": "Yet Another Anime", "score": 5.0},
            ]},
        )
    )
    anilist_route = respx.post(ANILIST_GRAPHQL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"Page": {"media": [_anilist_media(999, "Test Anime")]}}},
        )
    )
    # Mock minimo de AnimeThemes (mismo motivo que en el test anterior):
    # resultado unico no ambiguo para el titulo base del archivo (el canon de
    # Jikan se rechaza por no confiable, asi que la consulta sigue siendo
    # "Test Anime"), para que la resolucion de slug no abra picker. El atajo
    # por ID externo (mal_id=1 o 2, cualquiera que _jikan_rank haya elegido)
    # tambien resuelve directo con el mismo resultado.
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}))
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}}))

    resultado, logs = _run_resolver(tmp_path)

    assert resultado.get("ok") is True
    assert not any("usando AniList como respaldo" in s for s in logs)
    assert anilist_route.call_count == 0


@respx.mock
def test_jikan_responde_200_vacio_prueba_anilist_con_la_misma_consulta(tmp_path):
    """Corregido (gap de docs/ideas.md #8/#9): Jikan responde 200 con
    data=[] -- busco y genuinamente no encontro nada, no es una falla de
    servicio. jikan_resolver_titulo no lanza excepcion en ese caso
    (devuelve (q, None, False, 0.0), ver jikan.py:224-225), asi que el
    fallback a AniList ahora vive en un `else:` del try/except (no en el
    except), gatillado explicitamente por picked_base is None. Se
    confirma que la consulta que llega a AniList es identica a la que
    se le mando a Jikan -- sin ningun ajuste, ya que no hay ninguna señal
    sobre POR QUE Jikan no encontro nada que justifique cambiarla. El
    mensaje de log es distinto al del camino de excepcion ("no encontró
    resultados", no "no disponible") porque Jikan si respondio."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": []}))

    consultas_anilist = []

    def _anilist_side_effect(request):
        body = json.loads(request.content)
        consultas_anilist.append(body.get("variables", {}).get("search"))
        return httpx.Response(
            200,
            json={"data": {"Page": {"media": [_anilist_media(999, "Test Anime")]}}},
        )
    respx.post(ANILIST_GRAPHQL).mock(side_effect=_anilist_side_effect)

    # Atajo por ID externo (id=999, idMal=900999) resuelve directo -- mismo
    # patron ya usado en los otros tests de este archivo.
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}))

    resultado, logs = _run_resolver(tmp_path)

    assert resultado.get("ok") is True
    assert resultado["params"].titulo_usado == "Test Anime"
    assert consultas_anilist == ["Test Anime"]  # misma consulta que se le mando a Jikan, sin ajustes
    assert any("Jikan no encontró resultados, probando AniList" in s for s in logs)
    # Mensaje distinto al del camino de excepcion -- Jikan si respondio.
    assert not any("Jikan no disponible" in s for s in logs)


@respx.mock
def test_jikan_200_vacio_y_anilist_tambien_vacio_propaga_sin_encontrados(tmp_path):
    """Si ademas AniList tampoco encuentra nada, el flujo debe terminar
    igual que hoy termina el caso "nadie encontro nada" -- mismo mensaje
    final ("No se encontraron resultados.") y mismo camino de respaldo
    interno de Jikan, sin ningun cambio de comportamiento ahi."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": []}))
    respx.post(ANILIST_GRAPHQL).mock(
        return_value=httpx.Response(200, json={"data": {"Page": {"media": []}}})
    )
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": []}}))

    resultado, logs = _run_resolver(tmp_path)

    assert resultado.get("ok") is False
    assert "Jikan no devolvió resultados." in resultado.get("error", "")
    assert any("Jikan no encontró resultados, probando AniList" in s for s in logs)
    assert "  - No se encontraron resultados." in logs
