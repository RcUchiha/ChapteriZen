"""
Tests para el fallback Jikan -> AniList en ResolverWorker.run(): se activa
UNICAMENTE cuando jikan_resolver_titulo agota sus reintentos (503/504
persistente) -- no cuando Jikan responde con confiable=False.
"""
import httpx
import respx

from chapterizen.modelos import ParametrosTrabajo
from chapterizen.gui.workers import ResolverWorker

JIKAN_ANIME     = "https://api.jikan.moe/v4/anime"
ANILIST_GRAPHQL = "https://graphql.anilist.co"


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
        usar_exacto=False,  # basta con resolver el titulo, no necesitamos AnimeThemes/audio
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

    resultado, logs = _run_resolver(tmp_path)

    assert resultado.get("ok") is True
    assert not any("usando AniList como respaldo" in s for s in logs)
    assert anilist_route.call_count == 0
