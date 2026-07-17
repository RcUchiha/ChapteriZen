"""
Cobertura del atajo de resolucion de slug por ID externo de AnimeThemes
(filter[has]=resources&filter[site]=...&filter[external_id]=...),
agregado en _resolver_slug_con_picker (gui/resolver_worker.py) como
camino PRINCIPAL antes del texto -- confirmado que el endpoint funciona
asi contra AnimeThemes real (ver docs/TECH_DEBT.md).

Cada test verifica, ademas del resultado, si ANIMETHEMES_SEARCH (el
camino de texto) llego a ser llamado o no -- esa es la senal real de si
el atajo funciono como reemplazo (0 llamadas) o como se esperaba que
cayera al respaldo (>=1 llamada).
"""
import httpx
import respx

from chapterizen.modelos import ParametrosTrabajo, AnimeDetectado
from chapterizen.gui.resolver_worker import ResolverWorker

JIKAN_ANIME        = "https://api.jikan.moe/v4/anime"
ANILIST_GRAPHQL    = "https://graphql.anilist.co"
ANIMETHEMES_SEARCH = "https://api.animethemes.moe/search"
ANIMETHEMES_ANIME  = "https://api.animethemes.moe/anime"


def _worker(tmp_path, video_name, search_override=""):
    video = tmp_path / video_name
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida="",
        crear_subcarpeta=False,
        search_override=search_override,
    )
    worker = ResolverWorker(None, params, interactivo=False)

    logs = []
    worker.log.connect(logs.append)

    resultado = {}
    worker.resolved.connect(lambda p: resultado.update(ok=True, params=p))
    worker.failed.connect(lambda msg: resultado.update(ok=False, error=msg))

    return worker, logs, resultado


@respx.mock
def test_atajo_por_id_resuelve_directo_sin_tocar_el_camino_de_texto(tmp_path):
    """Jikan resuelve un unico resultado confiable (mal_id=1). El atajo
    por ID debe resolver el slug directo -- ANIMETHEMES_SEARCH no debe
    llamarse en absoluto."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 1, "title": "Test Anime", "type": "TV", "episodes": 12, "score": 8.0},
    ]}))
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}))
    ruta_texto = respx.get(ANIMETHEMES_SEARCH).mock(
        return_value=httpx.Response(200, json={"search": {"anime": []}})
    )

    worker, logs, resultado = _worker(tmp_path, "Test Anime - 01.mkv")
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "test-anime"
    assert resultado["params"].titulo_usado == "Test Anime"
    assert ruta_texto.call_count == 0
    assert any("match directo por MyAnimeList ID 1" in l for l in logs)


@respx.mock
def test_atajo_por_id_sin_match_cae_al_camino_de_texto(tmp_path):
    """AnimeThemes no tiene ningun recurso enlazado para el mal_id
    resuelto (0 resultados) -- debe caer exactamente al camino de texto
    de siempre, sin ningun cambio de comportamiento ahi."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 1, "title": "Test Anime", "type": "TV", "episodes": 12, "score": 8.0},
    ]}))
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": []}))
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Test Anime", "year": 2024, "season": None, "slug": "test-anime"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Test Anime - 01.mkv")
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "test-anime"
    assert not any("match directo por" in l for l in logs)


@respx.mock
def test_atajo_por_id_con_recurso_mal_enlazado_no_se_acepta_a_ciegas(tmp_path):
    """AnimeThemes SI tiene un recurso enlazado para el mal_id resuelto,
    pero el nombre que devuelve no comparte tokens con NINGUNO de los
    titulos que Jikan ya conoce para ese item (recurso externo mal
    mapeado dentro de AnimeThemes) -- la validacion cruzada debe
    rechazar el atajo y caer al camino de texto, que en este escenario
    si encuentra el anime correcto por otra via."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 1, "title": "Sutetsuyo", "type": "TV", "episodes": 12, "score": 8.0},
    ]}))
    # Recurso externo enlazado a una pagina de un anime totalmente
    # distinto -- ninguno de los titulos conocidos de picked_base
    # ("Sutetsuyo") comparte tokens de 4+ letras con esto.
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": [
        {"name": "Completely Unrelated Series Name", "year": 2019, "season": None, "slug": "completely-unrelated"},
    ]}))
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Ansatsusha de Aru Ore no Status ga Yuusha yori mo Akiraka ni Tsuyoi no da ga",
         "year": 2025, "season": None, "slug": "ansatsusha-de-aru-ore"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Sutetsuyo - 01.mkv")
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "ansatsusha-de-aru-ore"
    assert not any("match directo por" in l for l in logs)


@respx.mock
def test_override_salta_el_atajo_por_id_por_completo(tmp_path):
    """Si el usuario escribe una busqueda manual (search_override), el
    atajo por ID no debe intentarse -- ni siquiera se resuelve Jikan/
    AniList en ese camino (mismo criterio ya existente para jikan_item).
    Se confirma con ANIMETHEMES_ANIME.call_count == 0."""
    ruta_id = respx.get(ANIMETHEMES_ANIME).mock(
        return_value=httpx.Response(200, json={"anime": []})
    )
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Override Anime", "year": 2024, "season": None, "slug": "override-anime"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "cualquier_nombre.mkv", search_override="Override Anime")
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "override-anime"
    assert ruta_id.call_count == 0


@respx.mock
def test_atajo_por_id_en_camino_anilist_confirmado_por_trace_moe(tmp_path, monkeypatch):
    """Cuando trace.moe identifica con confianza alta (anilist_confirmado),
    no hay picked_base (Jikan se omite) -- el atajo debe usar
    detectado_anilist_id directamente, validado contra
    anilist_titulo_por_id (unico titulo conocido disponible en esta
    rama, ver limitacion documentada en _token_ok_contra_titulos_conocidos)."""
    # Filename puramente numerico -> dispara trace.moe en produccion.
    worker, logs, resultado = _worker(tmp_path, "12345.mkv")
    monkeypatch.setattr(
        worker, "_identificar_con_trace_moe",
        lambda video: AnimeDetectado(titulo="Some Anime", anilist_id=555, episodio=5, similitud=0.97),
    )

    respx.post(ANILIST_GRAPHQL).mock(return_value=httpx.Response(200, json={
        "data": {"Media": {"title": {"romaji": "Some Anime"}}}
    }))
    respx.get(ANIMETHEMES_ANIME).mock(return_value=httpx.Response(200, json={"anime": [
        {"name": "Some Anime", "year": 2024, "season": None, "slug": "some-anime"},
    ]}))
    ruta_texto = respx.get(ANIMETHEMES_SEARCH).mock(
        return_value=httpx.Response(200, json={"search": {"anime": []}})
    )

    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "some-anime"
    assert ruta_texto.call_count == 0
    assert any("match directo por Anilist ID 555" in l for l in logs)
