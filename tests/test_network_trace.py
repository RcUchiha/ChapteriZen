"""
Tests con red mockeada (respx) para identificar_anime_con_fotogramas(),
el pipeline de identificacion por lotes/consenso de trace.moe.

Las respuestas se enrutan segun el contenido del cuerpo de cada POST
(un marcador embebido en los bytes del "frame"), no segun el orden de
llegada -- los frames de un mismo lote se envian en paralelo via
ThreadPoolExecutor y su orden de ejecucion no esta garantizado.
"""
import json

import httpx
import pytest
import respx
from loguru import logger

from chapterizen import trace_moe as cz

TRACE_ENDPOINT = "https://api.trace.moe/search"
ANILIST_GRAPHQL = "https://graphql.anilist.co"


def _frame(tmp_path, name: str, marker: bytes):
    p = tmp_path / name
    p.write_bytes(marker)
    return p


def _mock_trace_por_marcador(respuestas: dict):
    """respuestas: {marcador_bytes: dict_json_result}"""

    def _side_effect(request: httpx.Request) -> httpx.Response:
        body = request.content
        for marcador, resultado in respuestas.items():
            if marcador in body:
                return httpx.Response(200, json={"result": resultado})
        return httpx.Response(200, json={"result": []})

    respx.post(TRACE_ENDPOINT).mock(side_effect=_side_effect)


def _mock_anilist_titulo(titulo: str):
    respx.post(ANILIST_GRAPHQL).mock(
        return_value=httpx.Response(
            200, json={"data": {"Media": {"title": {"romaji": titulo}}}}
        )
    )


@respx.mock
def test_consenso_rapido_con_tres_frames(tmp_path):
    """3 frames, mismo anime con similitud alta y mismo episodio -> debe
    resolver en el segundo lote (3 frames enviados de un maximo de 9)."""
    frames = [
        _frame(tmp_path, "frame_001.jpg", b"FRAME_A"),
        _frame(tmp_path, "frame_002.jpg", b"FRAME_B"),
        _frame(tmp_path, "frame_003.jpg", b"FRAME_C"),
    ]
    _mock_trace_por_marcador({
        b"FRAME_A": [{"anilist": 100, "similarity": 0.97, "episode": 5, "filename": ""}],
        b"FRAME_B": [{"anilist": 100, "similarity": 0.96, "episode": 5, "filename": ""}],
        b"FRAME_C": [{"anilist": 100, "similarity": 0.98, "episode": 5, "filename": ""}],
    })
    _mock_anilist_titulo("Test Anime")

    detectado, n_usados = cz.identificar_anime_con_fotogramas(frames)

    assert detectado.titulo == "Test Anime"
    assert detectado.anilist_id == 100
    assert detectado.episodio == 5
    assert n_usados == 3  # salida temprana, no se usaron los 9 disponibles


@respx.mock
def test_dispersion_de_episodio_sin_mayoria_devuelve_episodio_none(tmp_path):
    """Serie resuelta (mismo anilist, similitud alta) pero el episodio
    queda 1 vs 1 (con un frame sin dato de episodio) y no hay mas frames
    para desempatar -> episodio final debe ser None."""
    frames = [
        _frame(tmp_path, "frame_001.jpg", b"FRAME_A"),
        _frame(tmp_path, "frame_002.jpg", b"FRAME_B"),
        _frame(tmp_path, "frame_003.jpg", b"FRAME_C"),
    ]
    _mock_trace_por_marcador({
        b"FRAME_A": [{"anilist": 100, "similarity": 0.97, "episode": 5, "filename": ""}],
        b"FRAME_B": [{"anilist": 100, "similarity": 0.96, "episode": 6, "filename": ""}],
        b"FRAME_C": [{"anilist": 100, "similarity": 0.95, "episode": None, "filename": ""}],
    })
    _mock_anilist_titulo("Test Anime Disperso")

    detectado, n_usados = cz.identificar_anime_con_fotogramas(frames)

    assert detectado.titulo == "Test Anime Disperso"
    assert detectado.anilist_id == 100
    assert detectado.episodio is None
    assert n_usados == 3


@respx.mock
def test_sin_resultados_lanza_runtime_error(tmp_path):
    frames = [
        _frame(tmp_path, "frame_001.jpg", b"FRAME_A"),
        _frame(tmp_path, "frame_002.jpg", b"FRAME_B"),
    ]
    _mock_trace_por_marcador({})  # todas las respuestas caen al default: result=[]

    with pytest.raises(RuntimeError, match="no pudo identificar"):
        cz.identificar_anime_con_fotogramas(frames)


@respx.mock
def test_fallo_de_frame_queda_logueado_no_silenciado(tmp_path):
    """Regresion: un fotograma que falla (ej. 429 persistente tras agotar
    reintentos) antes solo se descartaba con 'except Exception: pass' --
    ni un solo rastro en ningun log, ni siquiera a nivel DEBUG. Confirmado
    en la practica: una corrida real contra trace.moe que aparento estar
    rate-limiteada solo dejo "0 devolvieron resultado utilizable" en el
    log, sin ninguna pista de si la causa fue 429, timeout u otra cosa.
    Ahora debe quedar un mensaje DEBUG con el tipo y texto de la excepcion
    real antes de seguir con el siguiente fotograma."""
    respx.post(TRACE_ENDPOINT).mock(
        return_value=httpx.Response(429, json={"error": "Too Many Requests"})
    )

    mensajes: list = []
    sink_id = logger.add(mensajes.append, level="DEBUG")
    try:
        frames = [
            _frame(tmp_path, "frame_001.jpg", b"FRAME_A"),
            _frame(tmp_path, "frame_002.jpg", b"FRAME_B"),
        ]
        with pytest.raises(RuntimeError, match="no pudo identificar"):
            cz.identificar_anime_con_fotogramas(frames)
    finally:
        logger.remove(sink_id)

    fallos_logueados = [m for m in mensajes if "fotograma falló" in m]
    assert fallos_logueados, "no quedo ningun log del fallo real de fotograma"
    assert any("HTTPStatusError" in m and "429" in m for m in fallos_logueados)


@respx.mock
def test_solo_402_lanza_error_de_cuota_agotada(tmp_path):
    """Si el 100% de los intentos de fotograma fallan con 402 (confirmado
    por status code -- cuota anonima de trace.moe agotada, ver
    docs/KNOWN_LIMITATIONS.md), el mensaje debe ser especifico de cuota
    agotada, no el generico "no pudo identificar" -- para que el usuario
    entienda que no es un problema de reconocimiento sino de limite de
    la API, y sepa que probar mas tarde (o con el nombre de archivo)
    tiene sentido."""
    respx.post(TRACE_ENDPOINT).mock(
        return_value=httpx.Response(402, json={"error": "Payment Required"})
    )
    frames = [
        _frame(tmp_path, "frame_001.jpg", b"FRAME_A"),
        _frame(tmp_path, "frame_002.jpg", b"FRAME_B"),
    ]

    with pytest.raises(RuntimeError, match="no tiene cuota disponible"):
        cz.identificar_anime_con_fotogramas(frames)


@respx.mock
def test_mezcla_de_causas_no_usa_mensaje_de_cuota(tmp_path):
    """Si solo ALGUNOS fotogramas fallan por 402 y otros por una causa
    distinta (ej. timeout persistente), NO debe usarse el mensaje de
    cuota agotada -- no hay certeza de que la cuota sea realmente el
    problema (podria ser una mezcla de cuota + un problema de red
    aparte), asi que se mantiene el mensaje generico existente en vez de
    aparentar una causa que no esta confirmada al 100%."""
    def _side_effect(request: httpx.Request) -> httpx.Response:
        body = request.content
        if b"FRAME_402" in body:
            return httpx.Response(402, json={"error": "Payment Required"})
        raise httpx.ConnectTimeout("boom")

    respx.post(TRACE_ENDPOINT).mock(side_effect=_side_effect)

    frames = [
        _frame(tmp_path, "frame_001.jpg", b"FRAME_402"),
        _frame(tmp_path, "frame_002.jpg", b"FRAME_TIMEOUT"),
    ]

    with pytest.raises(RuntimeError) as exc_info:
        cz.identificar_anime_con_fotogramas(frames)

    assert "no tiene cuota disponible" not in str(exc_info.value)
    assert "no pudo identificar" in str(exc_info.value)
