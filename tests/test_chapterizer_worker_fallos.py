"""
Cobertura nueva para ChapterizerWorker.run() -- hasta ahora sin ningun
test (confirmado por grep antes de este cambio). Se agrega minima y
dirigida a los 2 caminos que cambiaron de comportamiento al eliminar el
checkbox "OP/ED exactos": ya no existe una rama que caiga a capitulos
heuristicos -- todo fallo de matching exacto ahora es un fallo real
(fallo.emit(), sin XML generado), no un resolved.emit() con datos
aproximados.

Se mockean ffmpeg y AnimeThemes al minimo necesario para llegar al punto
de interes sin tocar red ni disco de audio real -- el objetivo de estos
tests es la logica de decision de ChapterizerWorker.run(), no el pipeline
de audio en si (eso ya lo cubre test_audio_matching.py por separado).
"""
import numpy as np

from chapterizen.modelos import ParametrosTrabajo
from chapterizen.gui import chapterizer_worker as cw_mod
from chapterizen.gui.chapterizer_worker import ChapterizerWorker


def _worker(tmp_path, monkeypatch, wav_names, dur=1400.0, puntuacion_minima=0.25):
    video = tmp_path / "Test Anime - 01.mkv"
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida=str(tmp_path),
        crear_subcarpeta=False,
        search_override="",
        slug="test-anime",
        titulo_usado="Test Anime",
        episodio=1,
        puntuacion_minima=puntuacion_minima,
    )
    worker = ChapterizerWorker(None, params)

    monkeypatch.setattr(cw_mod, "asegurar_ffmpeg", lambda: None)
    monkeypatch.setattr(cw_mod, "duracion_con_ffprobe", lambda video: dur)
    monkeypatch.setattr(cw_mod, "obtener_anime_de_animethemes", lambda slug: {})
    monkeypatch.setattr(cw_mod, "construir_mapa_mostrar_temas", lambda anime_json: {})
    monkeypatch.setattr(cw_mod, "extraer_audio_wav_mono_16k", lambda video, out, ss, duracion: None)
    monkeypatch.setattr(cw_mod, "leer_pcm16_mono_wav", lambda path: (np.zeros(1000, dtype=np.float32), 16000))
    monkeypatch.setattr(cw_mod, "obtener_features_con_cache", lambda y, hz: np.zeros((13, 20), dtype=np.float32))

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(exist_ok=True)
    for name in wav_names:
        (wav_dir / f"{name}.wav").write_bytes(b"")
    monkeypatch.setattr(
        cw_mod, "construir_cache_temas",
        lambda slug, anime_json, log_fn, episodio: (wav_dir, set()),
    )

    logs = []
    worker.log.connect(logs.append)
    resultado = {}
    worker.terminado.connect(lambda ruta: resultado.update(ok=True, ruta=ruta))
    worker.fallo.connect(lambda msg: resultado.update(ok=False, error=msg))

    return worker, logs, resultado


def test_sin_temas_catalogados_falla_con_mensaje_de_slug_sin_generar_xml(tmp_path, monkeypatch):
    """Camino C: AnimeThemes no tiene ningun OP/ED para el slug resuelto
    (wav_dir queda sin ningun archivo OP*/ED*). Antes del rediseño esto ya
    fallaba (no es el camino que cambio de comportamiento) -- este test
    fija el mensaje especifico pedido, que ahora incluye el slug."""
    worker, logs, resultado = _worker(tmp_path, monkeypatch, wav_names=[])

    worker.run()

    assert resultado.get("ok") is False
    assert resultado["error"] == (
        "AnimeThemes no tiene ningún OP/ED catalogado para 'test-anime'"
        " — no se puede generar el XML sin matching exacto."
    )
    # No debe haberse escrito ningun XML.
    assert not list(tmp_path.glob("*.xml"))


def test_sin_match_suficiente_falla_en_vez_de_caer_a_heuristico(tmp_path, monkeypatch):
    """Camino D (el rediseño real): hay temas descargados (wav_dir trae
    OP1/ED1), pero el matching no encuentra nada por encima del umbral --
    se fuerza mockeando _buscar_con_ventana para que siempre devuelva None,
    sin depender de que el audio sintetico compute un score real bajo.
    Antes de este cambio, este caso generaba capitulos heuristicos y
    terminaba en terminado.emit(); ahora debe terminar en fallo.emit() sin
    escribir ningun XML."""
    worker, logs, resultado = _worker(
        tmp_path, monkeypatch, wav_names=["OP1", "ED1"], puntuacion_minima=0.25,
    )
    monkeypatch.setattr(worker, "_buscar_con_ventana", lambda **kwargs: None)

    worker.run()

    assert resultado.get("ok") is False
    assert resultado["error"] == (
        "No se encontró coincidencia de audio suficiente para OP ni ED"
        " (umbral=0.25) — no se generará XML."
    )
    assert not any(l.startswith("⚠️ No pude coincidir con OP/ED. Usando modo heurístico.") for l in logs)
    assert not list(tmp_path.glob("*.xml"))
