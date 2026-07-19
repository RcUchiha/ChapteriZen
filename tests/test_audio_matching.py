"""
Regresion para el bug de aislamiento de cache que motivo el accessor
get_api_cache() (ver config.py): audio_matching.py hacia
"from .config import _API_CACHE" igual que jikan/anilist/animethemes
(cada modulo se quedaba con su propia referencia al objeto Cache), pero
quedo afuera de la lista de modulos parcheados en conftest.py hasta que
se detecto. Sin ese fix, obtener_features_con_cache() habria leido/escrito
la cache real de disco de produccion en cada corrida de test.

Ahora todos los modulos llaman a config.get_api_cache() en vez de guardar
su propia referencia, asi que conftest.py solo necesita parchear
config._api_cache una vez (ver _fresh_api_cache). Este test sigue de
guardia por si algun modulo nuevo vuelve a importar el objeto Cache
directamente en vez de pasar por el accessor.
"""
import numpy as np

from chapterizen import audio_matching as cz_audio
from chapterizen import jikan as cz_jikan


def test_audio_matching_usa_la_cache_de_prueba_no_la_de_produccion(monkeypatch):
    llamadas_a_extraer = []

    def _fake_extraer_features(y, sr):
        llamadas_a_extraer.append((y, sr))
        return np.zeros((32, 10), dtype=np.float32)

    monkeypatch.setattr(cz_audio, "extraer_features", _fake_extraer_features)

    y = np.ones(1000, dtype=np.float32)

    feat1 = cz_audio.obtener_features_con_cache(y, 16000)
    assert len(llamadas_a_extraer) == 1  # primera vez: no estaba en cache, se calculo

    feat2 = cz_audio.obtener_features_con_cache(y, 16000)
    assert len(llamadas_a_extraer) == 1  # segunda vez: vino de cache, no se recalculo
    np.testing.assert_array_equal(feat1, feat2)

    # audio_matching y jikan deben resolver al mismo objeto Cache de prueba
    # (el que parcheo _fresh_api_cache), no al de disco de produccion.
    assert cz_audio.get_api_cache() is cz_jikan.get_api_cache()


class TestResamplearAudio:
    """Caracterizacion de _resamplear_audio (gui/chapterizer_worker.py llama
    a esta funcion como red de seguridad para el caso raro -- casi nunca
    ocurre en la practica, ya que construir_cache_temas fuerza 16kHz via
    ffmpeg al descargar cada tema -- de un WAV que no vino a 16kHz)."""

    def test_longitud_de_salida_coincide_con_la_razon_de_tasas(self):
        y   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        out = cz_audio._resamplear_audio(y, hz_orig=10, hz_target=16)
        assert len(out) == 16

    def test_valores_de_referencia_librosa_resample(self):
        """Valores exactos de la implementacion actual (librosa.resample,
        respaldado por soxr) -- de rampa 0..9 a 10Hz, resampleada a 16Hz.
        Antes de este cambio se usaba interpolacion lineal manual (via
        np.interp); reemplazada tras validar contra audio real que
        librosa.resample da una señal mas fiel al original sin diferencia
        practica en el costo DTW resultante (ver docs/TECH_DEBT.md). Los
        valores de referencia cambian bastante respecto a la interpolacion
        lineal en esta rampa sintetica de 10 muestras por efectos de borde
        del resampler por sinc -- esperable en una señal tan corta y no
        periodica, no una señal de alarma (ver la comparacion contra audio
        real en docs/TECH_DEBT.md)."""
        y   = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        out = cz_audio._resamplear_audio(y, hz_orig=10, hz_target=16)
        esperado = np.array([
            0.15715376, 0.54374403, 1.11193871, 2.0137434,  2.59370518,
            2.90392971, 3.76905274, 4.64135456, 4.79975843, 5.40656137,
            6.68146515, 6.87969685, 6.80865288, 8.70031166, 9.74973106,
            5.81591272,
        ], dtype=np.float32)
        np.testing.assert_allclose(out, esperado, rtol=1e-5)

    def test_dtype_es_float32(self):
        y   = np.array([0, 1, 2, 3], dtype=np.float64)
        out = cz_audio._resamplear_audio(y, hz_orig=8, hz_target=16)
        assert out.dtype == np.float32
