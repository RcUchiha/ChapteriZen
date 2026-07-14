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
