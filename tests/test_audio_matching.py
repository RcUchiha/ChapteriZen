"""
Regresion para el bug de aislamiento de _API_CACHE: audio_matching.py
hace "from .config import _API_CACHE" igual que jikan/anilist/animethemes
(cada modulo se queda con su propia referencia al objeto Cache), pero
quedo afuera de _MODULOS_CON_API_CACHE en conftest.py hasta este fix.
Sin el fix, obtener_features_con_cache() habria leido/escrito la cache
real de disco de produccion en cada corrida de test, no la cache
temporal aislada -- dormido hasta ahora porque ningun test ejercitaba
ese codigo (confirmado por grep antes del fix), pero listo para activarse
apenas alguien agregara un test como este.
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

    # La cache que efectivamente se uso debe ser la misma instancia de
    # prueba que ya comparten jikan/anilist/animethemes gracias al fixture
    # _fresh_api_cache -- no el objeto Cache original de config.py (que
    # apunta a disco de produccion). Sin audio_matching en
    # _MODULOS_CON_API_CACHE, esta identidad seria distinta.
    assert cz_audio._API_CACHE is cz_jikan._API_CACHE
