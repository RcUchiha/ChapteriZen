import sys
from pathlib import Path

import pytest
from diskcache import Cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapterizen import config as cz_config

# jikan, animethemes, anilist y audio_matching obtienen la cache llamando a
# config.get_api_cache() en cada uso (no guardan una referencia propia al
# objeto Cache), asi que alcanza con parchear el singleton de config para
# que todos vean la misma cache de prueba -- ver get_api_cache() en config.py.


@pytest.fixture(autouse=True)
def _fresh_api_cache(tmp_path, monkeypatch):
    """Aisla cada test de _api_cache (diskcache real en disco de produccion)
    para que ninguna corrida anterior/posterior contamine resultados ni los
    tests compartan estado entre si."""
    cache = Cache(str(tmp_path / "test_api_cache"))
    monkeypatch.setattr(cz_config, "_api_cache", cache)
    yield
    cache.close()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """tenacity.nap.sleep llama a time.sleep(seconds) -- sin esto, cada
    reintento de _reintento_http esperaria hasta 8s de verdad (wait_exponential)."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
