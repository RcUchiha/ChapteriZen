import sys
from pathlib import Path

import pytest
from diskcache import Cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapterizen import config as cz_config
from chapterizen import trace_moe as cz_trace_moe
from chapterizen import animethemes as cz_animethemes
from chapterizen import jikan as cz_jikan

# Modulos que hicieron "from .config import _API_CACHE" -- cada uno tiene su
# propia referencia local al objeto Cache, asi que hay que parchear las 4
# (no solo config._API_CACHE) para que todos vean la misma cache de prueba.
_MODULOS_CON_API_CACHE = (cz_config, cz_trace_moe, cz_animethemes, cz_jikan)


@pytest.fixture(autouse=True)
def _fresh_api_cache(tmp_path, monkeypatch):
    """Aisla cada test de _API_CACHE (diskcache real en disco de produccion)
    para que ninguna corrida anterior/posterior contamine resultados ni los
    tests compartan estado entre si."""
    cache = Cache(str(tmp_path / "test_api_cache"))
    for modulo in _MODULOS_CON_API_CACHE:
        monkeypatch.setattr(modulo, "_API_CACHE", cache)
    yield
    cache.close()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """tenacity.nap.sleep llama a time.sleep(seconds) -- sin esto, cada
    reintento de _reintento_http esperaria hasta 8s de verdad (wait_exponential)."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
