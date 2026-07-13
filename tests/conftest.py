import sys
from pathlib import Path

import pytest
from diskcache import Cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chapterizen as cz


@pytest.fixture(autouse=True)
def _fresh_api_cache(tmp_path, monkeypatch):
    """Aisla cada test de _API_CACHE (diskcache real en disco de producción)
    para que ninguna corrida anterior/posterior contamine resultados ni los
    tests compartan estado entre si."""
    cache = Cache(str(tmp_path / "test_api_cache"))
    monkeypatch.setattr(cz, "_API_CACHE", cache)
    yield
    cache.close()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """tenacity.nap.sleep llama a time.sleep(seconds) -- sin esto, cada
    reintento de _reintento_http esperaria hasta 8s de verdad (wait_exponential)."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)
