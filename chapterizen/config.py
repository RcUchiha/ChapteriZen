"""Configuracion global: constantes, endpoints, cliente HTTP, cache en
disco, logging y politica de reintentos. Movido sin cambios desde
chapterizen.py (monolito original, v0.0.7)."""
import httpx
from pathlib import Path
from platformdirs import user_cache_dir, user_log_dir
from diskcache import Cache
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)


__author__  = "CiferrC"
__license__ = "MIT"
__version__ = "0.0.8"

# ─────────────────────────────────────────────
#  CLIENTE HTTP
# ─────────────────────────────────────────────

_TIMEOUTS = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)

_http = httpx.Client(
    timeout=_TIMEOUTS,
    follow_redirects=True,
    headers={"User-Agent": f"ChapteriZen/{__version__}"},
)

# ─────────────────────────────────────────────
#  CACHÉ EN DISCO (diskcache + platformdirs)
# ─────────────────────────────────────────────

# Carpeta estándar del SO: ~/.cache/ChapteriZen en Linux,
# %LOCALAPPDATA%\ChapteriZen\Cache en Windows
_CACHE_DIR   = Path(user_cache_dir("ChapteriZen"))
_THEMES_DIR  = _CACHE_DIR / "themes"          # audios OGG/WAV por slug
_DC_PATH     = _CACHE_DIR / "api_cache"       # respuestas de API (diskcache)
_api_cache   = Cache(_DC_PATH)                # TTL configurable por entrada

_TTL_API_DAYS    = 7    # respuestas de AnimeThemes/Jikan se cachean 7 días
_TTL_THEMES_DAYS = 30   # metadatos de temas se cachean 30 días

def get_api_cache() -> Cache:
    """Accessor del singleton de cache compartido -- los modulos que
    necesitan la cache deben llamar a esta funcion en vez de importar
    _api_cache directamente. Como la funcion resuelve el global de este
    modulo en cada llamada (no en el momento del import), los tests
    pueden swapear toda la cache con un solo monkeypatch.setattr(config,
    "_api_cache", ...) sin tener que parchear cada modulo importador por
    separado (ver tests/conftest.py)."""
    return _api_cache

# ─────────────────────────────────────────────
#  LOGGING (loguru)
# ─────────────────────────────────────────────

_LOG_DIR = Path(user_log_dir("ChapteriZen"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Quitar el sink de stderr por defecto y añadir solo el archivo rotativo.
# La GUI muestra los logs a través de las señales Qt — no necesitamos stderr.
logger.remove()
logger.add(
    _LOG_DIR / "chapterizen_{time:YYYY-MM-DD}.log",
    rotation="1 day",       # un archivo por día
    retention="14 days",    # conservar 2 semanas
    encoding="utf-8",
    level="DEBUG",
    format="{time:HH:mm:ss} | {level:<8} | {message}",
)

# ─────────────────────────────────────────────
#  REINTENTOS (tenacity)
# ─────────────────────────────────────────────

def _es_error_transitorio(exc: BaseException) -> bool:
    """Reintenta solo en timeouts y errores HTTP recuperables."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504, 522}
    return False

_reintento_http = retry(
    retry=retry_if_exception(_es_error_transitorio),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

TRACE_ENDPOINT     = "https://api.trace.moe/search"
ANILIST_GRAPHQL    = "https://graphql.anilist.co"
ANIMETHEMES_SEARCH = "https://api.animethemes.moe/search"
ANIMETHEMES_ANIME  = "https://api.animethemes.moe/anime"
JIKAN_ANIME        = "https://api.jikan.moe/v4/anime"
JIKAN_REL          = "https://api.jikan.moe/v4/anime/{id}/relations"

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".webm", ".mov", ".m2ts", ".ts", ".wmv", ".vob")
