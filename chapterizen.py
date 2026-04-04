#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__  = "CiferrC"
__license__ = "MIT"
__version__ = "0.0.6"

import re
import json
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import librosa
from xml.sax.saxutils import escape
from rapidfuzz import fuzz as _fuzz
from pydantic import BaseModel, Field
from platformdirs import user_cache_dir, user_log_dir
from diskcache import Cache
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QGroupBox,
    QFrame,
    QProgressBar,
)

# Parsers de nombres de archivo de anime (aniparse principal, anitopy fallback)
try:
    import aniparse as _aniparse
    _ANIPARSE_OK = True
except ImportError:
    _aniparse    = None
    _ANIPARSE_OK = False

try:
    import anitopy as _anitopy
    _ANITOPY_OK = True
except ImportError:
    _anitopy    = None
    _ANITOPY_OK = False

# scipy es opcional: si está disponible se usa para FFT más rápida
try:
    from scipy.fft import rfft, irfft, next_fast_len as _scipy_next_fast_len
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

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
_API_CACHE   = Cache(_DC_PATH)                # TTL configurable por entrada

_TTL_API_DAYS    = 7    # respuestas de AnimeThemes/Jikan se cachean 7 días
_TTL_THEMES_DAYS = 30   # metadatos de temas se cachean 30 días

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
        return exc.response.status_code in {429, 500, 502, 503, 504}
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
LITTERBOX_API      = "https://litterbox.catbox.moe/resources/internals/api.php"
ANIMETHEMES_SEARCH = "https://api.animethemes.moe/search"
ANIMETHEMES_ANIME  = "https://api.animethemes.moe/anime"
JIKAN_ANIME        = "https://api.jikan.moe/v4/anime"
JIKAN_REL          = "https://api.jikan.moe/v4/anime/{id}/relations"

VIDEO_EXTS = (".mkv", ".mp4", ".avi", ".webm", ".mov", ".m2ts", ".ts", ".wmv", ".vob")

# ─────────────────────────────────────────────
#  MODELOS PYDANTIC
# ─────────────────────────────────────────────

@dataclass
class ParsedAnime:
    """Resultado normalizado del parseo de un nombre de archivo de anime."""
    titulo:    str           # título limpio, listo para consultar Jikan
    temporada: Optional[int] # None si no se detectó
    episodio:  Optional[int] # None si no se detectó
    fuente:    str           # "aniparse" | "anitopy" | "merge" | "regex"


@dataclass
class TemaAudio:
    """Audio de un tema OP/ED precargado en memoria, listo para matching."""
    nombre:   str
    audio:    "np.ndarray"
    hz:       int
    frames:   int          # len(audio) // _HOP_LENGTH — precalculado
    features: "np.ndarray" # MFCC + chroma precalculados — evita recalcular en cada ventana


@dataclass
class CandidatoFFT:
    """Resultado de la fase FFT para un tema candidato."""
    tema:      "TemaAudio" # referencia al tema original — sin copiar arrays
    inicio:    float       # segundos en la ventana
    fin:       float       # segundos en la ventana
    score_fft: float


class AnimeDetectado(BaseModel):
    titulo:     str
    anilist_id: Optional[int] = None
    episodio:   Optional[int] = None
    similitud:  float

class ResultadoCoincidencia(BaseModel):
    nombre_tema: str
    inicio:      float
    fin:         float
    puntuacion:  float

class ParametrosTrabajo(BaseModel):
    video:             str
    carpeta_salida:    str
    crear_subcarpeta:  bool
    usar_exacto:       bool
    submuestreo:       int   = Field(default=32,   ge=1)
    porcion_theme:     float = Field(default=0.90, ge=0.5, le=1.0)
    puntuacion_minima: float = Field(default=0.25, ge=0.05, le=1.0)

    search_override: str = ""
    slug:            str = ""
    titulo_usado:    str = ""
    episodio:        int = 0

    model_config = {"arbitrary_types_allowed": True}

class PickRequest(BaseModel):
    kind:      str
    titulo:    str
    subtitulo: str
    columnas:  List[Tuple[str, int]]
    filas:     List[List[str]]
    payload:   List[dict]

    model_config = {"arbitrary_types_allowed": True}

# ─────────────────────────────────────────────
#  UTILIDADES SISTEMA / FFMPEG
# ─────────────────────────────────────────────

def log_clv(log, titulo: str, **kv):
    parts = [f"{k}={v!r}" for k, v in kv.items()]
    log(f"  - {titulo}: " + ", ".join(parts))

def ejecutar_comando(args: list[str]) -> str:
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        err    = (p.stderr or "").strip()
        salida = (p.stdout or "").strip()
        raise RuntimeError(err or salida or f"Comando falló: {args}")
    return p.stdout or ""

def asegurar_ffmpeg():
    ejecutar_comando(["ffmpeg",  "-version"])
    ejecutar_comando(["ffprobe", "-version"])

def duracion_con_ffprobe(ruta_video: str) -> float:
    salida = ejecutar_comando([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        ruta_video,
    ])
    return float(json.loads(salida)["format"]["duration"])

def extraer_fotogramas(
    ruta_video:     str,
    dir_salida:     str,
    cada_seg:       int = 120,
    max_fotogramas: int = 12,
) -> List[Path]:
    salida = Path(dir_salida)
    salida.mkdir(parents=True, exist_ok=True)
    ejecutar_comando([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", ruta_video,
        "-vf", f"fps=1/{cada_seg}",
        "-frames:v", str(max_fotogramas),
        str(salida / "frame_%03d.jpg"),
    ])
    return sorted(salida.glob("frame_*.jpg"))

def extraer_audio_wav_mono_16k(
    src_path: str,
    wav_path: str,
    ss:       Optional[float] = None,
    duracion: Optional[float] = None,
):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    cmd += ["-i", src_path]
    if duracion is not None:
        cmd += ["-t", str(duracion)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path]
    ejecutar_comando(cmd)

def leer_pcm16_mono_wav(path: str) -> Tuple[np.ndarray, int]:
    import wave
    with wave.open(path, "rb") as wf:
        ch        = wf.getnchannels()
        hz        = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n         = wf.getnframes()
        raw       = wf.readframes(n)

    if ch != 1 or sampwidth != 2:
        raise RuntimeError(
            f"WAV inesperado (ch={ch}, sampwidth={sampwidth}). "
            "Reexporta con ffmpeg."
        )
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return y, hz

# ─────────────────────────────────────────────
#  TRACE.MOE
# ─────────────────────────────────────────────

def subir_imagen_a_litterbox(img_bytes: bytes, expira: str = "1h") -> str:
    r = _http.post(
        LITTERBOX_API,
        data={"reqtype": "fileupload", "time": expira},
        files={"fileToUpload": ("frame.jpg", img_bytes, "image/jpeg")},
    )
    r.raise_for_status()
    url = r.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Litterbox no devolvió URL válida: {url!r}")
    return url

def trace_buscar_por_bytes(img_bytes: bytes) -> dict:
    img_url = subir_imagen_a_litterbox(img_bytes, expira="1h")
    r = _http.get(TRACE_ENDPOINT, params={"url": img_url})
    r.raise_for_status()
    return r.json()

def identificar_anime_con_fotogramas(rutas_fotogramas: List[Path]) -> AnimeDetectado:
    mejor     = None
    mejor_sim = -1.0

    for fp in rutas_fotogramas:
        data       = trace_buscar_por_bytes(fp.read_bytes())
        resultados = data.get("result") or []
        if not resultados:
            continue
        top1 = resultados[0]
        sim  = float(top1.get("similarity", 0.0))
        if sim > mejor_sim:
            mejor_sim = sim
            mejor     = top1

    if not mejor:
        raise RuntimeError("trace.moe no pudo identificar el anime con los fotogramas.")

    anilist_id = mejor.get("anilist")
    try:
        anilist_id = int(anilist_id) if anilist_id is not None else None
    except Exception:
        anilist_id = None

    ep = mejor.get("episode")
    try:
        ep = int(ep) if ep is not None else None
    except Exception:
        ep = None

    return AnimeDetectado(
        titulo=mejor.get("filename") or "Anime",
        anilist_id=anilist_id,
        episodio=ep,
        similitud=float(mejor_sim),
    )

# ─────────────────────────────────────────────
#  ANIMETHEMES
# ─────────────────────────────────────────────

@_reintento_http
def buscar_anime_en_animethemes(nombre_busqueda: str) -> List[dict]:
    clave = f"at_search:{nombre_busqueda.strip().casefold()}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    r = _http.get(
        ANIMETHEMES_SEARCH,
        params={"fields[search]": "anime", "q": nombre_busqueda},
    )
    r.raise_for_status()
    js      = r.json()
    result  = (((js or {}).get("search") or {}).get("anime") or [])
    _API_CACHE.set(clave, result, expire=_TTL_API_DAYS * 86400)
    return result

def obtener_anime_de_animethemes(slug: str) -> dict:
    clave  = f"at_anime:{slug}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached

    url      = f"{ANIMETHEMES_ANIME}/{slug}"
    intentos = [
        {"include": "animethemes.song.artists,animethemes.animethemeentries.videos.audio"},
        {"include": "animethemes.animethemeentries.videos.audio"},
        {},
    ]
    ultimo = None

    @_reintento_http
    def _get(params):
        r = _http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    for params in intentos:
        try:
            js = _get(params)
            if isinstance(js, dict) and "anime" in js and isinstance(js["anime"], dict):
                result = js["anime"]
            elif isinstance(js, dict) and ("animethemes" in js or "name" in js or "slug" in js):
                result = js
            else:
                result = {}
            _API_CACHE.set(clave, result, expire=_TTL_THEMES_DAYS * 86400)
            return result
        except httpx.HTTPStatusError as e:
            ultimo = e
            if e.response.status_code == 422:
                continue
            raise
    raise RuntimeError(f"AnimeThemes: no pude obtener /anime/{slug}. Último error: {ultimo}")

def construir_mapa_mostrar_temas(anime_json: dict) -> Dict[str, str]:
    salida: Dict[str, str] = {}
    for tema in (anime_json.get("animethemes") or []):
        slug_tema_raw = (tema.get("slug") or "").strip()
        slug_tema     = re.sub(r"v\d+$", "", slug_tema_raw, flags=re.I)
        if not slug_tema:
            continue
        cancion  = tema.get("song") or {}
        titulo   = (cancion.get("title") or "").strip()
        artistas = [
            (a.get("name") or "").strip()
            for a in (cancion.get("artists") or [])
            if (a.get("name") or "").strip()
        ]
        if not titulo or not artistas:
            continue
        artista  = ", ".join(artistas)
        etiqueta = None
        if slug_tema.upper().startswith("OP"):
            etiqueta = f'Opening: "{titulo}" por {artista}'
        elif slug_tema.upper().startswith("ED"):
            etiqueta = f'Ending: "{titulo}" por {artista}'
        if etiqueta:
            salida[slug_tema]     = etiqueta
            salida[slug_tema_raw] = etiqueta
    return salida

def nombre_archivo_seguro(name: str) -> str:
    s = str(name)
    s = re.sub(r'"(?=\w)', "\u201c", s)   # " antes de palabra → "
    s = re.sub(r'(?<=\w)"', "\u201d", s)  # " después de palabra → "
    s = s.replace(":", "꞉").replace("?", "？")
    s = re.sub(r'[<>/\\|*\x00-\x1F]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

@_reintento_http
def descargar_archivo(url: str, ruta_salida: str):
    """Descarga con streaming para no cargar archivos grandes en memoria."""
    with _http.stream("GET", url) as r:
        r.raise_for_status()
        Path(ruta_salida).write_bytes(r.read())

def _nombres_serie_iguales(a: str, b: str) -> bool:
    return (a or "").strip().casefold() == (b or "").strip().casefold()

def construir_cache_temas(slug: str, anime_json: dict, log) -> Path:
    """
    Descarga y convierte los audios de los temas de AnimeThemes.
    - Los archivos OGG/WAV se guardan en _THEMES_DIR/<slug>/
    - Los metadatos de cada tema se guardan en diskcache con TTL de 30 días
    Devuelve la ruta del directorio wav para este slug.
    """
    series_dir = _THEMES_DIR / nombre_archivo_seguro(slug)
    wav_dir    = series_dir / "wav"
    series_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(exist_ok=True)

    series_name = anime_json.get("name") or anime_json.get("slug") or "series"
    clave_serie = f"themes_meta:{slug}"
    meta_cached = _API_CACHE.get(clave_serie)

    # Si la serie cambió de nombre, limpiar archivos del directorio
    if meta_cached and not _nombres_serie_iguales(meta_cached.get("nombre_serie", ""), series_name):
        for p in series_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
        _API_CACHE.delete(clave_serie)
        meta_cached = None

    temas_meta: dict = (meta_cached or {}).get("temas", {})

    pendientes: List[Tuple[str, str]] = []
    temas = anime_json.get("animethemes") or []

    for tema in temas:
        base_slug = tema.get("slug") or ""
        cur_theme = (
            base_slug
            if (base_slug and base_slug[-1].isdigit())
            else (base_slug + "1" if base_slug else "")
        )
        if not cur_theme:
            continue

        audio_version       = 1
        links_audio_vistos: set = set()

        for entry in (tema.get("animethemeentries") or []):
            full_cur_theme = cur_theme
            if audio_version > 1:
                full_cur_theme += f"v{audio_version}"

            for video in (entry.get("videos") or []):
                if (video.get("overlap") or "None") != "None":
                    continue
                audio      = video.get("audio") or {}
                link       = audio.get("link")
                updated_at = audio.get("updated_at")
                if not link or link in links_audio_vistos:
                    continue

                links_audio_vistos.add(link)

                ogg_path = series_dir / f"{full_cur_theme}.ogg"
                wav_path = wav_dir / f"{full_cur_theme}.wav"
                rec      = temas_meta.get(full_cur_theme)

                if (
                    rec
                    and rec.get("updated_at") == updated_at
                    and ogg_path.exists()
                    and wav_path.exists()
                ):
                    log(f"  - {full_cur_theme}: caché OK")
                    audio_version += 1
                    break

                temas_meta[full_cur_theme] = {"updated_at": updated_at, "link": link}
                pendientes.append((full_cur_theme, link))
                audio_version += 1
                break

    if pendientes:
        log(f"• Descargando {len(pendientes)} temas desde AnimeThemes (paralelo)…")
    else:
        log("• Temas ya en caché.")

    def _bajar_y_convertir(item: Tuple[str, str]) -> Tuple[str, Optional[str]]:
        theme_name, link = item
        ogg_path = series_dir / f"{theme_name}.ogg"
        wav_path = wav_dir / f"{theme_name}.wav"
        try:
            descargar_archivo(link, str(ogg_path))
            extraer_audio_wav_mono_16k(str(ogg_path), str(wav_path))
            return theme_name, None
        except Exception as e:
            return theme_name, str(e)

    if pendientes:
        max_workers = min(4, len(pendientes))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futuros = {ex.submit(_bajar_y_convertir, item): item for item in pendientes}
            for fut in as_completed(futuros):
                theme_name, err = fut.result()
                if err:
                    log(f"  - ⚠️ {theme_name}: error al descargar/convertir: {err}")
                else:
                    log(f"  - ✅ {theme_name}: listo")

    # Persistir metadatos actualizados en diskcache
    _API_CACHE.set(
        clave_serie,
        {"nombre_serie": series_name, "temas": temas_meta},
        expire=_TTL_THEMES_DAYS * 86400,
    )
    return wav_dir

# ─────────────────────────────────────────────
#  MATCHING DE AUDIO — pipeline híbrido FFT → DTW
# ─────────────────────────────────────────────

# Pesos del score final (configurables)
_W_DTW = 0.70
_W_FFT = 0.30

# Parámetros de extracción de features
_SR_FEATURES  = 16000   # tasa de muestreo (ya usamos 16kHz)
_HOP_LENGTH   = 512     # hop para MFCC/chroma (~32ms a 16kHz)
_N_MFCC       = 20
_TOP_K_FFT     = 3       # cuántos candidatos pasan a DTW
_FFT_PRUNING_MIN = 0.08  # early pruning: si el mejor FFT score < esto, saltar DTW
_USE_CHROMA    = True    # False para deshabilitar chroma (más robusto con diálogo encima)
_CHROMA_WEIGHT = 0.8    # peso relativo de chroma vs MFCC (1.0 = igual peso)

# Ventana deslizante para búsqueda de OP/ED
_SLIDE_WIN_SEC  = 90    # duración de cada ventana de comparación (segundos)
_SLIDE_STEP_SEC = 15    # paso entre ventanas (segundos) — overlap del 83%
_SLIDE_OP_MAX   = 300   # cuántos segundos del inicio explorar para el OP
_SLIDE_ED_MAX   = 300   # cuántos segundos del final explorar para el ED

# ── helpers scipy opcionales ──────────────────

def _siguiente_potencia_de_2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _mejor_nfft(n: int) -> int:
    if _SCIPY_AVAILABLE:
        return _scipy_next_fast_len(n)
    return _siguiente_potencia_de_2(n)

def _rfft(x: np.ndarray, n: int) -> np.ndarray:
    if _SCIPY_AVAILABLE:
        return rfft(x, n=n, workers=-1)
    return np.fft.rfft(x, n=n)

def _irfft(x: np.ndarray, n: int) -> np.ndarray:
    if _SCIPY_AVAILABLE:
        return irfft(x, n=n, workers=-1)
    return np.fft.irfft(x, n=n)

# ── caché de features ─────────────────────────

def _clave_features(y: np.ndarray) -> str:
    """
    Clave basada en hash del array numpy crudo (determinístico).
    Hashear y.tobytes() en vez del archivo WAV evita variaciones
    por metadata/padding que ffmpeg puede cambiar entre runs.
    """
    sha = hashlib.sha256(y.tobytes()).hexdigest()[:16]
    chroma_flag = "c1" if _USE_CHROMA else "c0"
    return f"feat:{sha}:sr{_SR_FEATURES}:hop{_HOP_LENGTH}:mfcc{_N_MFCC}:{chroma_flag}"

def extraer_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extrae MFCC (20 coef) + chroma opcional (12 bandas).
    Cada feature se normaliza con librosa.util.normalize antes de
    apilar — evita sesgos por volumen residual entre versiones TV/BD.
    Devuelve una matriz (n_feat, T) de float32.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=_N_MFCC, hop_length=_HOP_LENGTH
    )
    mfcc = librosa.util.normalize(mfcc, axis=1)  # normalizar por fila

    if _USE_CHROMA:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=_HOP_LENGTH)
        chroma = librosa.util.normalize(chroma, axis=1) * _CHROMA_WEIGHT
        T      = min(mfcc.shape[1], chroma.shape[1])
        feat   = np.vstack([mfcc[:, :T], chroma[:, :T]])   # (32, T)
    else:
        feat = mfcc                                          # (20, T)

    return feat.astype(np.float32)

def obtener_features_con_cache(y: np.ndarray, sr: int) -> np.ndarray:
    """Devuelve features desde caché si existen, o las extrae y las guarda."""
    clave  = _clave_features(y)
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    feat = extraer_features(y, sr)
    _API_CACHE.set(clave, feat, expire=_TTL_THEMES_DAYS * 86400)
    return feat

# ── paso 1: FFT para top-K candidatos ────────

def _fft_score(
    y_ep:          np.ndarray,
    y_th:          np.ndarray,
    hz:            int,
    submuestreo:   int   = 32,
    porcion_theme: float = 0.90,
) -> Optional[Tuple[float, float, float]]:
    """
    Correlación cruzada FFT. Devuelve (inicio_seg, fin_seg, score_norm).
    Sin umbral mínimo — el filtrado lo hace el top-K, no un threshold fijo.
    """
    if submuestreo < 1:
        submuestreo = 1

    ep     = y_ep[::submuestreo].astype(np.float32, copy=False)
    th     = y_th[::submuestreo].astype(np.float32, copy=False)
    hz_sub = hz / submuestreo

    if len(th) < int(hz_sub * 5):
        return None

    ep = (ep - ep.mean()) / (ep.std() + 1e-8)
    th = (th - th.mean()) / (th.std() + 1e-8)

    th_len = max(int(len(th) * porcion_theme), int(len(th) * 0.5))
    th_use = th[:th_len]

    M  = len(th_use)
    N  = len(ep)
    if M >= N:
        return None

    silencio = int(5 * hz_sub)
    ep2      = np.concatenate([np.zeros(silencio, dtype=np.float32), ep])
    N2       = len(ep2)

    rev  = th_use[::-1]
    L    = N2 + M - 1
    nfft = _mejor_nfft(L)

    conv   = _irfft(_rfft(ep2, nfft) * _rfft(rev, nfft), nfft)[:L]
    valida = conv[M - 1 : M - 1 + (N2 - M + 1)]
    if valida.size == 0:
        return None

    pico       = float(valida.max())
    idx        = int(valida.argmax())
    # Normalización suave: mapea [0,∞) → [0,1) preservando diferencias relativas
    score_raw  = pico / float(M)
    score_norm = score_raw / (1.0 + score_raw)

    desfase_seg  = max(0.0, (idx - silencio) / hz_sub)
    dur_tema_seg = len(th) / hz_sub
    return float(desfase_seg), float(desfase_seg + dur_tema_seg), float(score_norm)

# ── paso 2: DTW sobre candidatos ─────────────

def _dtw_score(feat_ep: np.ndarray, feat_th: np.ndarray) -> float:
    """
    DTW con subseq=True (matching parcial) entre matrices de features.
    Costo normalizado por la longitud del path óptimo.
    Menor = mejor.
    """
    D, wp = librosa.sequence.dtw(
        X=feat_th,    # (n_feat, T_tema) — el patrón a buscar
        Y=feat_ep,    # (n_feat, T_ep)   — el episodio donde buscar
        metric="cosine",
        subseq=True,
    )
    return float(D[-1, wp[-1, 1]]) / max(1, len(wp))

def formatear_tiempo(t: float) -> str:
    total_ms = int(round(t * 1000))
    h,  rem  = divmod(total_ms, 3_600_000)
    m,  rem  = divmod(rem,      60_000)
    s,  ms   = divmod(rem,      1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# ─────────────────────────────────────────────
#  CHAPTERS XML
# ─────────────────────────────────────────────

def tiempo_mkv(t: float) -> str:
    total_ns = int(round(t * 1_000_000_000))
    h,  rem  = divmod(total_ns, 3_600_000_000_000)
    m,  rem  = divmod(rem,      60_000_000_000)
    s,  ns   = divmod(rem,      1_000_000_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ns:09d}"

def crear_chapters_xml(ch_list: List[Tuple[float, str]]) -> str:
    atomos = []
    for inicio, titulo in ch_list:
        atomos.append(f"""
      <ChapterAtom>
        <ChapterTimeStart>{tiempo_mkv(inicio)}</ChapterTimeStart>
        <ChapterDisplay>
          <ChapterString>{escape(titulo)}</ChapterString>
          <ChapterLanguage>und</ChapterLanguage>
        </ChapterDisplay>
      </ChapterAtom>""")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Chapters>\n"
        "  <EditionEntry>"
        + "".join(atomos)
        + "\n  </EditionEntry>\n"
        "</Chapters>\n"
    )

def guardar_chapters(ruta_salida: str, chapters: List[Tuple[float, str]]):
    chapters = sorted(
        {(float(t), str(n)) for (t, n) in chapters},
        key=lambda x: x[0],
    )
    Path(ruta_salida).write_text(crear_chapters_xml(chapters), encoding="utf-8")

def chapters_heuristicos(dur: float) -> List[Tuple[float, str]]:
    op_inicio = 60.0 if dur > 180.0 else 0.0
    ed_inicio = max(0.0, dur - 95.0)
    return [(0.0, "Prólogo"), (op_inicio, "Opening"), (ed_inicio, "Ending")]

# ─────────────────────────────────────────────
#  PARSING DE NOMBRES DE ARCHIVO (aniparse / anitopy / regex)
# ─────────────────────────────────────────────

# Ruido residual para el fallback regex y para score_titulo
_RE_RUIDO_TITULO = re.compile(
    r"(?i)\b(2160p|1080p|720p|480p|4k|8k"
    r"|10bit|10-bit|8bit|hi10p?"
    r"|x264|x265|hevc|av1|h\.?26[45]"
    r"|web[- ]?(?:dl|rip)|webrip|bdrip|blu[- ]?ray|bluray|dvd"
    r"|hdr10\+?|hdr|dolby\s*vision|\bdv\b|atmos"
    r"|aac\d*\.?\d*|flac|opus|eac3|ac3|ddp?\d*\.?\d*"
    r"|jpn|eng|spa|lat|msubs?|multisub|multi|dual[- ]?audio"
    r"|uncensored|censored|repack|proper|remux"
    r"|amzn|\bcr\b|\bnf\b|dsnp|adn)\b"
    r"|[\[\(\{][^\]\)\}]{0,90}[\]\)\}]"  # bloques entre brackets
)

def _score_titulo(title: str) -> int:
    """
    Evalúa qué tan limpio está un título candidato.
    Puntuación más alta = mejor candidato para consultar Jikan.
    Se usa para decidir qué resultado (aniparse vs anitopy) tiene el mejor título.
    """
    if not title:
        return -999
    score = 0
    if 3 <= len(title) <= 80:
        score += 2
    # Penalizar ruido técnico típico de releases
    if _RE_RUIDO_TITULO.search(title):
        score -= 2
    # Penalizar número suelto al final SOLO si no parece temporada explícita
    # (e.g. "Kingdom 5" puede ser legítimo — se maneja luego en el merge)
    if re.search(r"(?<!S)\s\d{1,2}$", title):
        score -= 1   # penalización leve, no definitiva
    # Penalizar hashes o números largos
    if re.search(r"\b[0-9A-Fa-f]{6,}\b", title):
        score -= 3
    return score


def _safe_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def _parse_con_aniparse(stem: str) -> Optional[dict]:
    if not _ANIPARSE_OK:
        return None
    try:
        result = _aniparse.parse(stem)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _parse_con_anitopy(stem: str) -> Optional[dict]:
    if not _ANITOPY_OK:
        return None
    try:
        result = _anitopy.parse(stem)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _parsed_dict_a_campos(d: dict) -> tuple:
    """Extrae (titulo, temporada, episodio) de un dict de aniparse/anitopy."""
    titulo    = (d.get("anime_title") or "").strip()
    temporada = _safe_int(d.get("anime_season"))
    episodio  = _safe_int(d.get("episode_number"))
    return titulo, temporada, episodio


def _fallback_regex(stem: str) -> "ParsedAnime":
    """
    Parser de último recurso basado en regex.
    Mantiene compatibilidad con nombres que las bibliotecas no manejen.
    """
    # Quitar bloques entre brackets (grupo, tags, hash)
    s = re.sub(r"[\[\(\{][^\]\)\}]{0,90}[\]\)\}]", " ", stem)
    # Quitar tag de release al final (e.g. "-SubsPlease")
    s = re.sub(r"-[A-Za-z0-9]+$", " ", s)

    # Temporada textual
    temporada: Optional[int] = None
    for pat, grp in [
        (r"\b(\d+)\s*(?:st|nd|rd|th)\s*season\b", 1),
        (r"\bseason[_\s\-]*(\d+)\b", 1),
        (r"(?:^|[\s._-])s(\d{1,2})(?:$|[\s._-])", 1),
    ]:
        m = re.search(pat, s, re.I)
        if m:
            temporada = _safe_int(m.group(grp))
            break

    # Episodio
    episodio: Optional[int] = None
    for pat, grps in [
        (r"(?i)\bS(\d{1,2})E(\d{1,3})(?:v\d+)?\b", (1, 2)),
        (r"(?i)\b(\d{1,2})x(\d{1,3})\b",             (1, 2)),
    ]:
        m = re.search(pat, s)
        if m:
            temporada = temporada or _safe_int(m.group(grps[0]))
            episodio  = _safe_int(m.group(grps[1]))
            break
    if episodio is None:
        m = re.search(r"(?i)\b(?:EP?|E)\s*(\d{1,3})(?:v\d+)?\b", s)
        if m:
            episodio = _safe_int(m.group(1))
    if episodio is None:
        m = re.search(r"-\s+(\d{1,3})(?:v\d+)?(?:\s|$|\[|\()", s)
        if m:
            ep = _safe_int(m.group(1))
            if ep and 1 <= ep <= 399:
                episodio = ep

    # Limpiar título: quitar episodio, ruido técnico, separadores
    titulo = s
    titulo = re.sub(r"(?i)\bS\d{1,2}E\d{1,3}(?:v\d+)?\b", " ", titulo)
    titulo = re.sub(r"(?i)\b(?:EP?|E)\d{1,3}(?:v\d+)?\b",   " ", titulo)
    titulo = re.sub(r"-\s*\d{1,3}(?:v\d+)?(?:\s|$)",          " ", titulo)
    titulo = _RE_RUIDO_TITULO.sub(" ", titulo)
    titulo = re.sub(r"[._]+", " ", titulo)
    titulo = re.sub(r"\s+", " ", titulo).strip(" -_.")

    logger.debug(f"[parsing] fallback regex para {stem!r} → título={titulo!r}")
    return ParsedAnime(titulo=titulo, temporada=temporada, episodio=episodio, fuente="fallback")


def parsear_nombre_archivo(ruta_video: str) -> "ParsedAnime":
    """
    Punto de entrada único para parsear nombres de archivo de anime.

    Estrategia:
      1. aniparse  (principal — mejor con nombres modernos)
      2. anitopy   (respaldo  — más probado en variedad)
      3. merge     de ambos si los dos producen resultado
      4. regex     fallback si las bibliotecas no están instaladas o fallan
    """
    stem = Path(ruta_video).stem

    a = _parse_con_aniparse(stem)
    b = _parse_con_anitopy(stem)

    if a is None and b is None:
        return _fallback_regex(stem)

    titulo_a, temp_a, ep_a = _parsed_dict_a_campos(a) if a else ("", None, None)
    titulo_b, temp_b, ep_b = _parsed_dict_a_campos(b) if b else ("", None, None)

    # Merge consciente de temporada: si una biblioteca detectó season pero la otra
    # dejó el número pegado al título (e.g. "Kingdom 5" cuando season=5), limpiarlo.
    temp_combinada = temp_a if temp_a is not None else temp_b
    if temp_combinada:
        titulo_a = re.sub(rf"\s{temp_combinada}$", "", titulo_a).strip()
        titulo_b = re.sub(rf"\s{temp_combinada}$", "", titulo_b).strip()

    # Elegir el mejor título por score de limpieza
    score_a = _score_titulo(titulo_a)
    score_b = _score_titulo(titulo_b)

    if score_b > score_a:
        titulo_elegido = titulo_b
        fuente         = "anitopy" if b and not a else "merge"
    else:
        titulo_elegido = titulo_a
        fuente         = "aniparse" if a and not b else "merge"

    # Si el título elegido sigue teniendo ruido (score < 1), caer a regex.
    # Umbral 1 en lugar de 0 para capturar falsos positivos como "Frieren 1080p".
    if _score_titulo(titulo_elegido) < 1:
        logger.debug(f"[parsing] título descartado ({titulo_elegido!r}), fallback a regex")
        return _fallback_regex(stem)

    # Episodio: primer valor no-None gana (aniparse tiene prioridad)
    temporada = temp_combinada
    episodio  = ep_a if ep_a is not None else ep_b

    resultado = ParsedAnime(
        titulo=titulo_elegido,
        temporada=temporada,
        episodio=episodio,
        fuente=fuente,
    )
    logger.debug(
        f"[parsing] {Path(ruta_video).name!r} → "
        f"aniparse={titulo_a!r} | anitopy={titulo_b!r} → "
        f"final={resultado.titulo!r} "
        f"(fuente={resultado.fuente}, T={resultado.temporada}, E={resultado.episodio})"
    )
    return resultado


# Wrappers de compatibilidad — mantienen la firma anterior intacta
# para no reescribir el ResolverWorker de golpe.

def quitar_sufijo_episodio(s: str) -> str:
    """Quita un sufijo ' - NN' de un título canónico (no de nombre de archivo)."""
    return re.sub(r"(?i)\s*-\s*\d{1,3}(?:v\d+)?\s*$", "", (s or "").strip())

def quitar_marcador_temporada(s: str) -> str:
    """Quita marcadores de temporada textual de un título canónico."""
    x = (s or "").strip()
    x = re.sub(r"(?i)\b(\d+)\s*(st|nd|rd|th)\s*season\b", "", x)
    x = re.sub(r"(?i)\bseason[_\s\-]*\d+\b", "", x)
    x = re.sub(r"\s+", " ", x).strip(" -_:")
    return x

def inferir_consulta_desde_nombre_archivo(ruta_video: str) -> str:
    return parsear_nombre_archivo(ruta_video).titulo

def _extraer_temporada_desde_slug_o_nombre(s: str) -> Optional[int]:
    if not s:
        return None
    x = s.casefold()
    for pat in [
        r"\b(\d+)(?:st|nd|rd|th)_season\b",
        r"\bseason[_\s\-]*(\d+)\b",
        r"(?:^|[_\-\s])s(\d+)(?:$|[_\-\s])",
        r"\b(\d+)(?:st|nd|rd|th)\b",
    ]:
        m = re.search(pat, x)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def _preferir_resultados_por_temporada(
    resultados: List[dict],
    temporada:  Optional[int],
) -> List[dict]:
    if not resultados or not temporada or temporada <= 1:
        return resultados

    def temporada_item(it: dict) -> Optional[int]:
        slug = it.get("slug") or ""
        name = (it.get("name") or "").strip()
        t    = _extraer_temporada_desde_slug_o_nombre(slug)
        return t if t is not None else _extraer_temporada_desde_slug_o_nombre(name)

    exactos, desconocidos, otros = [], [], []
    for it in resultados:
        t = temporada_item(it)
        if t is None:        desconocidos.append(it)
        elif t == temporada: exactos.append(it)
        else:                otros.append(it)

    if exactos:      return exactos + desconocidos + otros
    if desconocidos: return desconocidos + otros
    return sorted(resultados, key=lambda it: (
        abs(temporada_item(it) - temporada) if temporada_item(it) is not None else 999
    ))

# ─────────────────────────────────────────────
#  JIKAN / MAL
# ─────────────────────────────────────────────

@_reintento_http
def jikan_buscar_anime(q: str, limite: int = 10) -> List[dict]:
    clave  = f"jikan_search:{q.strip().casefold()}:{limite}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    r = _http.get(JIKAN_ANIME, params={"q": q, "limit": limite})
    r.raise_for_status()
    result = (r.json() or {}).get("data") or []
    _API_CACHE.set(clave, result, expire=_TTL_API_DAYS * 86400)
    return result

@_reintento_http
def jikan_relaciones(id_anime: int) -> List[dict]:
    clave  = f"jikan_rel:{id_anime}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    r = _http.get(JIKAN_REL.format(id=int(id_anime)))
    r.raise_for_status()
    result = (r.json() or {}).get("data") or []
    _API_CACHE.set(clave, result, expire=_TTL_API_DAYS * 86400)
    return result

def jikan_resolver_temporada_por_sequel(elemento_base: dict, temporada: int) -> dict:
    if not elemento_base or not temporada or temporada <= 1:
        return elemento_base

    actual = elemento_base
    for paso in range(temporada - 1):
        relaciones = jikan_relaciones(int(actual["mal_id"]))
        secuela    = None
        for rel in relaciones:
            if (rel.get("relation") or "").casefold() == "sequel":
                entries = rel.get("entry") or []
                if entries:
                    secuela = entries[0]
                break

        if not secuela:
            raise RuntimeError(
                f"Jikan: no encontré secuela en paso {paso + 1}/{temporada - 1} "
                f"(mal_id={actual.get('mal_id')}). "
                "La cadena de secuelas está incompleta en MAL."
            )

        id_secuela = int(secuela["mal_id"])

        @_reintento_http
        def _get_secuela():
            r = _http.get(f"{JIKAN_ANIME}/{id_secuela}")
            r.raise_for_status()
            return r.json()

        data = (_get_secuela() or {}).get("data")
        if not data:
            raise RuntimeError(f"Jikan: la secuela mal_id={id_secuela} no devolvió datos.")
        actual = data

    return actual

def jikan_titulos_desde_item(item: dict) -> List[str]:
    titulos = []
    for k in ("title", "title_english", "title_japanese"):
        t = item.get(k)
        if t:
            titulos.append(t)
    for t in item.get("titles") or []:
        tv = t.get("title")
        if tv:
            titulos.append(tv)
    for sinonimo in item.get("title_synonyms") or []:
        if sinonimo:
            titulos.append(sinonimo)

    vistos: set = set()
    salida      = []
    for t in titulos:
        tt  = str(t).strip()
        key = tt.casefold()
        if tt and key not in vistos:
            vistos.add(key)
            salida.append(tt)
    return salida

def _normalizar_titulo(s: str) -> str:
    s = (s or "").casefold()
    s = s.replace("'", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _aceptar_canon_sin_perder_tokens(base: str, canon: str) -> bool:
    bn   = _normalizar_titulo(base)
    cn   = _normalizar_titulo(canon)
    bt   = [t for t in bn.split() if len(t) >= 4]
    ct   = set(t for t in cn.split() if len(t) >= 4)
    stop = {
        "season", "part", "cour", "movie", "film", "tv", "ova", "ona",
        "the", "and", "of", "to", "in", "no", "na", "ga", "wo",
    }
    bt      = [t for t in bt if t not in stop]
    missing = [t for t in bt if t not in ct]
    return len(missing) == 0

def _ratio(a: str, b: str) -> float:
    # rapidfuzz devuelve 0–100, normalizamos a 0–1
    return _fuzz.ratio(a, b) / 100.0

def _jikan_text_score(q: str, item: dict) -> float:
    qn = _normalizar_titulo(q)
    if not qn:
        return 0.0
    titulos    = jikan_titulos_desde_item(item)
    cand_norms = [c for c in (_normalizar_titulo(t) for t in titulos if t) if c]
    if not cand_norms:
        return 0.0
    best_ratio = max(_ratio(qn, c) for c in cand_norms)
    qt         = set(qn.split())
    bonus      = max(len(qt & set(c.split())) / max(1, len(qt)) for c in cand_norms)
    return best_ratio * 0.75 + bonus * 0.25

def _jikan_rank(q: str, resultados: List[dict]) -> List[dict]:
    def mal_score(it):
        s = it.get("score")
        try:
            return float(s) if s is not None else 0.0
        except Exception:
            return 0.0
    return sorted(
        resultados,
        key=lambda it: (_jikan_text_score(q, it), mal_score(it)),
        reverse=True,
    )

def extraer_temporada_y_episodio_desde_nombre_archivo(
    ruta_video: str,
) -> Tuple[Optional[int], Optional[int]]:
    p = parsear_nombre_archivo(ruta_video)
    return p.temporada, p.episodio

def jikan_resolver_titulo(q: str) -> Tuple[str, Optional[dict], bool]:
    q = (q or "").strip()
    if not q:
        return q, None, False
    resultados = jikan_buscar_anime(q, limite=10)
    if not resultados:
        return q, None, False
    if len(resultados) == 1:
        it   = resultados[0]
        main = (it.get("title") or "").strip() or q
        return main, it, True
    ordenados = _jikan_rank(q, resultados)
    mejor     = ordenados[0]
    segundo   = ordenados[1] if len(ordenados) > 1 else None
    ts1       = _jikan_text_score(q, mejor)
    ts2       = _jikan_text_score(q, segundo) if segundo else 0.0
    confiable = ts1 >= 0.72 and (ts1 - ts2) >= 0.08
    main      = (mejor.get("title") or "").strip() or q
    return main, mejor, confiable

def animethemes_coincidencia_exacta_por_titulo(
    resultados: List[dict],
    titulo_objetivo: str,
) -> Optional[dict]:
    tgt = _normalizar_titulo(titulo_objetivo)
    for it in resultados or []:
        name = it.get("name") or it.get("titulo") or ""
        if _normalizar_titulo(name) == tgt:
            return it
    return None

def filtrar_por_token_obligatorio(consulta_base: str, resultados: List[dict]) -> List[dict]:
    tok = _normalizar_titulo(consulta_base).split()
    tok = [t for t in tok if len(t) >= 4]
    if not tok:
        return resultados
    salida = [
        it for it in resultados
        if all(
            t in _normalizar_titulo(it.get("name") or it.get("titulo") or "")
            for t in tok[:1]
        )
    ]
    return salida or resultados

# ─────────────────────────────────────────────
#  SALIDA / NAMING
# ─────────────────────────────────────────────

def construir_ruta_salida(
    video_path:       str,
    carpeta_salida:   str,
    crear_subcarpeta: bool,
    titulo_anime:     str,
    episodio:         int,
) -> str:
    vdir = str(Path(video_path).parent)
    base = carpeta_salida.strip() if carpeta_salida and carpeta_salida.strip() else vdir
    if crear_subcarpeta:
        base = str(Path(base) / "Chapters")
    Path(base).mkdir(parents=True, exist_ok=True)
    ep     = int(episodio) if episodio is not None else 0
    titulo = nombre_archivo_seguro(titulo_anime or "Anime")
    fname  = f"{titulo} - {ep:02d} [Chapters].xml"
    return str(Path(base) / fname)

# ─────────────────────────────────────────────
#  PICKERS (PyQt6)
# ─────────────────────────────────────────────

class DialogoSelectorTabla(QDialog):
    def __init__(
        self,
        ventana_padre,
        titulo:    str,
        subtitulo: str,
        columnas:  List[Tuple[str, int]],
        filas:     List[List[str]],
    ):
        super().__init__(ventana_padre)
        self.setWindowTitle(titulo)
        self.setModal(True)
        self.resize(980, 420)

        lay = QVBoxLayout()
        lbl = QLabel(subtitulo)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(len(columnas))
        self.table.setRowCount(len(filas))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        for j, (name, w) in enumerate(columnas):
            self.table.setHorizontalHeaderItem(j, QTableWidgetItem(name))
            self.table.setColumnWidth(j, w)

        for i, fila in enumerate(filas):
            for j, val in enumerate(fila):
                self.table.setItem(i, j, QTableWidgetItem(val))

        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        lay.addWidget(self.table)

        btnrow     = QHBoxLayout()
        btn_ok     = QPushButton("Usar seleccionado")
        btn_cancel = QPushButton("Cancelar")
        btnrow.addWidget(btn_ok)
        btnrow.addStretch(1)
        btnrow.addWidget(btn_cancel)
        lay.addLayout(btnrow)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        self.table.cellDoubleClicked.connect(lambda *_: self.accept())

        self.setLayout(lay)
        if filas:
            self.table.selectRow(0)

    def indice_seleccionado(self) -> Optional[int]:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        return int(sel[0].row())

# ─────────────────────────────────────────────
#  WORKERS (QThread)
# ─────────────────────────────────────────────

class _BaseWorker(QThread):
    """Base compartida para todos los workers — provee _log con nivel automático."""
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)

    def _log(self, s: str):
        self.log.emit(s)
        if s.startswith("  - ⚠️") or s.startswith("⚠️"):
            logger.warning(s)
        elif s.startswith("❌"):
            logger.error(s)
        else:
            logger.info(s)


class ResolverWorker(_BaseWorker):
    need_pick  = pyqtSignal(object)
    resolved   = pyqtSignal(object)
    failed     = pyqtSignal(str)

    def __init__(self, ventana, params: ParametrosTrabajo, interactivo: bool = True):
        super().__init__(ventana)
        self.ventana     = ventana
        self.params      = params
        self.interactivo = interactivo

        self._mx     = QMutex()
        self._cv     = QWaitCondition()
        self._cancel = False

        self._pick_index: Optional[int] = None
        self._pick_ready: bool          = False

    def cancelar(self):
        self._mx.lock()
        self._cancel = True
        self._cv.wakeAll()
        self._mx.unlock()

    def entregar_pick(self, idx: Optional[int]):
        self._mx.lock()
        self._pick_index = idx
        self._pick_ready = True
        self._cv.wakeAll()
        self._mx.unlock()

    def _wait_pick(self) -> Optional[int]:
        self._mx.lock()
        while not self._pick_ready and not self._cancel:
            self._cv.wait(self._mx)
        idx    = self._pick_index
        cancel = self._cancel
        self._mx.unlock()
        return None if cancel else idx

    def _pedir_pick(self, req: PickRequest) -> Optional[int]:
        self._mx.lock()
        self._pick_index = None
        self._pick_ready = False
        self._mx.unlock()
        self.need_pick.emit(req)
        return self._wait_pick()

    def run(self):
        try:
            p     = self.params
            video = p.video

            asegurar_ffmpeg()

            temporada, ep = extraer_temporada_y_episodio_desde_nombre_archivo(video)
            episodio      = int(ep or 0)
            temporada     = 1 if temporada is None else int(temporada)

            self._log("• ResolverWorker: analizando nombre de archivo…")
            self.progress.emit(5)
            log_clv(self._log, "parsed", temporada=temporada, episodio=episodio)

            override = (p.search_override or "").strip()
            if override:
                consulta_base = override
                self._log("• Usando nombre de búsqueda (anulación) desde interfaz…")
                log_clv(self._log, "override", q=consulta_base)
            else:
                consulta_base  = inferir_consulta_desde_nombre_archivo(video)
                consulta_base  = quitar_sufijo_episodio(consulta_base)
                consulta_jikan = quitar_marcador_temporada(consulta_base)

                self._log("• Jikan (base)…")
                log_clv(self._log, "jikan_query", q=consulta_jikan, from_base=consulta_base)
                canon_base, picked_base, ok_base = jikan_resolver_titulo(consulta_jikan)
                log_clv(
                    self._log, "jikan_base_result",
                    canon=canon_base, ok=ok_base,
                    mal_id=(picked_base or {}).get("mal_id"),
                )

                if temporada >= 2 and picked_base:
                    try:
                        self._log("• Jikan (resolviendo temporada secuela)…")
                        picked_season = jikan_resolver_temporada_por_sequel(picked_base, temporada)
                        canon_season  = (
                            (picked_season.get("title") or "").strip()
                            or canon_base or consulta_base
                        )
                        if canon_season and _aceptar_canon_sin_perder_tokens(consulta_base, canon_season):
                            consulta_base = canon_season
                        else:
                            self._log(
                                f"  - ⚠️ Ignorando canon de temporada por recorte: "
                                f"{consulta_base!r} → {canon_season!r}"
                            )
                    except Exception as e:
                        self._log(f"  - ⚠️ Secuela falló: {e}. Usando canon base si está disponible.")
                        if ok_base and canon_base and _aceptar_canon_sin_perder_tokens(consulta_base, canon_base):
                            consulta_base = canon_base
                else:
                    if ok_base and canon_base and _aceptar_canon_sin_perder_tokens(consulta_base, canon_base):
                        consulta_base = canon_base

            if not p.usar_exacto:
                p.slug         = ""
                p.episodio     = episodio
                p.titulo_usado = consulta_base
                self.progress.emit(30)
                self.resolved.emit(p)
                return

            slug, titulo_usado = self._resolver_slug_con_picker(consulta_base, temporada)
            self.progress.emit(30)
            p.slug         = slug
            p.titulo_usado = titulo_usado
            p.episodio     = episodio
            self.resolved.emit(p)

        except Exception as e:
            self.failed.emit(str(e))

    def _resolver_slug_con_picker(self, consulta: str, temporada: int) -> Tuple[str, str]:
        self._log("• AnimeThemes (resolviendo slug)…")
        resultados = buscar_anime_en_animethemes(consulta)
        resultados = filtrar_por_token_obligatorio(consulta, resultados)
        raw        = list(resultados)
        resultados = _preferir_resultados_por_temporada(resultados, temporada)

        self._log(
            f"  - Resultados: crudos={len(raw)} → priorizados={len(resultados)} "
            f"(temporada={temporada})"
        )

        if not resultados:
            return self._resolver_via_jikan_con_picker(consulta)

        if len(resultados) == 1:
            it   = resultados[0]
            slug = (it.get("slug") or "").strip()
            name = (it.get("name") or consulta).strip()
            if not slug:
                raise RuntimeError("AnimeThemes: resultado sin slug.")
            return slug, name

        exacto = animethemes_coincidencia_exacta_por_titulo(resultados, consulta)
        if exacto:
            slug = (exacto.get("slug") or "").strip()
            name = (exacto.get("name") or consulta).strip()
            if not slug:
                raise RuntimeError("AnimeThemes: resultado sin slug.")
            return slug, name

        if not self.interactivo:
            raise RuntimeError("AnimeThemes ambiguo (modo no interactivo)")

        filas = [
            [
                it.get("name") or "(sin nombre)",
                str(it.get("year") or ""),
                str(it.get("season") or ""),
                it.get("slug") or "",
            ]
            for it in resultados
        ]
        req = PickRequest(
            kind="animethemes",
            titulo="Selecciona el anime correcto (AnimeThemes)",
            subtitulo=(
                f"AnimeThemes devolvió múltiples resultados para: {consulta!r}. "
                "Elige el correcto:"
            ),
            columnas=[("Nombre", 520), ("Año", 70), ("Temporada", 110), ("Slug", 260)],
            filas=filas,
            payload=resultados,
        )
        idx = self._pedir_pick(req)
        if idx is None:
            raise RuntimeError("Selección cancelada.")
        elegido = resultados[int(idx)]
        slug    = (elegido.get("slug") or "").strip()
        name    = (elegido.get("name") or consulta).strip()
        if not slug:
            raise RuntimeError("AnimeThemes: seleccionado sin slug.")
        return slug, name

    def _resolver_via_jikan_con_picker(self, consulta: str) -> Tuple[str, str]:
        self._log("• Respaldo: Jikan…")
        resultados = jikan_buscar_anime(consulta, limite=10)
        if not resultados:
            raise RuntimeError("Jikan no devolvió resultados.")

        elegido = None
        if len(resultados) == 1:
            elegido = resultados[0]
        else:
            if not self.interactivo:
                raise RuntimeError("Jikan ambiguo (modo no interactivo)")

            filas = [
                [
                    el.get("title") or "(sin título)",
                    el.get("type") or "",
                    str(el.get("year") or ""),
                    "" if el.get("episodes") is None else str(el["episodes"]),
                    "" if el.get("score") is None else f"{float(el['score']):.2f}",
                ]
                for el in resultados
            ]
            req = PickRequest(
                kind="jikan",
                titulo="Selecciona el anime correcto (Jikan/MAL)",
                subtitulo=(
                    f"Se encontraron múltiples resultados para: {consulta!r}. "
                    "Elige el correcto:"
                ),
                columnas=[("Título", 620), ("Tipo", 80), ("Año", 70), ("Eps", 60), ("Puntaje", 80)],
                filas=filas,
                payload=resultados,
            )
            idx = self._pedir_pick(req)
            if idx is None:
                raise RuntimeError("Selección cancelada.")
            elegido = resultados[int(idx)]

        for cand in jikan_titulos_desde_item(elegido):
            at = buscar_anime_en_animethemes(cand)
            if not at:
                continue
            if len(at) == 1:
                slug = (at[0].get("slug") or "").strip()
                name = (at[0].get("name") or cand).strip()
                if slug:
                    return slug, name

            if self.interactivo:
                filas = [
                    [
                        it.get("name") or "(sin nombre)",
                        str(it.get("year") or ""),
                        str(it.get("season") or ""),
                        it.get("slug") or "",
                    ]
                    for it in at
                ]
                req = PickRequest(
                    kind="animethemes",
                    titulo="Selecciona el anime correcto (AnimeThemes)",
                    subtitulo=(
                        f"AnimeThemes devolvió múltiples resultados para: {cand!r}. "
                        "Elige el correcto:"
                    ),
                    columnas=[("Nombre", 520), ("Año", 70), ("Temporada", 110), ("Slug", 260)],
                    filas=filas,
                    payload=at,
                )
                idx = self._pedir_pick(req)
                if idx is None:
                    raise RuntimeError("Selección cancelada.")
                it   = at[int(idx)]
                slug = (it.get("slug") or "").strip()
                name = (it.get("name") or cand).strip()
                if slug:
                    return slug, name

        raise RuntimeError("No encontré la serie en AnimeThemes vía Jikan.")


class ChapterizerWorker(_BaseWorker):
    terminado = pyqtSignal(str)
    fallo     = pyqtSignal(str)

    def __init__(self, ventana, params: ParametrosTrabajo):
        super().__init__(ventana)
        self.ventana = ventana
        self.params  = params

    def run(self):
        try:
            p     = self.params
            video = p.video

            asegurar_ffmpeg()

            self._log("• Analizando video…")
            self.progress.emit(35)
            dur = duracion_con_ffprobe(video)
            self._log(f"  - Duración: {dur:.2f}s")

            slug         = (p.slug or "").strip()
            titulo_usado = (p.titulo_usado or "").strip() or "Anime"
            episodio     = int(p.episodio or 0)

            ruta_salida = construir_ruta_salida(
                video_path=video,
                carpeta_salida=p.carpeta_salida,
                crear_subcarpeta=p.crear_subcarpeta,
                titulo_anime=titulo_usado,
                episodio=episodio,
            )
            self._log(f"• Archivo de salida: {ruta_salida}")

            if not p.usar_exacto:
                chapters = chapters_heuristicos(dur)
                guardar_chapters(ruta_salida, chapters)
                self.progress.emit(100)
                self._log(f"✅ Completado (heurístico): {ruta_salida}")
                self.terminado.emit(ruta_salida)
                return

            if not slug:
                raise RuntimeError(
                    "Slug vacío. La serie no fue resuelta en el hilo principal "
                    "(no se puede hacer coincidencia exacta sin AnimeThemes)."
                )

            series_dir = _THEMES_DIR / nombre_archivo_seguro(slug)

            self._log("• Obteniendo datos de AnimeThemes…")
            self.progress.emit(40)
            anime_json         = obtener_anime_de_animethemes(slug)
            mapa_titulos_temas = construir_mapa_mostrar_temas(anime_json)
            self._log(f"  - Serie: {anime_json.get('name') or slug}")
            self._log(f"  - Slug:  {slug}")
            self._log(f"  - Caché: {series_dir}")

            wav_dir = construir_cache_temas(slug, anime_json, self._log)

            # Precargar todos los WAVs de temas UNA sola vez — evita re-leer disco
            # en cada ventana del sliding window (potencialmente 20+ lecturas por tema).
            # También se resamplea aquí si es necesario (edge case) y se precalcula
            # frames_t para no repetirlo en cada llamada a _buscar_con_ventana.
            _SR_TARGET = _SR_FEATURES  # 16 000 Hz — mismo hz que el audio del episodio
            self._log("• Precargando audio de temas…")
            wavs_temas: List[TemaAudio] = []
            for ruta in sorted(wav_dir.glob("*.wav")):
                if not ruta.stem.upper().startswith(("OP", "ED")):
                    continue
                try:
                    y_th, hz_th = leer_pcm16_mono_wav(str(ruta))
                    if hz_th != _SR_TARGET:
                        self._log(f"  - ⚠️ {ruta.stem}: resampleando {hz_th}Hz → {_SR_TARGET}Hz…")
                        razon     = _SR_TARGET / hz_th
                        nuevo_len = int(len(y_th) * razon)
                        x_orig    = np.linspace(0, len(y_th) - 1, len(y_th))
                        x_nuevo   = np.linspace(0, len(y_th) - 1, nuevo_len)
                        y_th      = np.interp(x_nuevo, x_orig, y_th).astype(np.float32)
                        hz_th     = _SR_TARGET
                    feat_th = obtener_features_con_cache(y_th, hz_th)
                    wavs_temas.append(TemaAudio(
                        nombre=ruta.stem,
                        audio=y_th,
                        hz=hz_th,
                        frames=int(len(y_th) / _HOP_LENGTH),
                        features=feat_th,
                    ))
                except Exception as e:
                    self._log(f"  - ⚠️ {ruta.stem}: no se pudo cargar ({e}), omitido")

            if not wavs_temas:
                raise RuntimeError(
                    "No encontré WAVs de OP/ED en caché. "
                    "(¿AnimeThemes no trae audios?)"
                )

            self._log("• Búsqueda con ventana deslizante…")
            self.progress.emit(55)

            # ── Extraer audio de las zonas UNA sola vez ──────────────────
            op_zona_inicio = 0.0
            op_zona_fin    = min(_SLIDE_OP_MAX, dur * 0.6)
            ed_zona_inicio = max(0.0, dur - _SLIDE_ED_MAX)
            ed_zona_fin    = dur

            with tempfile.TemporaryDirectory() as dir_tmp:
                tmp = Path(dir_tmp)

                op_wav = str(tmp / "zona_op.wav")
                ed_wav = str(tmp / "zona_ed.wav")

                self._log(
                    f"  Extrayendo zona OP ({op_zona_inicio:.0f}s→{op_zona_fin:.0f}s) "
                    f"y ED ({ed_zona_inicio:.0f}s→{ed_zona_fin:.0f}s)…"
                )
                extraer_audio_wav_mono_16k(
                    video, op_wav,
                    ss=op_zona_inicio,
                    duracion=op_zona_fin - op_zona_inicio,
                )
                extraer_audio_wav_mono_16k(
                    video, ed_wav,
                    ss=ed_zona_inicio,
                    duracion=ed_zona_fin - ed_zona_inicio,
                )

                y_op, hz_op = leer_pcm16_mono_wav(op_wav)
                y_ed, hz_ed = leer_pcm16_mono_wav(ed_wav)

            # ── Extraer features globales de cada zona ────────────────────
            self._log("  Extrayendo features OP y ED…")
            feat_op = obtener_features_con_cache(y_op, hz_op)
            feat_ed = obtener_features_con_cache(y_ed, hz_ed)

            mejor_op = self._buscar_con_ventana(
                y_zona=y_op, feat_zona=feat_op, hz=hz_op,
                wavs_temas=wavs_temas, objetivo="OP",
                zona_offset=op_zona_inicio,
                zona_dur=op_zona_fin - op_zona_inicio,
                params=p,
            )
            self.progress.emit(82)

            mejor_ed_raw = self._buscar_con_ventana(
                y_zona=y_ed, feat_zona=feat_ed, hz=hz_ed,
                wavs_temas=wavs_temas, objetivo="ED",
                zona_offset=ed_zona_inicio,
                zona_dur=ed_zona_fin - ed_zona_inicio,
                params=p,
            )
            self.progress.emit(90)

            # Con ventana deslizante los tiempos ya son absolutos del episodio
            mejor_ed: Optional[ResultadoCoincidencia] = mejor_ed_raw

            PRE_OP  = "Introducción"
            EPISODE = "Episodio"
            POST_ED = "Conclusión"

            if not mejor_op and not mejor_ed:
                self._log("⚠️ No pude coincidir con OP/ED. Usando modo heurístico.")
                chapters = chapters_heuristicos(dur)
            else:
                marcas_tiempo: List[float] = []
                if mejor_op:
                    marcas_tiempo.extend([mejor_op.inicio, mejor_op.fin])
                if mejor_ed:
                    marcas_tiempo.extend([mejor_ed.inicio, mejor_ed.fin])
                marcas_tiempo = sorted(marcas_tiempo)

                def cerca_del_inicio(t: float) -> bool: return t < 4.0
                def cerca_del_final(t: float)  -> bool: return t > dur - 4.0

                ajusta_inicio = cerca_del_inicio(marcas_tiempo[0]) if marcas_tiempo else False
                ajusta_final  = cerca_del_final(marcas_tiempo[-1])  if marcas_tiempo else False
                solo_ed       = bool(marcas_tiempo and marcas_tiempo[0] > (dur / 2.0))

                titulo_op = (
                    mapa_titulos_temas.get(mejor_op.nombre_tema)
                    or f"Opening ({mejor_op.nombre_tema})"
                ) if mejor_op else "Opening"

                titulo_ed = (
                    mapa_titulos_temas.get(mejor_ed.nombre_tema)
                    or f"Ending ({mejor_ed.nombre_tema})"
                ) if mejor_ed else "Ending"

                chapters: List[Tuple[float, str]] = [(0.0, PRE_OP)]

                if marcas_tiempo:
                    if ajusta_inicio and not solo_ed:
                        chapters[0] = (0.0, titulo_op)
                        chapters.append((marcas_tiempo[1], EPISODE))
                        if len(marcas_tiempo) == 4:
                            chapters.append((marcas_tiempo[2], titulo_ed))
                            if not ajusta_final:
                                chapters.append((marcas_tiempo[3], POST_ED))
                    elif solo_ed:
                        chapters[0] = (0.0, EPISODE)
                        chapters.append((marcas_tiempo[0], titulo_ed))
                        if not ajusta_final and len(marcas_tiempo) >= 2:
                            chapters.append((marcas_tiempo[1], POST_ED))
                    else:
                        chapters[0] = (0.0, PRE_OP)
                        chapters.append((marcas_tiempo[0], titulo_op))
                        chapters.append((marcas_tiempo[1], EPISODE))
                        if len(marcas_tiempo) == 4:
                            chapters.append((marcas_tiempo[2], titulo_ed))
                            if not ajusta_final:
                                chapters.append((marcas_tiempo[3], POST_ED))

            guardar_chapters(ruta_salida, chapters)
            self.progress.emit(100)
            self._log(f"✅ Listo: {ruta_salida}")
            self.terminado.emit(ruta_salida)

        except Exception as e:
            self.fallo.emit(str(e))

    def _buscar_con_ventana(
        self,
        y_zona:      np.ndarray,         # PCM completo de la zona (samples)
        feat_zona:   np.ndarray,         # features globales (n_feat, T_zona)
        hz:          int,
        wavs_temas:  List[TemaAudio],
        objetivo:    str,
        zona_offset: float,              # segundos absolutos del inicio de la zona
        zona_dur:    float,              # duración de la zona en segundos
        params,
    ) -> Optional[ResultadoCoincidencia]:
        """
        Ventana deslizante sobre la zona indicada — sin llamadas a ffmpeg.
        El audio PCM y las features ya están en memoria; cada ventana es
        un slice de arrays, lo que elimina ~40 procesos ffmpeg por episodio.

        Conversiones de unidades:
          segundos → samples  : s * hz
          segundos → frames   : int(s * hz / _HOP_LENGTH)
          frames   → segundos : f * _HOP_LENGTH / hz
        """
        mejor:     Optional[ResultadoCoincidencia] = None
        zona_dur_s = zona_dur
        n_ventanas = max(1, int((zona_dur_s - _SLIDE_WIN_SEC) / _SLIDE_STEP_SEC) + 1)

        self._log(
            f"  [{objetivo}] Ventana deslizante: zona {zona_offset:.0f}s→"
            f"{zona_offset + zona_dur_s:.0f}s "
            f"| win={_SLIDE_WIN_SEC}s step={_SLIDE_STEP_SEC}s ({n_ventanas} ventanas)"
        )

        win_samples  = int(_SLIDE_WIN_SEC * hz)
        step_samples = int(_SLIDE_STEP_SEC * hz)

        # Mínimo de frames: mayor entre 8s absolutos y 25% del tema más corto.
        # 8s es más flexible que 10s para OPs/EDs cortos o recortados.
        # 25% del tema es suficiente para que DTW tenga contexto real sin ser tan
        # restrictivo como el 30% anterior.
        min_frames_absoluto = int(8 * hz / _HOP_LENGTH)
        frames_tema_min = min_frames_absoluto
        for tema in wavs_temas:
            if not tema.nombre.upper().startswith(objetivo):
                continue
            frames_tema_min = min(frames_tema_min, int(0.25 * tema.frames))
        min_frames_win = max(min_frames_absoluto, frames_tema_min)

        for paso, s_inicio_raw in enumerate(range(0, int(zona_dur_s * hz) - win_samples + 1, step_samples)):

            # ── Coherencia frame↔sample ───────────────────────────────────
            f_inicio = int(round(s_inicio_raw / _HOP_LENGTH))
            s_inicio = f_inicio * _HOP_LENGTH

            # ── Clamps ────────────────────────────────────────────────────
            f_fin = min(f_inicio + int(win_samples / _HOP_LENGTH), feat_zona.shape[1])
            s_fin = min(s_inicio + win_samples,                    len(y_zona))

            # ── Validar ventana no vacía ──────────────────────────────────
            # Puede ocurrir si f_inicio >= feat_zona.shape[1] tras el clamp.
            if f_fin <= f_inicio or s_fin <= s_inicio:
                continue

            # ── Filtro de ventana mínima ──────────────────────────────────
            if (f_fin - f_inicio) < min_frames_win:
                continue

            # ── Slices en memoria — sin ffmpeg ────────────────────────────
            y_win    = y_zona[s_inicio:s_fin]
            feat_win = feat_zona[:, f_inicio:f_fin]

            # Offset absoluto de esta ventana en el episodio
            ss_abs = zona_offset + s_inicio / hz

            res = self._coincidencia_con_features(
                y_win=y_win,
                feat_win=feat_win,
                hz=hz,
                wavs_temas=wavs_temas,
                objetivo=objetivo,
                params=params,
            )

            if res is not None:
                abs_res = ResultadoCoincidencia(
                    nombre_tema=res.nombre_tema,
                    inicio=res.inicio + ss_abs,
                    fin=res.fin   + ss_abs,
                    puntuacion=res.puntuacion,
                )
                self._log(
                    f"    ✓ win{paso} ({ss_abs:.0f}s): {abs_res.nombre_tema} "
                    f"{formatear_tiempo(abs_res.inicio)}→{formatear_tiempo(abs_res.fin)} "
                    f"score={abs_res.puntuacion:.3f}"
                )
                if mejor is None or abs_res.puntuacion > mejor.puntuacion:
                    mejor = abs_res

        if mejor:
            self._log(
                f"  ✓ [{objetivo}] Mejor global: {mejor.nombre_tema} "
                f"{formatear_tiempo(mejor.inicio)}→{formatear_tiempo(mejor.fin)} "
                f"(score={mejor.puntuacion:.3f})"
            )
        else:
            self._log(f"  ✗ [{objetivo}] Sin coincidencia en ninguna ventana.")

        return mejor

    def _coincidencia_con_features(
        self,
        y_win:      np.ndarray,          # PCM de la ventana (para FFT)
        feat_win:   np.ndarray,          # features de la ventana (para DTW)
        hz:         int,
        wavs_temas: List[TemaAudio],
        objetivo:   str,
        params,
    ) -> Optional[ResultadoCoincidencia]:
        """
        Pipeline FFT→DTW para una sola ventana.
        Separa la lógica de matching del slicing para mantener claridad
        de unidades: y_win son samples, feat_win son frames.
        Los audios ya vienen precargados, resampleados y con frames_t calculado.
        """
        candidatos_fft: List[CandidatoFFT] = []

        for tema in wavs_temas:
            if not tema.nombre.upper().startswith(objetivo):
                continue

            res_fft = _fft_score(
                y_win, tema.audio, hz,
                submuestreo=params.submuestreo,
                porcion_theme=params.porcion_theme,
            )
            if res_fft is None:
                continue

            inicio_fft, fin_fft, fft_s = res_fft
            candidatos_fft.append(CandidatoFFT(
                tema=tema,
                inicio=inicio_fft,
                fin=fin_fft,
                score_fft=fft_s,
            ))

        if not candidatos_fft:
            return None

        candidatos_fft.sort(key=lambda c: c.score_fft, reverse=True)
        candidatos_top = candidatos_fft[:_TOP_K_FFT]

        # ── Early pruning con threshold spread-aware ─────────────────────
        # En lugar de solo el percentil 25, usamos p25 + 0.5 * spread (IQR).
        # Esto detecta si hay un candidato que destaca del resto:
        #   - Si todos los scores son similares y bajos → spread pequeño,
        #     threshold sube y filtra la ventana (todos malos por igual).
        #   - Si un candidato destaca → spread grande, threshold no lo bloquea.
        # Más robusto que percentil fijo ante audios con distintos niveles de señal.
        fft_scores   = [c.score_fft for c in candidatos_top]
        p25          = float(np.percentile(fft_scores, 25))
        p75          = float(np.percentile(fft_scores, 75))
        spread       = p75 - p25
        thr_dinamico = max(_FFT_PRUNING_MIN, p25 + 0.5 * spread)
        mejor_fft_s  = candidatos_top[0].score_fft

        logger.debug(
            f"FFT max={mejor_fft_s:.3f} p25={p25:.3f} p75={p75:.3f} "
            f"spread={spread:.3f} threshold={thr_dinamico:.3f}"
        )

        if mejor_fft_s < thr_dinamico:
            return None

        self._log(
            f"  → Top-{len(candidatos_top)} candidatos FFT: "
            + ", ".join(f"{c.tema.nombre}({c.score_fft:.3f})" for c in candidatos_top)
        )

        # DTW usa el slice de features ya calculado — sin re-extraer
        mejor: Optional[ResultadoCoincidencia] = None
        mejor_score = -1.0

        for cand in candidatos_top:
            try:
                dtw_costo   = _dtw_score(feat_win, cand.tema.features)
                dtw_s       = max(0.0, 1.0 - dtw_costo / 50.0)
                score_final = _W_DTW * dtw_s + _W_FFT * cand.score_fft

                self._log(
                    f"  - {cand.tema.nombre}: DTW={dtw_costo:.2f} dtw_s={dtw_s:.3f} "
                    f"fft_s={cand.score_fft:.3f} → score={score_final:.3f}"
                )

                if score_final < params.puntuacion_minima:
                    self._log(f"    ↳ descartado (score {score_final:.3f} < umbral {params.puntuacion_minima})")
                    continue

                if score_final > mejor_score:
                    mejor_score = score_final
                    mejor       = ResultadoCoincidencia(
                        nombre_tema=cand.tema.nombre,
                        inicio=cand.inicio,
                        fin=cand.fin,
                        puntuacion=score_final,
                    )

            except Exception as e:
                self._log(f"  - ⚠️ {cand.tema.nombre}: error en DTW ({e}), usando FFT como respaldo")
                if cand.score_fft >= params.puntuacion_minima and cand.score_fft > mejor_score:
                    mejor_score = cand.score_fft
                    mejor       = ResultadoCoincidencia(
                        nombre_tema=cand.tema.nombre,
                        inicio=cand.inicio,
                        fin=cand.fin,
                        puntuacion=cand.score_fft,
                    )

        if mejor:
            self._log(
                f"  ✓ Mejor: {mejor.nombre_tema} "
                f"{formatear_tiempo(mejor.inicio)}→{formatear_tiempo(mejor.fin)} "
                f"(score={mejor.puntuacion:.3f})"
            )

        return mejor


# ─────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #de765d;
    padding: 8px 0px;
}
QLabel#section {
    font-size: 11px;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLineEdit {
    background-color: #313131;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #d4d4d4;
}
QLineEdit:focus  { border: 1px solid #de765d; }
QLineEdit:read-only { color: #aaaaaa; }
QPushButton#browse {
    background-color: #313131;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 14px;
    color: #d4d4d4;
    min-width: 72px;
}
QPushButton#browse:hover    { background-color: #3a3a3a; border-color: #de765d; }
QPushButton#browse:disabled { color: #555555; border-color: #2a2a2a; }
QPushButton#run {
    background-color: #de765d;
    border: none;
    border-radius: 8px;
    padding: 10px 30px;
    color: #1e1e1e;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#run:hover    { background-color: #e88b74; }
QPushButton#run:disabled { background-color: #333333; color: #555555; }
QProgressBar {
    background-color: #313131;
    border: none;
    border-radius: 5px;
    height: 10px;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #de765d, stop:1 #c4923a);
    border-radius: 5px;
}
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    color: #888888;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #888888;
}
QCheckBox { color: #aaaaaa; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #444444;
    border-radius: 3px;
    background-color: #1e1e1e;
}
QCheckBox::indicator:checked { background-color: #de765d; border-color: #de765d; }
QPlainTextEdit {
    background-color: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 8px;
    color: #a6e3a1;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QFrame#separator { background-color: #2a2a2a; max-height: 1px; }
QTableWidget {
    background-color: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #d4d4d4;
    gridline-color: #2a2a2a;
}
QHeaderView::section {
    background-color: #1a1a1a;
    color: #888888;
    border: none;
    padding: 4px 8px;
    font-size: 11px;
    text-transform: uppercase;
}
"""


class FieldRow(QWidget):
    def __init__(
        self,
        label:       str,
        btn_text:    str  = "Buscar",
        read_only:   bool = False,
        placeholder: str  = "",
    ):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setObjectName("section")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.entry = QLineEdit()
        self.entry.setReadOnly(read_only)
        if placeholder:
            self.entry.setPlaceholderText(placeholder)
        row.addWidget(self.entry)

        self.btn = QPushButton(btn_text)
        self.btn.setObjectName("browse")
        row.addWidget(self.btn)

        layout.addLayout(row)

    def get(self) -> str:
        return self.entry.text().strip()

    def set(self, val: str):
        self.entry.setText(val)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChapteriZen")
        self.setMinimumWidth(900)
        self._worker:   Optional[ChapterizerWorker] = None
        self._resolver: Optional[ResolverWorker]    = None
        self._construir_interfaz()
        self.setStyleSheet(STYLE)

    def _construir_interfaz(self):
        from PyQt6.QtCore import Qt

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(14)

        title = QLabel("🎞️ ChapteriZen")
        title.setObjectName("title")
        root.addWidget(title)

        sep = QFrame()
        sep.setObjectName("separator")
        root.addWidget(sep)

        self.row_video = FieldRow(
            "Video", btn_text="Buscar", read_only=True,
            placeholder="Selecciona el archivo de video…",
        )
        self.row_video.btn.clicked.connect(self.elegir_video)
        root.addWidget(self.row_video)

        self.row_outdir = FieldRow(
            "Carpeta de salida", btn_text="Elegir",
            placeholder="Si lo dejas vacío, se guardará en la carpeta del video",
        )
        self.row_outdir.btn.clicked.connect(self.elegir_carpeta_salida)
        root.addWidget(self.row_outdir)

        self.chk_subcarpeta = QCheckBox('Crear subcarpeta "Chapters"')
        root.addWidget(self.chk_subcarpeta)

        self.chk_exacto = QCheckBox("OP/ED exactos (AnimeThemes + coincidencia de audio)")
        self.chk_exacto.setChecked(True)
        root.addWidget(self.chk_exacto)

        self.row_search = FieldRow(
            "Búsqueda en AnimeThemes (opcional)",
            placeholder="Dejar vacío para detectar automáticamente",
        )
        self.row_search.btn.hide()
        root.addWidget(self.row_search)

        box    = QGroupBox("Parámetros de coincidencia")
        boxlay = QHBoxLayout()
        boxlay.setSpacing(16)

        for attr, label, default, width in [
            ("ed_sub",     "Submuestreo",                     "32",   70),
            ("ed_portion", "Porción del theme (0.5~1.0)",     "0.90", 80),
            ("ed_min",     "Umbral de puntuación (0.10~0.40)","0.25", 80),
        ]:
            col = QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(QLabel(label.upper()))
            field = QLineEdit(default)
            field.setFixedWidth(width)
            col.addWidget(field)
            boxlay.addLayout(col)
            setattr(self, attr, field)

        boxlay.addStretch(1)
        box.setLayout(boxlay)
        root.addWidget(box)

        sep2 = QFrame()
        sep2.setObjectName("separator")
        root.addWidget(sep2)

        self.btn_run = QPushButton("Generar XML")
        self.btn_run.setObjectName("run")
        self.btn_run.clicked.connect(self.iniciar)
        root.addWidget(self.btn_run, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        log_lbl = QLabel("LOG DE PROCESO")
        log_lbl.setObjectName("section")
        root.addWidget(log_lbl)

        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(2000)
        self.log_widget.setMinimumHeight(160)
        root.addWidget(self.log_widget, 1)

    def _agregar_log(self, s: str):
        self.log_widget.appendPlainText(s)

    def _todos_controles(self):
        return [
            self.row_video.btn, self.row_outdir.btn,
            self.row_outdir.entry,
            self.chk_subcarpeta, self.chk_exacto,
            self.row_search.entry,
            self.ed_sub, self.ed_portion, self.ed_min,
            self.btn_run,
        ]

    def habilitar_controles(self, enabled: bool):
        for w in self._todos_controles():
            w.setEnabled(enabled)

    def elegir_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Selecciona un video", "",
            "Videos (*.mkv *.mp4 *.avi *.webm *.mov *.m2ts *.ts *.wmv *.vob);;Todos (*.*)",
        )
        if fp:
            self.row_video.set(fp)

    def elegir_carpeta_salida(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Selecciona una carpeta de salida")
        if carpeta:
            self.row_outdir.set(carpeta)

    def iniciar(self):
        video = self.row_video.get()
        if not video or not Path(video).exists() or not video.lower().endswith(VIDEO_EXTS):
            QMessageBox.critical(self, "Error", "Selecciona un video válido.")
            return

        try:
            params = ParametrosTrabajo(
                video=video,
                carpeta_salida=self.row_outdir.get(),
                crear_subcarpeta=self.chk_subcarpeta.isChecked(),
                usar_exacto=self.chk_exacto.isChecked(),
                submuestreo=int(self.ed_sub.text().strip() or "32"),
                porcion_theme=float(self.ed_portion.text().strip() or "0.90"),
                puntuacion_minima=float(self.ed_min.text().strip() or "0.25"),
                search_override=self.row_search.get(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Parámetros inválidos:\n{e}")
            return

        self.log_widget.clear()
        self.progress.setValue(0)
        self.habilitar_controles(False)

        self._resolver = ResolverWorker(self, params, interactivo=True)
        self._resolver.log.connect(self._agregar_log)
        self._resolver.progress.connect(self.progress.setValue)
        self._resolver.need_pick.connect(self._on_need_pick)
        self._resolver.resolved.connect(self._on_resolved_params)
        self._resolver.failed.connect(self._on_fail)
        self._resolver.start()

    def _on_need_pick(self, req: PickRequest):
        dlg = DialogoSelectorTabla(
            self, req.titulo, req.subtitulo, req.columnas, req.filas
        )
        idx = dlg.indice_seleccionado() if dlg.exec() == QDialog.DialogCode.Accepted else None
        if self._resolver:
            self._resolver.entregar_pick(idx)

    def _on_resolved_params(self, params: ParametrosTrabajo):
        self._agregar_log("• Resolución completa. Iniciando coincidencia/capítulos…")
        self._worker = ChapterizerWorker(self, params)
        self._worker.log.connect(self._agregar_log)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.terminado.connect(self._on_done)
        self._worker.fallo.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, ruta_salida: str):
        self.habilitar_controles(True)
        self._resolver = None
        self._worker   = None
        QMessageBox.information(self, "OK", f"Chapters generados:\n{ruta_salida}")

    def _on_fail(self, msg: str):
        self.habilitar_controles(True)
        self.progress.setValue(0)
        if self._resolver and self._resolver.isRunning():
            self._resolver.cancelar()
            self._resolver.wait(2000)
        self._resolver = None
        self._worker   = None
        self._agregar_log(f"❌ Error: {msg}")
        QMessageBox.critical(self, "Error", msg)


def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = VentanaPrincipal()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()