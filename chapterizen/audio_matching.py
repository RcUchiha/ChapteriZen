"""Pipeline hibrido de matching de audio FFT -> DTW para localizar
OP/ED. Movido sin cambios desde chapterizen.py (monolito original,
v0.0.7)."""
import hashlib
from typing import Optional, Tuple

import numpy as np
import librosa

from .config import get_api_cache, _TTL_THEMES_DAYS


# scipy es opcional: si está disponible se usa para FFT más rápida
try:
    from scipy.fft import rfft, irfft, next_fast_len as _scipy_next_fast_len
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

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
    cached = get_api_cache().get(clave)
    if cached is not None:
        return cached
    feat = extraer_features(y, sr)
    get_api_cache().set(clave, feat, expire=_TTL_THEMES_DAYS * 86400)
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

    # Padding de silencio al inicio para que la correlación cruzada pueda
    # detectar cuando el OP/ED empieza exactamente en t=0. Sin él, la ventana
    # de correlación no tiene margen para "deslizarse" hasta el inicio del audio.
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

def _tiempo_sin_ms(t: float) -> str:
    total = int(round(t))
    h, rem = divmod(total, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
