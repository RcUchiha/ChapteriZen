"""Utilidades de sistema/ffmpeg: ejecucion de comandos, extraccion de
fotogramas y audio. Movido sin cambios desde chapterizen.py (monolito
original, v0.0.7)."""
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np


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

def extraer_fotogramas_centrado(
    ruta_video:   str,
    dir_salida:   str,
    duracion:     float,
    n_fotogramas: int = 9,
) -> List[Path]:
    """
    Extrae N fotogramas distribuidos uniformemente a lo largo del video,
    ordenados de más central a más extremo. El primer frame (mitad del video)
    es el más representativo del contenido del episodio, maximizando la
    efectividad de la salida temprana en trace.moe.
    Cada frame se extrae con un seek independiente (-ss antes de -i) para
    que la posición sea exacta sin decodificar el video completo.
    """
    salida = Path(dir_salida)
    salida.mkdir(parents=True, exist_ok=True)

    # Posiciones uniformes que evitan los extremos del video
    puntos = [duracion * (i + 1) / (n_fotogramas + 1) for i in range(n_fotogramas)]

    # Reordenar de más central (índice n//2) hacia los extremos
    centro_idx = n_fotogramas // 2
    orden: List[int] = [centro_idx]
    for i in range(1, centro_idx + 1):
        if centro_idx - i >= 0:
            orden.append(centro_idx - i)
        if centro_idx + i < n_fotogramas:
            orden.append(centro_idx + i)

    rutas: List[Path] = []
    for prioridad, idx in enumerate(orden, 1):
        ts   = puntos[idx]
        ruta = salida / f"frame_{prioridad:03d}.jpg"
        cmd  = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}", "-i", ruta_video,
            "-vframes", "1", str(ruta),
        ]
        try:
            ejecutar_comando(cmd)
            if ruta.exists():
                rutas.append(ruta)
        except Exception:
            pass

    return rutas

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
