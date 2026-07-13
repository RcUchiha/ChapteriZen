"""Construccion de la ruta de salida del XML de chapters. Movido sin
cambios desde chapterizen.py (monolito original, v0.0.7)."""
from pathlib import Path

from .animethemes import nombre_archivo_seguro


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
