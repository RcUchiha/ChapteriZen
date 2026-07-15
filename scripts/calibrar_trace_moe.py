"""
Script standalone de calibracion para el punto 5 del analisis de
resolver_worker.py: elegir el valor de una nueva constante propuesta,
_TRACE_UMBRAL_CONFIANZA_ALTA (todavia NO existe en el codigo -- esto es
solo recoleccion de datos, no toca resolver_worker.py ni trace_moe.py).

Corre identificar_anime_con_fotogramas() directamente sobre una carpeta
de episodios reales, sin pasar por ResolverWorker/Jikan/AnimeThemes --
mide unicamente que tan seguido acierta trace.moe cuando su similitud de
consenso cae en la banda 0.85-0.95, que es la pregunta real detras de
elegir 0.88 vs 0.90 vs 0.92 (ver docs/TECH_DEBT.md y la conversacion que
origino este script).

Uso:
    python scripts/calibrar_trace_moe.py <carpeta_con_episodios>

Imprime, por cada video: archivo, similitud, anilist_id detectado y
titulo. No hay comparacion automatica contra un "correcto" -- el usuario
ya sabe que es cada episodio de su propia libreria, asi que compara a
mano contra la columna impresa.

Ademas de stdout (con flush por linea), cada resultado se agrega de
inmediato a calibracion_resultados.csv (junto a este archivo, modo
append) con flush por fila -- si el proceso se corta a mitad de una
carpeta grande (Ctrl+C, cierre de terminal, un 429 persistente que agota
reintentos en TODOS los frames de un video), lo ya procesado antes del
corte queda guardado en el CSV, no solo en el scrollback de la terminal.

Nota sobre logging (ver entrada nueva en docs/TECH_DEBT.md): importar
chapterizen.config configura un sink de loguru hacia el log real de
produccion (chapterizen_YYYY-MM-DD.log) apenas se importa el paquete,
sin distinguir GUI de un script de una vez. Este script reemplaza ese
sink por uno propio (calibracion_trace_moe.log, junto a este archivo)
INMEDIATAMENTE despues de ese import y ANTES de llamar a cualquier otra
funcion de chapterizen -- ninguna linea de esta corrida termina en el
archivo de produccion.
"""
import csv
import sys
import tempfile
import time
from pathlib import Path

_ESPERA_ENTRE_VIDEOS_SEG = 2.0  # cortesia hacia trace.moe (servicio gratuito compartido)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapterizen.config import VIDEO_EXTS  # dispara chapterizen/__init__.py -> config.py

from loguru import logger

_LOG_CALIBRACION = Path(__file__).with_name("calibracion_trace_moe.log")
logger.remove()  # saca el sink de produccion que config.py acaba de agregar
logger.add(_LOG_CALIBRACION, level="DEBUG", encoding="utf-8")

# Recien ahora se importa el resto -- estos modulos no agregan sinks
# propios, solo usan el logger ya redirigido de arriba.
from chapterizen.ffmpeg_utils import (
    asegurar_ffmpeg,
    duracion_con_ffprobe,
    extraer_fotogramas_centrado,
)
from chapterizen.trace_moe import identificar_anime_con_fotogramas


def _identificar(video: Path):
    """Mismo mecanismo que ResolverWorker._identificar_con_trace_moe
    (gui/resolver_worker.py), pero standalone (sin QThread ni self._log)."""
    asegurar_ffmpeg()
    with tempfile.TemporaryDirectory() as dir_tmp:
        duracion = duracion_con_ffprobe(str(video))
        frames   = extraer_fotogramas_centrado(str(video), dir_tmp, duracion)
        if not frames:
            raise RuntimeError("no se pudieron extraer fotogramas")
        resultado, _ = identificar_anime_con_fotogramas(
            frames, log_fn=lambda s: print(f"    {s}")
        )
        return resultado


def main():
    if len(sys.argv) != 2:
        print(f"Uso: python {Path(__file__).name} <carpeta_con_episodios>")
        sys.exit(1)

    carpeta = Path(sys.argv[1])
    if not carpeta.is_dir():
        print(f"No es una carpeta: {carpeta}")
        sys.exit(1)

    videos = sorted(
        p for p in carpeta.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"Sin videos ({', '.join(VIDEO_EXTS)}) en {carpeta}")
        sys.exit(1)

    csv_path = Path(__file__).with_name("calibracion_resultados.csv")
    print(f"{len(videos)} video(s) encontrados. Log detallado (DEBUG) en: {_LOG_CALIBRACION}")
    print(f"Resultados incrementales (sobreviven un corte a mitad de carpeta) en: {csv_path}\n")
    print(f"{'archivo':<45} {'similitud':>10}  {'anilist_id':>10}  titulo")
    print("-" * 100, flush=True)

    # Se abre en modo "a" y se hace flush por fila (no al final) para que un
    # Ctrl+C o un cierre abrupto a mitad de carpeta no pierda lo ya procesado.
    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        escritor = csv.writer(f_csv)
        if f_csv.tell() == 0:
            escritor.writerow(["archivo", "similitud", "anilist_id", "titulo", "error"])
            f_csv.flush()

        for i, video in enumerate(videos):
            try:
                r   = _identificar(video)
                aid = "" if r.anilist_id is None else str(r.anilist_id)
                print(f"{video.name:<45} {r.similitud:>9.2%}  {aid:>10}  {r.titulo}", flush=True)
                escritor.writerow([video.name, f"{r.similitud:.4f}", aid, r.titulo, ""])
            except Exception as e:
                print(f"{video.name:<45} ERROR: {e}", flush=True)
                escritor.writerow([video.name, "", "", "", str(e)])
            f_csv.flush()
            if i < len(videos) - 1:
                time.sleep(_ESPERA_ENTRE_VIDEOS_SEG)


if __name__ == "__main__":
    main()
