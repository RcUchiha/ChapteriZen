"""Generacion del XML de chapters para mkvmerge. Movido sin cambios
desde chapterizen.py (monolito original, v0.0.7)."""
from pathlib import Path
from typing import Tuple, List
from xml.sax.saxutils import escape


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
