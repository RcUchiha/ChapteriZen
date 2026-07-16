r"""
Simula el camino de decision de ResolverWorker.run() (gui/resolver_worker.py)
hasta el punto exacto donde hoy se llamaria a trace.moe -- sin llamar a
trace.moe. Sirve para ver, sobre una carpeta real de videos, cuantos
archivos ya quedan resueltos por el parsing solo y cuantos necesitarian
el paso siguiente (trace.moe), sin gastar ni arriesgar la cuota de
trace.moe (que sigue en estado desconocido desde la ultima calibracion,
ver docs/KNOWN_LIMITATIONS.md).

No reimplementa la logica de merge de aniparse/anitopy: usa
parsear_nombre_archivo() de produccion tal cual. Los campos
"*_fuente" son diagnostico por INTROSPECCION (comparar el resultado
final contra lo que devuelve cada parser crudo por separado), no una
re-decision -- por eso pueden marcarse "indeterminado" en el puñado de
casos donde el recorte de sufijo de temporada hace que el titulo final
no matchee texto-a-texto con ninguno de los dos crudos.

Condicion real que dispara trace.moe en produccion (resolver_worker.py,
confirmado leyendo el codigo antes de escribir esto, no asumido):

    consulta_base = inferir_consulta_desde_nombre_archivo(video)
    consulta_base = quitar_sufijo_episodio(consulta_base)
    if not _titulo_es_usable(consulta_base) or re.fullmatch(r'\d+', consulta_base.strip()):
        # ... aca es donde se llamaria a trace.moe

Es un OR compuesto -- _titulo_es_usable sola NO alcanza, porque títulos
puramente numericos (ej. "2451") pasan _titulo_es_usable pero igual
disparan el fallback por el chequeo de regex aparte (ver
tests/test_parsing.py::TestTituloEsUsable::test_titulo_puramente_numerico_2451_ES_usable).
Este script replica el OR completo, no solo la primera mitad.

Uso:
    python scripts/simular_decision_resolver.py <carpeta_con_episodios>
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapterizen.config import VIDEO_EXTS  # dispara chapterizen/__init__.py -> config.py

from loguru import logger

# Mismo cuidado que calibrar_trace_moe.py: importar chapterizen configura
# un sink de loguru hacia el log real de produccion apenas se importa el
# paquete (ver docs/TECH_DEBT.md). Este script no llama a ninguna funcion
# que loguee nada sensible, pero se redirige de todas formas por si
# parsear_nombre_archivo emite logger.debug(...) con detalle de parsing.
_LOG_SIMULACION = Path(__file__).with_name("simulacion_decision_resolver.log")
logger.remove()
logger.add(_LOG_SIMULACION, level="DEBUG", encoding="utf-8")

# Recien ahora se importa el resto de chapterizen.parsing -- no agrega
# sinks propios, solo usa el logger ya redirigido de arriba.
from chapterizen.parsing import (
    parsear_nombre_archivo,
    quitar_sufijo_episodio,
    _titulo_es_usable,
    _normalizar_titulo_parser,
    _parse_con_aniparse,
    _parse_con_anitopy,
    _campos_desde_aniparse,
    _campos_desde_anitopy,
)


def _detectar_fuentes(stem: str, resultado):
    """Introspeccion diagnostica (no una re-decision): compara el
    resultado final de parsear_nombre_archivo() contra lo que devuelve
    cada parser crudo por separado, para etiquetar de donde vino cada
    campo. Si fuente_produccion == "fallback", los tres campos vienen
    del regex de ultimo recurso, no de ningun parser."""
    if resultado.fuente == "fallback":
        return "regex_fallback", "regex_fallback", "regex_fallback"

    a = _parse_con_aniparse(stem)
    b = _parse_con_anitopy(stem)
    titulo_a, temp_a, ep_a = _campos_desde_aniparse(a) if a else ("", None, None)
    titulo_b, temp_b, ep_b = _campos_desde_anitopy(b) if b else ("", None, None)
    titulo_a_n = _normalizar_titulo_parser(titulo_a)
    titulo_b_n = _normalizar_titulo_parser(titulo_b)
    final_n = _normalizar_titulo_parser(resultado.titulo or "")

    if titulo_a_n and final_n == titulo_a_n and final_n != titulo_b_n:
        titulo_fuente = "aniparse"
    elif titulo_b_n and final_n == titulo_b_n and final_n != titulo_a_n:
        titulo_fuente = "anitopy"
    elif titulo_a_n and titulo_b_n and final_n == titulo_a_n == titulo_b_n:
        titulo_fuente = "ambos_iguales"
    else:
        # Puede pasar por el recorte de sufijo de temporada (parsing.py
        # le quita " <temporada>" al final antes de elegir) -- no se
        # fuerza una atribucion en ese caso, se marca honestamente.
        titulo_fuente = "indeterminado"

    if resultado.temporada is None:
        temporada_fuente = "ninguno"
    elif resultado.temporada == temp_a:
        temporada_fuente = "aniparse"
    elif resultado.temporada == temp_b:
        temporada_fuente = "anitopy"
    else:
        temporada_fuente = "indeterminado"

    if resultado.episodio is None:
        episodio_fuente = "ninguno"
    elif resultado.episodio == ep_a:
        episodio_fuente = "aniparse"
    elif resultado.episodio == ep_b:
        episodio_fuente = "anitopy"
    else:
        episodio_fuente = "indeterminado"

    return titulo_fuente, temporada_fuente, episodio_fuente


def main():
    if len(sys.argv) != 2:
        print(f"Uso: python {Path(__file__).name} <carpeta_con_episodios>")
        sys.exit(1)

    carpeta = Path(sys.argv[1])
    if not carpeta.is_dir():
        print(f"No es una carpeta: {carpeta}")
        sys.exit(1)

    videos = sorted(
        p for p in carpeta.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"Sin videos ({', '.join(VIDEO_EXTS)}) en {carpeta}")
        sys.exit(1)

    csv_path = Path(__file__).with_name("simulacion_decision_resolver.csv")
    print(f"{len(videos)} video(s) encontrados. Log detallado (DEBUG) en: {_LOG_SIMULACION}")
    print(f"Resultados incrementales en: {csv_path}\n")

    resueltos_por_parsing = []
    necesitarian_trace_moe = []

    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        escritor = csv.writer(f_csv)
        if f_csv.tell() == 0:
            escritor.writerow([
                "archivo", "titulo", "titulo_tras_quitar_sufijo",
                "temporada", "episodio", "fuente_produccion",
                "titulo_fuente", "temporada_fuente", "episodio_fuente",
                "titulo_usable", "es_puramente_numerico",
                "necesitaria_trace_moe",
            ])
            f_csv.flush()

        for video in videos:
            stem = video.stem
            resultado = parsear_nombre_archivo(str(video))

            # Mismo orden exacto que ResolverWorker.run(): inferir el
            # titulo y LUEGO quitarle un posible sufijo de episodio antes
            # de evaluar si haria falta trace.moe.
            consulta_base = quitar_sufijo_episodio(resultado.titulo)
            usable = _titulo_es_usable(consulta_base)
            puramente_numerico = bool(re.fullmatch(r"\d+", consulta_base.strip()))
            necesita_trace_moe = (not usable) or puramente_numerico

            titulo_fuente, temporada_fuente, episodio_fuente = _detectar_fuentes(stem, resultado)

            fila = [
                video.name, resultado.titulo, consulta_base,
                resultado.temporada, resultado.episodio, resultado.fuente,
                titulo_fuente, temporada_fuente, episodio_fuente,
                usable, puramente_numerico, necesita_trace_moe,
            ]
            escritor.writerow(fila)
            f_csv.flush()

            print(
                f"{video.name:<70} T={str(resultado.temporada):<5} "
                f"E={str(resultado.episodio):<5} "
                f"{'[NECESITA TRACE.MOE]' if necesita_trace_moe else ''}"
            )

            if necesita_trace_moe:
                necesitarian_trace_moe.append(video.name)
            else:
                resueltos_por_parsing.append(video.name)

    print("\n" + "=" * 90)
    print(f"Resueltos por el parsing solo: {len(resueltos_por_parsing)} / {len(videos)}")
    print(f"Necesitarian trace.moe (o resolucion manual): {len(necesitarian_trace_moe)} / {len(videos)}")
    print("=" * 90)
    for n in necesitarian_trace_moe:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
