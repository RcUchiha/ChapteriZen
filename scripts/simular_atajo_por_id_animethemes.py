r"""
Simulacion (no toca produccion) para evaluar el atajo por ID externo
propuesto para AnimeThemes: filter[has]=resources&filter[site]=...&
filter[external_id]=... contra Jikan/AniList, comparado con
_aceptar_canon_sin_perder_tokens() como validacion cruzada antes de
aceptar el atajo sin picker.

Reusa funciones de produccion tal cual (jikan_resolver_titulo,
anilist_buscar_titulo, buscar_anime_en_animethemes,
filtrar_por_token_obligatorio, _preferir_resultados_por_temporada,
animethemes_coincidencia_exacta_por_titulo, _aceptar_canon_sin_perder_tokens)
-- no reimplementa esa logica. La UNICA pieza que no existe todavia en
produccion es la consulta al endpoint por ID (animethemes_buscar_por_id_externo
no esta implementada aun), asi que ese HTTP se hace crudo aca mismo, con
el mismo patron de cache/reintento que el resto de animethemes.py.

Por archivo, hace en este orden (mismo orden que ResolverWorker.run()
hasta el punto de _resolver_slug_con_picker, sin trace.moe -- los
archivos que lo necesitarian se marcan aparte y no consumen su cuota):

  1. Parsear filename -> consulta_base, temporada, episodio.
  2. Si necesitaria trace.moe en produccion (titulo no usable / puramente
     numerico / artefacto pegado): fuera de alcance, se marca y se sigue.
  3. Resolver via Jikan (respaldo AniList si Jikan cae) -> picked_base.
  4. Construir ids_externos candidatos desde picked_base (mal_id, o
     id/idMal si es shape de AniList).
  5. Consultar AnimeThemes por CADA id candidato hasta el primer match
     unico (exactamente 1 resultado).
  6. Si hay match por ID: validar con _aceptar_canon_sin_perder_tokens
     contra consulta_base.
  7. Para comparar: correr TAMBIEN el camino de texto actual completo
     (buscar + filtrar_por_token_obligatorio + preferir_por_temporada +
     coincidencia_exacta) y ver si resuelve solo o necesitaria picker hoy.
  8. Registrar todo en CSV para poder auditar cada fila a mano.

Uso:
    python scripts/simular_atajo_por_id_animethemes.py <carpeta1> [<carpeta2> ...]
"""
import csv
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chapterizen.config import (
    VIDEO_EXTS,
    _http,
    _reintento_http,
    _es_error_transitorio,
    get_api_cache,
    _TTL_API_DAYS,
    ANIMETHEMES_ANIME,
)

from loguru import logger

_LOG_SIMULACION = Path(__file__).with_name("simulacion_atajo_id_animethemes.log")
logger.remove()
logger.add(_LOG_SIMULACION, level="DEBUG", encoding="utf-8")

from chapterizen.parsing import (
    inferir_consulta_desde_nombre_archivo,
    quitar_sufijo_episodio,
    quitar_marcador_temporada,
    _titulo_es_usable,
    _titulo_tiene_artefacto_pegado,
    _preferir_resultados_por_temporada,
)
from chapterizen.jikan import (
    extraer_temporada_y_episodio_desde_nombre_archivo,
    jikan_resolver_titulo,
    jikan_titulos_desde_item,
    _aceptar_canon_sin_perder_tokens,
    animethemes_coincidencia_exacta_por_titulo,
    filtrar_por_token_obligatorio,
)
from chapterizen.anilist import anilist_buscar_titulo, anilist_titulos_desde_item
from chapterizen.animethemes import buscar_anime_en_animethemes

_ESPERA_ENTRE_ARCHIVOS_SEG = 0.3  # cortesia hacia Jikan/AniList/AnimeThemes


@_reintento_http
def _animethemes_por_id(site: str, external_id: int) -> List[dict]:
    """Misma forma que la animethemes_buscar_por_id_externo propuesta
    (no implementada en produccion todavia) -- vive aca solo para esta
    simulacion, con cache real de la app (get_api_cache()) igual que el
    resto de animethemes.py."""
    clave  = f"at_by_id:{site}:{external_id}"
    cached = get_api_cache().get(clave)
    if cached is not None:
        return cached
    r = _http.get(ANIMETHEMES_ANIME, params={
        "filter[has]": "resources",
        "filter[site]": site,
        "filter[external_id]": str(external_id),
    })
    r.raise_for_status()
    result = (r.json() or {}).get("anime") or []
    get_api_cache().set(clave, result, expire=_TTL_API_DAYS * 86400)
    return result


def _ids_externos_de(picked_base: dict) -> List[Tuple[str, int]]:
    if "mal_id" in picked_base:
        mal_id = picked_base.get("mal_id")
        return [("MyAnimeList", mal_id)] if mal_id else []
    ids = []
    if picked_base.get("id"):
        ids.append(("Anilist", picked_base["id"]))
    if picked_base.get("idMal"):
        ids.append(("MyAnimeList", picked_base["idMal"]))
    return ids


def _titulos_conocidos_de(picked_base: dict) -> List[str]:
    """Titulos que Jikan/AniList YA conoce para este picked_base (incluye
    title_japanese/native) -- shape-aware, mismo criterio que usa
    _resolver_slug_con_picker para elegir entre jikan_titulos_desde_item
    y anilist_titulos_desde_item."""
    if "mal_id" in picked_base:
        return jikan_titulos_desde_item(picked_base)
    return anilist_titulos_desde_item(picked_base)


def _token_ok_contra_titulos_conocidos(nombre_at: str, titulos_conocidos: List[str]) -> bool:
    """Validacion alternativa: en vez de comparar nombre_at contra el
    texto crudo del filename, compararlo contra CADA titulo que
    Jikan/AniList ya conoce para el mismo picked_base (incluye
    japones/native) -- acepta si nombre_at no pierde tokens de AL MENOS
    uno de esos titulos. No protege contra un picked_base mal
    identificado desde el origen (ver docs/TECH_DEBT.md): si
    Jikan/AniList ya resolvio la serie equivocada, todos los titulos
    conocidos heredan ese mismo error y la validacion no lo detecta."""
    return any(
        _aceptar_canon_sin_perder_tokens(t, nombre_at)
        for t in titulos_conocidos
        if t
    )


def _texto_resuelve_solo(consulta_base: str, temporada: int) -> Optional[dict]:
    """Replica el primer intento (sin titulos alternativos, sin picker)
    de _resolver_slug_con_picker: unico resultado tras filtrar, o
    coincidencia exacta de titulo entre los priorizados."""
    resultados = buscar_anime_en_animethemes(consulta_base)
    resultados = filtrar_por_token_obligatorio(consulta_base, resultados)
    resultados = _preferir_resultados_por_temporada(resultados, temporada)
    if len(resultados) == 1:
        return resultados[0]
    return animethemes_coincidencia_exacta_por_titulo(resultados, consulta_base)


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {Path(__file__).name} <carpeta1> [<carpeta2> ...]")
        sys.exit(1)

    videos = []
    for arg in sys.argv[1:]:
        carpeta = Path(arg)
        if not carpeta.is_dir():
            print(f"No es una carpeta: {carpeta}")
            sys.exit(1)
        videos.extend(
            p for p in carpeta.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
    videos = sorted(videos)
    if not videos:
        print(f"Sin videos ({', '.join(VIDEO_EXTS)}) en las carpetas dadas")
        sys.exit(1)

    csv_path = Path(__file__).with_name("simulacion_atajo_id_animethemes_v2.csv")
    print(f"{len(videos)} video(s) encontrados. Log detallado (DEBUG) en: {_LOG_SIMULACION}")
    print(f"Resultados incrementales en: {csv_path}\n")

    contadores = {
        "requiere_trace_moe": 0,
        "sin_resultado_jikan_anilist": 0,
        "sin_id_candidato": 0,
        "id_sin_match": 0,
        "id_match_token_ok": 0,
        "id_match_token_rechaza": 0,
        "error": 0,
    }
    # Contadores v2 aparte -- se cuentan solo sobre las filas que SI
    # llegan a tener un resultado_id (mismo universo que token_ok v1).
    contadores_v2 = {"v2_ok": 0, "v2_rechaza": 0}
    desacuerdos_token_ok = []       # token_ok (v1) True pero slug difiere del texto
    rechazos_a_revisar   = []       # token_ok (v1) False -- puede ser correcto o falso positivo

    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        escritor = csv.writer(f_csv)
        if f_csv.tell() == 0:
            escritor.writerow([
                "archivo", "consulta_base", "temporada", "episodio",
                "categoria", "site_id", "external_id",
                "nombre_animethemes_por_id", "slug_por_id",
                "token_ok", "token_ok_v2", "titulos_conocidos",
                "texto_resuelve_solo", "nombre_texto", "slug_texto",
                "slugs_coinciden",
            ])
            f_csv.flush()

        for video in videos:
            time.sleep(_ESPERA_ENTRE_ARCHIVOS_SEG)
            stem = video.name

            temporada_raw, ep = extraer_temporada_y_episodio_desde_nombre_archivo(str(video))
            episodio  = int(ep or 0)
            temporada = 1 if temporada_raw is None else int(temporada_raw)

            consulta_base = inferir_consulta_desde_nombre_archivo(str(video))
            consulta_base = quitar_sufijo_episodio(consulta_base)

            if (
                not _titulo_es_usable(consulta_base)
                or re.fullmatch(r"\d+", consulta_base.strip())
                or _titulo_tiene_artefacto_pegado(consulta_base)
            ):
                contadores["requiere_trace_moe"] += 1
                escritor.writerow([stem, consulta_base, temporada, episodio, "requiere_trace_moe"] + [""] * 11)
                f_csv.flush()
                print(f"{stem:<70} [requiere trace.moe -- fuera de alcance]")
                continue

            consulta_jikan = quitar_marcador_temporada(consulta_base)

            try:
                titulo_resuelto, picked_base, titulo_confiable, ts1 = jikan_resolver_titulo(consulta_jikan)
            except Exception as e:
                if not _es_error_transitorio(e):
                    contadores["error"] += 1
                    escritor.writerow([stem, consulta_base, temporada, episodio, f"error:{e}"] + [""] * 11)
                    f_csv.flush()
                    print(f"{stem:<70} [ERROR: {e}]")
                    continue
                try:
                    titulo_resuelto, picked_base, titulo_confiable, ts1 = anilist_buscar_titulo(consulta_jikan)
                except Exception as e2:
                    contadores["error"] += 1
                    escritor.writerow([stem, consulta_base, temporada, episodio, f"error:{e2}"] + [""] * 11)
                    f_csv.flush()
                    print(f"{stem:<70} [ERROR (AniList): {e2}]")
                    continue

            if not picked_base:
                contadores["sin_resultado_jikan_anilist"] += 1
                escritor.writerow([stem, consulta_base, temporada, episodio, "sin_resultado_jikan_anilist"] + [""] * 11)
                f_csv.flush()
                print(f"{stem:<70} [sin resultado Jikan/AniList]")
                continue

            ids_externos = _ids_externos_de(picked_base)

            texto_it     = _texto_resuelve_solo(consulta_base, temporada)
            texto_slug   = (texto_it or {}).get("slug") or ""
            texto_nombre = (texto_it or {}).get("name") or ""

            if not ids_externos:
                contadores["sin_id_candidato"] += 1
                escritor.writerow([
                    stem, consulta_base, temporada, episodio, "sin_id_candidato",
                    "", "", "", "", "", "", "",
                    bool(texto_it), texto_nombre, texto_slug, "",
                ])
                f_csv.flush()
                print(f"{stem:<70} [sin id candidato en picked_base]")
                continue

            resultado_id = None
            site_usado   = None
            id_usado     = None
            for site, ext_id in ids_externos:
                resultados = _animethemes_por_id(site, ext_id)
                if len(resultados) == 1:
                    resultado_id = resultados[0]
                    site_usado   = site
                    id_usado     = ext_id
                    break

            if resultado_id is None:
                contadores["id_sin_match"] += 1
                escritor.writerow([
                    stem, consulta_base, temporada, episodio, "id_sin_match",
                    "|".join(f"{s}:{i}" for s, i in ids_externos), "", "", "", "", "", "",
                    bool(texto_it), texto_nombre, texto_slug, "",
                ])
                f_csv.flush()
                print(f"{stem:<70} [sin match por ID en {['%s:%s' % (s,i) for s,i in ids_externos]}]")
                continue

            nombre_id = resultado_id.get("name") or ""
            slug_id   = resultado_id.get("slug") or ""
            token_ok  = _aceptar_canon_sin_perder_tokens(consulta_base, nombre_id)
            slugs_coinciden = (slug_id == texto_slug) if texto_slug else None

            titulos_conocidos = _titulos_conocidos_de(picked_base)
            token_ok_v2 = _token_ok_contra_titulos_conocidos(nombre_id, titulos_conocidos)
            contadores_v2["v2_ok" if token_ok_v2 else "v2_rechaza"] += 1

            categoria = "id_match_token_ok" if token_ok else "id_match_token_rechaza"
            contadores[categoria] += 1

            escritor.writerow([
                stem, consulta_base, temporada, episodio, categoria,
                site_usado, id_usado, nombre_id, slug_id, token_ok, token_ok_v2,
                "|".join(titulos_conocidos), bool(texto_it), texto_nombre, texto_slug, slugs_coinciden,
            ])
            f_csv.flush()

            marca    = "✓" if token_ok else "✗"
            marca_v2 = "✓" if token_ok_v2 else "✗"
            print(f"{stem:<70} [{categoria}] token_ok={marca} token_ok_v2={marca_v2} slug_id={slug_id!r} slug_texto={texto_slug!r}")

            if token_ok and texto_slug and slug_id != texto_slug:
                desacuerdos_token_ok.append((stem, consulta_base, slug_id, texto_slug))
            if not token_ok:
                rechazos_a_revisar.append((stem, consulta_base, nombre_id, slug_id, texto_slug, token_ok_v2))

    print("\n" + "=" * 90)
    total = len(videos)
    for k, v in contadores.items():
        print(f"{k:<32} {v:>4} / {total}")
    for k, v in contadores_v2.items():
        print(f"{k:<32} {v:>4} / {total}")
    print("=" * 90)

    print(f"\nDesacuerdos con token_ok=True pero slug distinto del texto ({len(desacuerdos_token_ok)}):")
    for stem, cb, sid, stx in desacuerdos_token_ok:
        print(f"  - {stem}\n      consulta_base={cb!r}\n      slug_por_id={sid!r} vs slug_texto={stx!r}")

    print(f"\nRechazos de token_ok v1 (revisar si son correctos o falsos positivos, y si v2 los rescata) ({len(rechazos_a_revisar)}):")
    for stem, cb, nombre_id, sid, stx, tok_v2 in rechazos_a_revisar:
        print(f"  - {stem}\n      consulta_base={cb!r}\n      nombre_animethemes_por_id={nombre_id!r} (slug={sid!r})\n      slug_texto={stx!r}  token_ok_v2={tok_v2}")


if __name__ == "__main__":
    main()
