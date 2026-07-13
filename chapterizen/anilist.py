"""Integracion con AniList: busqueda de anime por texto, pensada como
fuente alternativa cuando Jikan falla o resulta ambiguo (no conectada al
flujo real todavia -- ver ResolverWorker en gui/workers.py).

anilist_titulo_por_id (lookup por ID de AniList, usado por trace.moe)
sigue viviendo en trace_moe.py -- gui/workers.py ya importa esa funcion
desde ahi y no se toca en esta fase. Este modulo es solo para busqueda
por texto.
"""
import re
from typing import Optional, Tuple, List

from loguru import logger
from rapidfuzz import fuzz as _fuzz

from .config import _http, _reintento_http, _API_CACHE, _TTL_API_DAYS, ANILIST_GRAPHQL


_ANILIST_SEARCH_QUERY = """
query ($search: String) {
  Page(page: 1, perPage: 10) {
    media(search: $search, type: ANIME) {
      id
      idMal
      title {
        romaji
        english
        native
        userPreferred
      }
      synonyms
      format
      status
      episodes
      startDate {
        year
        month
        day
      }
    }
  }
}
"""


@_reintento_http
def _anilist_buscar_media(consulta: str) -> List[dict]:
    clave  = f"anilist_search:{consulta.strip().casefold()}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    r = _http.post(
        ANILIST_GRAPHQL,
        json={"query": _ANILIST_SEARCH_QUERY, "variables": {"search": consulta}},
    )
    r.raise_for_status()
    data   = r.json() or {}
    result = (((data.get("data") or {}).get("Page") or {}).get("media") or [])
    _API_CACHE.set(clave, result, expire=_TTL_API_DAYS * 86400)
    return result


def anilist_titulos_desde_item(item: dict) -> List[str]:
    """Extrae titulos/sinonimos utiles de un Media de AniList, en el mismo
    espiritu que jikan_titulos_desde_item (jikan.py) pero para el shape de
    AniList: title.romaji/english/native/userPreferred + synonyms."""
    titulos = []
    title = item.get("title") or {}
    for k in ("romaji", "english", "native", "userPreferred"):
        t = title.get(k)
        if t:
            titulos.append(t)
    for sinonimo in item.get("synonyms") or []:
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


def _ratio(a: str, b: str) -> float:
    # rapidfuzz devuelve 0–100, normalizamos a 0–1
    return _fuzz.ratio(a, b) / 100.0


def _anilist_text_score(q: str, item: dict) -> float:
    qn = _normalizar_titulo(q)
    if not qn:
        return 0.0
    titulos    = anilist_titulos_desde_item(item)
    cand_norms = [c for c in (_normalizar_titulo(t) for t in titulos if t) if c]
    if not cand_norms:
        return 0.0
    best_ratio = max(_ratio(qn, c) for c in cand_norms)
    qt         = set(qn.split())
    bonus      = max(len(qt & set(c.split())) / max(1, len(qt)) for c in cand_norms)
    return best_ratio * 0.75 + bonus * 0.25


def _titulo_principal(item: dict, consulta: str) -> str:
    title = item.get("title") or {}
    return (
        title.get("romaji")
        or title.get("english")
        or title.get("userPreferred")
        or title.get("native")
        or consulta
    )


def anilist_buscar_titulo(consulta: str) -> Tuple[str, Optional[dict], bool, float]:
    """Busca un anime por texto en AniList. Devuelve (canon, item, confiable, ts1)
    -- misma forma que jikan_resolver_titulo() (jikan.py) para que conectar
    un fallback Jikan -> AniList sea un simple intercambio de funcion.

    canon: titulo elegido (romaji preferido; english/userPreferred/native
    como respaldo, o la consulta original si no hubo resultados).
    item: dict crudo del Media de AniList devuelto por la busqueda (None
    si no hubo resultados).
    confiable: True si hay un unico resultado, o si el mejor candidato
    tiene alta similitud de texto con margen claro sobre el segundo
    (mismo criterio que jikan_resolver_titulo: ts1 >= 0.72 y diff >= 0.08).
    ts1: score de similitud de texto contra el mejor candidato (0.0 si no
    hubo resultados).
    """
    consulta = (consulta or "").strip()
    if not consulta:
        return consulta, None, False, 0.0

    resultados = _anilist_buscar_media(consulta)
    if not resultados:
        return consulta, None, False, 0.0

    if len(resultados) == 1:
        it = resultados[0]
        return _titulo_principal(it, consulta), it, True, 1.0

    ordenados = sorted(
        resultados,
        key=lambda it: _anilist_text_score(consulta, it),
        reverse=True,
    )
    mejor     = ordenados[0]
    segundo   = ordenados[1] if len(ordenados) > 1 else None
    ts1       = _anilist_text_score(consulta, mejor)
    ts2       = _anilist_text_score(consulta, segundo) if segundo else 0.0
    confiable = ts1 >= 0.72 and (ts1 - ts2) >= 0.08

    logger.debug(
        f"  - anilist_score: q={consulta!r}, n={len(resultados)}, "
        f"ts1={ts1:.3f}, ts2={ts2:.3f}, diff={ts1 - ts2:.3f}, confiable={confiable}"
    )

    return _titulo_principal(mejor, consulta), mejor, confiable, ts1
