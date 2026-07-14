"""Integracion con AniList: busqueda de anime por texto (fallback cuando
Jikan agota reintentos, ver ResolverWorker en gui/workers.py) y lookup de
titulo por ID (usado por trace_moe.py para resolver el titulo de un
anilist_id detectado por consenso de fotogramas).

anilist_titulo_por_id vivia en trace_moe.py; se movio aqui para
consolidar toda la logica de AniList en un solo modulo. trace_moe.py la
importa de vuelta desde aqui.
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


_ANILIST_RELATIONS_QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    relations {
      edges {
        relationType
        node {
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
        }
      }
    }
  }
}
"""


@_reintento_http
def _anilist_relaciones(anilist_id: int) -> List[dict]:
    """Devuelve los edges de relations (relationType + node completo) de
    un Media, analogo a jikan_relaciones (jikan.py) pero en un solo
    round-trip: a diferencia de Jikan (cuyas entries de relations son
    stubs livianos que exigen una segunda consulta de detalle), el node
    de AniList ya trae title/episodes/format/status completos."""
    clave  = f"anilist_rel:{anilist_id}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    r = _http.post(
        ANILIST_GRAPHQL,
        json={"query": _ANILIST_RELATIONS_QUERY, "variables": {"id": anilist_id}},
    )
    r.raise_for_status()
    data   = r.json() or {}
    edges  = (((data.get("data") or {}).get("Media") or {}).get("relations") or {}).get("edges") or []
    _API_CACHE.set(clave, edges, expire=_TTL_API_DAYS * 86400)
    return edges


def anilist_avanzar_a_secuela(actual: dict, contexto: str = "") -> dict:
    """Un paso en la cadena de secuelas de AniList. Analogo a
    _avanzar_a_secuela (jikan.py): mismo nivel de confianza (toma el
    primer relationType == SEQUEL sin validar de mas -- relationType ya
    distingue SEQUEL de SPIN_OFF/SIDE_STORY, igual que el campo
    'relation' de Jikan), pero sin el segundo round-trip que Jikan
    necesita para el detalle del siguiente eslabon.

    contexto: frase libre que se agrega al mensaje de error para indicar
    en que paso de la cadena ocurrio el fallo (mismo patron que
    _avanzar_a_secuela)."""
    anilist_id = int(actual["id"])
    edges      = _anilist_relaciones(anilist_id)

    secuela = None
    for edge in edges:
        if (edge.get("relationType") or "").upper() == "SEQUEL":
            secuela = edge.get("node")
            break

    if not secuela:
        ctx = f" — {contexto}" if contexto else ""
        raise RuntimeError(
            f"AniList: sin secuela para '{_titulo_principal(actual, '')}'"
            f" (anilist_id={anilist_id}){ctx}."
        )
    return secuela


def anilist_resolver_temporada_por_sequel(elemento_base: dict, temporada: int) -> dict:
    """Analogo a jikan_resolver_temporada_por_sequel (jikan.py): navega
    la cadena de secuelas de AniList hasta el numero de temporada pedido.

    Puramente mecanica, igual que su par de Jikan -- no valida el titulo
    resultante. La decision de aceptar o rechazar el canon (via
    _aceptar_canon_sin_perder_tokens) es responsabilidad del caller
    (gui/workers.py), en el mismo punto donde ya se aplica para Jikan,
    para que el mensaje de rechazo sea identico sin importar la fuente."""
    if not elemento_base or not temporada or temporada <= 1:
        return elemento_base

    actual = elemento_base
    for paso in range(temporada - 1):
        actual = anilist_avanzar_a_secuela(actual, contexto=f"paso {paso + 1}/{temporada - 1}")
    return actual


def anilist_navegar_por_episodio(
    base_entry:   dict,
    episodio_abs: int,
) -> Tuple[dict, int, int]:
    """Analogo a jikan_navegar_por_episodio (jikan.py): navega la cadena
    de secuelas de AniList para ubicar episodio_abs (numeracion global
    del archivo, sin temporada) en la temporada y episodio relativo
    correctos. El conteo de episodios de cada eslabon ya viene incluido
    en el node de relations (ver _anilist_relaciones) -- no hace falta
    una consulta separada por candidato.

    Devuelve (entry_anilist, episodio_relativo, numero_temporada).
    Lanza RuntimeError si la cadena esta incompleta o falta el conteo de
    episodios en algun eslabon (ambos son datos reales de AniList).

    Puramente mecanica, igual que jikan_navegar_por_episodio -- no valida
    el titulo resultante (Jikan tampoco lo hace en este punto hoy; la
    unica proteccion existente en ese camino es _aplicar_canon, generica
    y aplicada por el caller al final del flujo)."""
    actual      = base_entry
    temp_num    = 1
    ep_restante = episodio_abs

    while True:
        eps = actual.get("episodes")
        try:
            eps = int(eps) if eps else 0
        except (TypeError, ValueError):
            eps = 0

        if eps <= 0:
            raise RuntimeError(
                f"AniList: '{_titulo_principal(actual, '')}' (anilist_id={actual.get('id')}) "
                "no tiene conteo de episodios — imposible detectar temporada por conteo."
            )

        if ep_restante <= eps:
            return actual, ep_restante, temp_num

        ep_restante -= eps
        temp_num    += 1

        actual = anilist_avanzar_a_secuela(actual, contexto=f"hacia temporada {temp_num}")


@_reintento_http
def anilist_titulo_por_id(anilist_id: int) -> Optional[str]:
    """Obtiene el título romaji de un anime por su ID exacto de AniList."""
    clave  = f"anilist_id:{anilist_id}"
    cached = _API_CACHE.get(clave)
    if cached is not None:
        return cached
    query = """
    query ($id: Int) {
      Media(id: $id, type: ANIME) {
        title { romaji }
      }
    }
    """
    r = _http.post(
        ANILIST_GRAPHQL,
        json={"query": query, "variables": {"id": anilist_id}},
    )
    r.raise_for_status()
    titulo = (
        ((r.json().get("data") or {}).get("Media") or {})
        .get("title", {})
        .get("romaji")
    )
    if titulo:
        _API_CACHE.set(clave, titulo, expire=_TTL_API_DAYS * 86400)
    return titulo or None
