"""Integracion con Jikan/MAL: busqueda de anime, resolucion de titulo,
navegacion de la cadena de secuelas y verificacion cruzada de titulos.
Movido sin cambios desde chapterizen.py (monolito original, v0.0.7)."""
import re
from typing import Optional, Tuple, List

from loguru import logger
from rapidfuzz import fuzz as _fuzz

from .config import _http, _reintento_http, _API_CACHE, _TTL_API_DAYS, JIKAN_ANIME, JIKAN_REL
from .ffmpeg_utils import log_clv
from .parsing import parsear_nombre_archivo


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

def _avanzar_a_secuela(actual: dict, contexto: str = "") -> dict:
    """Un paso en la cadena de secuelas de Jikan. Lanza RuntimeError si no hay secuela.

    contexto — frase libre que se añade al mensaje de error para precisar
    en qué punto de la cadena ocurrió el fallo. Cada caller pasa algo
    distinto: jikan_resolver_temporada_por_sequel envía "paso N/total" y
    jikan_navegar_por_episodio envía "hacia temporada N", de modo que el
    error final combina "qué anime/mal_id falló" (siempre presente) con
    "en qué contexto" (específico del caller)."""
    relaciones = jikan_relaciones(int(actual["mal_id"]))
    secuela    = None
    for rel in relaciones:
        if (rel.get("relation") or "").casefold() == "sequel":
            entries = rel.get("entry") or []
            if entries:
                secuela = entries[0]
            break

    if not secuela:
        ctx = f" — {contexto}" if contexto else ""
        raise RuntimeError(
            f"Jikan: sin secuela para '{actual.get('title')}'"
            f" (mal_id={actual.get('mal_id')}){ctx}."
        )

    id_secuela = int(secuela["mal_id"])

    @_reintento_http
    def _get():
        r = _http.get(f"{JIKAN_ANIME}/{id_secuela}")
        r.raise_for_status()
        return r.json()

    data = (_get() or {}).get("data")
    if not data:
        raise RuntimeError(f"Jikan: la secuela mal_id={id_secuela} no devolvió datos.")
    return data


def jikan_resolver_temporada_por_sequel(elemento_base: dict, temporada: int) -> dict:
    if not elemento_base or not temporada or temporada <= 1:
        return elemento_base

    actual = elemento_base
    for paso in range(temporada - 1):
        actual = _avanzar_a_secuela(actual, contexto=f"paso {paso + 1}/{temporada - 1}")
    return actual

def jikan_navegar_por_episodio(
    base_entry:   dict,
    episodio_abs: int,
) -> Tuple[dict, int, int]:
    """
    Navega la cadena de secuelas para ubicar `episodio_abs` (numeración
    global del archivo, sin temporada) en la temporada y episodio relativo
    correctos.

    Ejemplo: base_entry=S1 (12 ep), episodio_abs=15 → (entrada_S2, 3, 2).

    Devuelve (entry_jikan, episodio_relativo, numero_temporada).
    Lanza RuntimeError si la cadena está incompleta o falta el conteo de
    episodios en algún eslabón (ambos son datos reales de MAL/Jikan).
    """
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
                f"Jikan: '{actual.get('title')}' (mal_id={actual.get('mal_id')}) "
                "no tiene conteo de episodios — imposible detectar temporada por conteo."
            )

        if ep_restante <= eps:
            return actual, ep_restante, temp_num

        ep_restante -= eps
        temp_num    += 1

        actual = _avanzar_a_secuela(actual, contexto=f"hacia temporada {temp_num}")

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
    """
    Acepta el canon de Jikan solo si no descarta tokens significativos del título base.
    Permite que el canon añada información ("Bleach" → "Bleach (2004)") pero rechaza
    reemplazos que recorten el título original ("Sword Art Online" → "SAO Alicization")
    o que cambien de temporada ("Bleach" → "Bleach: Sennen Kessen-hen").
    """
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

def jikan_resolver_titulo(q: str) -> Tuple[str, Optional[dict], bool, float]:
    """Devuelve (canon, item, confiable, ts1).
    En ResolverWorker.run() estas posiciones se reciben como
    (titulo_resuelto, picked_base, titulo_confiable, ts1_base) — mismos
    valores, distintos nombres de variable local.
    ts1 mide qué tan bien coincide el query con el primer resultado de
    Jikan — útil para desambiguación posterior."""
    q = (q or "").strip()
    if not q:
        return q, None, False, 0.0
    resultados = jikan_buscar_anime(q, limite=10)
    if not resultados:
        return q, None, False, 0.0
    if len(resultados) == 1:
        it   = resultados[0]
        main = (it.get("title") or "").strip() or q
        return main, it, True, 1.0
    ordenados = _jikan_rank(q, resultados)
    mejor     = ordenados[0]
    segundo   = ordenados[1] if len(ordenados) > 1 else None
    ts1       = _jikan_text_score(q, mejor)
    ts2       = _jikan_text_score(q, segundo) if segundo else 0.0
    confiable = ts1 >= 0.72 and (ts1 - ts2) >= 0.08
    log_clv(
        logger.debug, "jikan_score",
        q=q, n=len(resultados),
        ts1=round(ts1, 3), ts2=round(ts2, 3),
        diff=round(ts1 - ts2, 3), confiable=confiable,
    )
    main      = (mejor.get("title") or "").strip() or q
    return main, mejor, confiable, ts1

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

def _comparar_titulos_para_verificacion(titulo_a: str, titulo_b: str) -> Tuple[Optional[bool], str]:
    """
    Compara dos títulos (uno de Jikan, otro de AniList vía trace.moe) para
    determinar si representan la misma entrada de anime.

    Devuelve (resultado, motivo):
      (True,  "igualdad_exacta") — idénticos tras normalización
      (True,  "similitud_alta")  — fuzzy ratio ≥ 0.85 (transliteración, etc.)
      (False, "prefijo")         — entradas distintas de la misma franquicia
      (None,  "ninguno")         — no determinado

    Tres pasos, en orden:
    1. Igualdad exacta tras normalización → mismo contenido.
    2. Relación de prefijo: si uno es prefijo estricto del otro seguido de
       espacio, dos puntos o guion, es señal de entradas distintas de la misma
       franquicia (S1 vs S2, base vs Recap, etc.) — mismo punto ciego que generó
       el problema de Wistoria, por eso este paso va ANTES del fuzzy ratio.
    3. Fuzzy ratio ≥ 0.85 → mismo contenido (diferencias menores de transliteración).
    4. Ninguno resolvió con certeza → None.
       Abrir el picker es siempre más seguro que silenciar la divergencia: cuando
       ya tenemos datos de cross-verificación inconcluyentes, descartarlos en
       silencio sería peor que el "no confirmado" original que los ignoraba por
       completo.
    """
    a = _normalizar_titulo(titulo_a)
    b = _normalizar_titulo(titulo_b)

    if not a or not b:
        return None, "ninguno"

    if a == b:
        return True, "igualdad_exacta"

    # Relación de prefijo estricta: el título más corto es prefijo del largo
    # y el carácter siguiente es un separador (espacio, dos puntos, guion).
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if long_.startswith(short) and len(long_) > len(short) and long_[len(short)] in " :-(":
        return False, "prefijo"

    if _ratio(a, b) >= 0.85:
        return True, "similitud_alta"

    return None, "ninguno"
