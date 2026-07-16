"""Parsing de nombres de archivo de anime (aniparse / anitopy / regex).
Movido sin cambios desde chapterizen.py (monolito original, v0.0.7)."""
import re
from pathlib import Path
from typing import Optional, List

from loguru import logger

from .modelos import ParsedAnime


# Parsers de nombres de archivo de anime (aniparse principal, anitopy fallback)
try:
    import aniparse as _aniparse
    _ANIPARSE_OK = True
except ImportError:
    _aniparse    = None
    _ANIPARSE_OK = False

try:
    import anitopy as _anitopy
    _ANITOPY_OK = True
except ImportError:
    _anitopy    = None
    _ANITOPY_OK = False

# ── Tokens de ruido para detección rápida y predecible ──────────────────────
# Set de tokens exactos (lowercase) que son tags de release, nunca títulos.
# Complementado por _RE_RUIDO_TITULO para casos compuestos/pegados (e.g. "AAC2.0").
# Lista basada en investigación real de tags de Nyaa.si -- algunos
# candidatos (ej. "WEB" suelto) se excluyeron deliberadamente por riesgo
# real de falso positivo confirmado con datos de AniList; ver
# tests/test_parsing.py (TestEsTokenRuidoAmpliacion) antes de "corregir"
# una ausencia pensando que falta agregarla.
_RUIDO_TOKENS: frozenset = frozenset({
    "1080p", "2160p", "720p", "480p", "4k", "8k",
    "10bit", "10-bit", "8bit", "hi10p", "hi10",
    "x264", "x265", "hevc", "av1", "h264", "h265", "avc",
    "webrip", "webdl", "web-dl", "bdrip", "bluray", "blu-ray", "dvd", "bd",
    "hdr", "hdr10", "dv", "atmos",
    "aac", "flac", "opus", "eac3", "ac3", "ddp", "srt",
    "jpn", "eng", "spa", "lat", "pt-br", "vostfr",
    "sub", "subs", "msubs", "multisub", "multisubs", "multi",
    "uncensored", "censored", "repack", "proper", "remux",
    "amzn", "nf", "dsnp", "adn", "bili", "tver", "ytb",
})

# Palabras semánticamente débiles: títulos de 1-2 tokens formados solo por estas
# palabras casi nunca son nombres de serie válidos — son artefactos de parsing fallido.
# Se usa conjunto pequeño y deliberado para evitar falsos positivos.
# Definida antes que _TITULOS_INVALIDOS porque este la incluye por unión (derivación intencional).
_PALABRAS_DEBILES: frozenset = frozenset({
    "final", "movie", "film", "part", "episode", "ep", "special",
})

# Títulos que son válidos como identificadores de anime pero nunca como nombre de serie.
# Evita que "OP1", "ED2", "OVA" pasen como títulos usables.
_TITULOS_INVALIDOS: frozenset = _PALABRAS_DEBILES | frozenset({
    "op", "ed", "ova", "pv", "cm", "nced", "ncop", "preview", "trailer",
})

# Ruido residual — para casos compuestos/pegados que el set no cubre (e.g. "AAC2.0", "HEVC10bit")
_RE_RUIDO_TITULO = re.compile(
    r"(?i)\b(2160p|1080p|720p|480p|4k|8k"
    r"|10bit|10-bit|8bit|hi10p?"
    r"|x264|x265|hevc|av1|h\.?26[45]"
    r"|web[- ]?(?:dl|rip)|webrip|bdrip|blu[- ]?ray|bluray|dvd"
    r"|hdr10\+?|hdr|dolby\s*vision|\bdv\b|atmos"
    r"|aac\d*\.?\d*|flac|opus|eac3|ac3|ddp?\d*\.?\d*|srt"
    r"|jpn|eng|spa|lat|msubs?|multiple[-_ ]?subtitles?|multi[-_ ]?subs?|multi|dual[-_ ]?audio"
    r"|uncensored|censored|repack|proper|remux"
    r"|amzn|\bcr\b|\bnf\b|dsnp|adn)\b"
    r"|[\[\(\{][^\]\)\}]{0,90}[\]\)\}]"  # bloques entre brackets
)


# Regex anclado al inicio del token — captura tags compuestos/pegados que el \b
# de _RE_RUIDO_TITULO no detecta. Razón: en Python, \b solo marca frontera entre
# \w y \W; letra y dígito son ambos \w, así que no hay \b entre 'C' y '1' en
# 'HEVC10bit'. El ancla ^ cubre ese caso sin necesitar \b al inicio del token.
_RE_RUIDO_TOKEN_INICIO = re.compile(
    r"(?i)^(2160p|1080p|720p|480p|4k|8k"
    r"|10bit|10-bit|8bit|hi10p?"
    r"|x264|x265|hevc|av1|h\.?26[45]"
    r"|web[- ]?(?:dl|rip)|webrip|bdrip|blu[- ]?ray|bluray|dvd"
    r"|hdr10\+?|hdr|dolby"
    r"|aac|flac|opus|eac3|ac3|ddp|srt"
    r"|jpn|eng|spa|lat)"
)

def _es_token_ruido(token: str) -> bool:
    """True si el token es un tag de release.
    1. Lookup en set exacto (rápido)
    2. Regex con \\b para tags normales (e.g. 'AAC2.0')
    3. Regex anclado al inicio para tags pegados (e.g. 'HEVC10bit')
    """
    t = token.lower()
    return (
        t in _RUIDO_TOKENS
        or bool(_RE_RUIDO_TITULO.search(token))
        or bool(_RE_RUIDO_TOKEN_INICIO.match(token))
    )


def _score_titulo(title: str) -> int:
    """
    Evalúa qué tan limpio está un título candidato.
    Solo se usa para COMPARAR entre aniparse y anitopy — no como filtro de calidad.
    El filtro de calidad lo hace _titulo_es_usable().

    Nota: no penaliza longitud corta — títulos como '86' son válidos.
    """
    if not title:
        return -999
    score = 0
    # Bonificar longitud razonable (no penalizar cortos — '86' es un título válido)
    if len(title) <= 80:
        score += 2
    # Penalizar si algún token es ruido técnico
    if any(_es_token_ruido(t) for t in title.split()):
        score -= 2
    # Penalizar hashes hexadecimales largos (e.g. "F4FB217B" en el nombre)
    if re.search(r"\b[0-9A-Fa-f]{6,}\b", title):
        score -= 3
    return score



def _titulo_es_usable(title: str) -> bool:
    """
    Determina si un título es lo suficientemente limpio para enviarse a Jikan.
    Conservador a propósito: solo rechaza lo claramente inutilizable.

    Permite: "86", "Air", "K-On!", "Mob Psycho 100", "Golden Kamuy Final Season"
    Rechaza: "1080p AAC x264", "F4FB217B", "OVA", "Final", "Movie", "Part 1"
    """
    if not title or len(title) < 2:
        return False

    # Rechazar abreviaciones que nunca son títulos de serie
    if title.strip().lower() in _TITULOS_INVALIDOS:
        return False

    # Rechazar títulos semánticamente débiles: 1-2 palabras formadas solo por
    # términos genéricos de anime (e.g. "Final", "Movie", "Part 1").
    # Títulos largos que los contengan ("Golden Kamuy Final Season") pasan sin problema.
    tokens_lower = [t.lower() for t in title.split()]
    # Requiere al menos un token en _PALABRAS_DEBILES para no bloquear números
    # solos como '86' (título válido) o '100' (Mob Psycho 100).
    if (
        len(tokens_lower) <= 2
        and any(t in _PALABRAS_DEBILES for t in tokens_lower)
        and all(t in _PALABRAS_DEBILES or t.isdigit() for t in tokens_lower)
    ):
        return False

    # Rechazar si el título completo es un hash hexadecimal suelto (e.g. "F4FB217B")
    if re.fullmatch(r"[0-9A-Fa-f]{6,}", title.strip()):
        return False

    # Contar tokens de ruido y calcular ratio
    tokens = title.split()
    ruido_count = sum(1 for t in tokens if _es_token_ruido(t))
    ratio_ruido = ruido_count / len(tokens)

    # Rechazar si 60%+ de los tokens son ruido técnico (umbral explícito, no magia)
    if ratio_ruido >= 0.6:
        return False

    return True


def _titulo_tiene_artefacto_pegado(title: str) -> bool:
    """
    Detecta un dígito y una letra pegados sin separador en el título --
    señal de que un tag técnico (ej. "S01E19") quedó sin limpiar del todo
    tras normalizar (confirmado con archivo real: "Tojima Wants to Be a
    Kamen Rider S01E19.I Have No Regrets...", ver docs/KNOWN_LIMITATIONS.md).
    _normalizar_titulo_parser solo convierte a espacio los puntos entre DOS
    LETRAS -- un punto entre un dígito y una letra (ej. "19.I") no matchea
    ese regex y queda pegado.

    Deliberadamente SEPARADA de _titulo_es_usable, no un chequeo agregado
    ahí adentro: _titulo_es_usable la usa también parsear_nombre_archivo()
    para decidir internamente si confiar en el título elegido o caer a
    _fallback_regex. Si este chequeo viviera ahí, un título con este
    patrón desviaría a parsear_nombre_archivo() hacia su propio regex de
    respaldo -- que puede producir un título igual de imperfecto pero
    SIN ningún dígito pegado (confirmado: le deja pegado el tag de
    plataforma/release en vez del de episodio), evitando que este mismo
    chequeo se dispare de nuevo más adelante y frustrando el propósito
    real, que es activar la identificación por fotogramas (trace.moe) en
    ResolverWorker.run() -- el único lugar que debe llamar a esta función.
    """
    return bool(re.search(r"\d[a-zA-Z]|[a-zA-Z]\d", title))


def _safe_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def _parse_con_aniparse(stem: str) -> Optional[dict]:
    if not _ANIPARSE_OK:
        return None
    try:
        result = _aniparse.parse(stem)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _parse_con_anitopy(stem: str) -> Optional[dict]:
    if not _ANITOPY_OK:
        return None
    try:
        result = _anitopy.parse(stem)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _campos_desde_anitopy(d: dict) -> tuple:
    """Extrae (titulo, temporada, episodio) del schema plano de anitopy
    (anime_title / anime_season / episode_number a nivel raiz)."""
    titulo    = (d.get("anime_title") or "").strip()
    temporada = _safe_int(d.get("anime_season"))
    episodio  = _safe_int(d.get("episode_number"))
    return titulo, temporada, episodio


def _campos_desde_aniparse(d: dict) -> tuple:
    """Extrae (titulo, temporada, episodio) del schema anidado de aniparse
    (series[0].title / season[0].number / episode[0].number) -- distinto
    del schema plano de anitopy. Antes de este fix, _parsed_dict_a_campos
    usaba las claves de anitopy para leer TAMBIEN el dict de aniparse, asi
    que aniparse siempre devolvia ("", None, None) sin excepcion (ver
    docs/TECH_DEBT.md)."""
    series = d.get("series") or []
    if not series:
        return "", None, None
    s = series[0]
    titulo    = (s.get("title") or "").strip()
    temporada = _safe_int(s["season"][0].get("number")) if s.get("season") else None
    episodio  = _safe_int(s["episode"][0].get("number")) if s.get("episode") else None
    return titulo, temporada, episodio


def _fallback_regex(stem: str) -> "ParsedAnime":
    """
    Parser de último recurso basado en regex.
    Mantiene compatibilidad con nombres que las bibliotecas no manejen.
    """
    # Quitar bloques entre brackets (grupo, tags, hash)
    s = re.sub(r"[\[\(\{][^\]\)\}]{0,90}[\]\)\}]", " ", stem)
    # Quitar tag de release al final (e.g. "-SubsPlease")
    s = re.sub(r"-[A-Za-z0-9]+$", " ", s)

    # Temporada textual
    temporada: Optional[int] = None
    for pat, grp in [
        (r"\b(\d+)\s*(?:st|nd|rd|th)\s*season\b", 1),
        (r"\bseason[_\s\-]*(\d+)\b", 1),
        (r"(?:^|[\s._-])s(\d{1,2})(?:$|[\s._-])", 1),
    ]:
        m = re.search(pat, s, re.I)
        if m:
            temporada = _safe_int(m.group(grp))
            break

    # Episodio
    episodio: Optional[int] = None
    for pat, grps in [
        (r"(?i)\bS(\d{1,2})E(\d{1,3})(?:v\d+)?\b", (1, 2)),
        (r"(?i)\b(\d{1,2})x(\d{1,3})\b",             (1, 2)),
    ]:
        m = re.search(pat, s)
        if m:
            temporada = temporada or _safe_int(m.group(grps[0]))
            episodio  = _safe_int(m.group(grps[1]))
            break
    if episodio is None:
        m = re.search(r"(?i)\b(?:EP?|E)\s*(\d{1,3})(?:v\d+)?\b", s)
        if m:
            episodio = _safe_int(m.group(1))
    if episodio is None:
        m = re.search(r"-\s+(\d{1,3})(?:v\d+)?(?:\s|$|\[|\()", s)
        if m:
            ep = _safe_int(m.group(1))
            if ep and 1 <= ep <= 399:
                episodio = ep

    # Limpiar título: quitar episodio, ruido técnico, separadores
    titulo = s
    titulo = re.sub(r"(?i)\bS\d{1,2}E\d{1,3}(?:v\d+)?\b", " ", titulo)
    titulo = re.sub(r"(?i)\b(?:EP?|E)\d{1,3}(?:v\d+)?\b",   " ", titulo)
    titulo = re.sub(r"-\s*\d{1,3}(?:v\d+)?(?:\s|$)",          " ", titulo)
    titulo = _RE_RUIDO_TITULO.sub(" ", titulo)
    titulo = re.sub(r"[._]+", " ", titulo)
    titulo = re.sub(r"\s+", " ", titulo).strip(" -_.")

    logger.debug(f"[parsing] fallback regex para {stem!r} → título={titulo!r}")
    return ParsedAnime(titulo=titulo, temporada=temporada, episodio=episodio, fuente="fallback")


def _normalizar_titulo_parser(titulo: str) -> str:
    """
    Normaliza el título que devuelve aniparse/anitopy antes de evaluarlo.
    Convierte puntos y underscores entre letras en espacios (scene releases)
    y colapsa espacios múltiples.
    No afecta números ni signos de puntuación legítimos.
    """
    # Puntos/underscores entre letras → espacio (e.g. 'HELL.MODE.The' → 'HELL MODE The')
    titulo = re.sub(r"(?<=[a-zA-Z])[._]+(?=[a-zA-Z])", " ", titulo)
    # Colapsar espacios múltiples
    titulo = re.sub(r"\s+", " ", titulo)
    return titulo.strip()


def parsear_nombre_archivo(ruta_video: str) -> "ParsedAnime":
    """
    Punto de entrada único para parsear nombres de archivo de anime.

    Estrategia:
      1. aniparse  (principal — mejor con nombres modernos)
      2. anitopy   (respaldo  — más probado en variedad)
      3. merge     de ambos si los dos producen resultado
      4. regex     fallback si las bibliotecas no están instaladas o fallan
    """
    stem = Path(ruta_video).stem

    a = _parse_con_aniparse(stem)
    b = _parse_con_anitopy(stem)

    if a is None and b is None:
        return _fallback_regex(stem)

    titulo_a, temp_a, ep_a = _campos_desde_aniparse(a) if a else ("", None, None)
    titulo_b, temp_b, ep_b = _campos_desde_anitopy(b) if b else ("", None, None)

    # Normalizar puntos/underscores en títulos de scene releases (e.g. 'HELL.MODE.The...')
    titulo_a = _normalizar_titulo_parser(titulo_a)
    titulo_b = _normalizar_titulo_parser(titulo_b)

    # No confiar en el episodio de aniparse si su título quedó vacío --
    # señal de que aniparse no encontró texto real de título y solo
    # interpretó números sueltos del nombre de archivo como episodio
    # (confirmado con el caso sintético "12345.mkv": aniparse interpreta
    # el número completo como episodio con su propia confianza en 0.0;
    # ver docs/TECH_DEBT.md). No afecta ningún archivo real evaluado
    # donde aniparse ya acierta el episodio -- en esos casos su título
    # nunca queda vacío.
    if not titulo_a:
        ep_a = None

    # Merge consciente de temporada: si una biblioteca detectó season pero la otra
    # dejó el número pegado al título (e.g. "Kingdom 5" cuando season=5), limpiarlo.
    # Desconfiar de la temporada de aniparse si coincide con el episodio que
    # leyó anitopy -- señal de que aniparse confundió un dígito de episodio
    # con uno de temporada (confirmado con datos reales: "Golden Kamuy Final
    # Season - 07" -- aniparse lee temporada=7 en vez de episodio=7; ver
    # docs/TECH_DEBT.md). En ese caso se prefiere la temporada de anitopy.
    if temp_a is not None and ep_b is not None and temp_a == ep_b:
        temp_combinada = temp_b
    else:
        temp_combinada = temp_a if temp_a is not None else temp_b
    if temp_combinada:
        titulo_a = re.sub(rf"\s{temp_combinada}$", "", titulo_a).strip()
        titulo_b = re.sub(rf"\s{temp_combinada}$", "", titulo_b).strip()

    # Elegir el mejor título por score de limpieza. Desempate invertido
    # hacia anitopy: aniparse solo gana si supera estrictamente el score,
    # no en empate -- confirmado con datos reales que un empate hoy
    # favorecía a aniparse por defecto, y aniparse trunca palabras del
    # título en algunos casos (ej. serie "Does It Count If You Lose Your
    # Innocence to an Android" -- aniparse corta "Android"; ver
    # docs/TECH_DEBT.md). anitopy es la fuente que venía siendo confiable.
    score_a = _score_titulo(titulo_a)
    score_b = _score_titulo(titulo_b)

    if score_a > score_b:
        titulo_elegido = titulo_a
        fuente         = "aniparse" if a and not b else "aniparse+anitopy"
    else:
        titulo_elegido = titulo_b
        fuente         = "anitopy" if b and not a else "aniparse+anitopy"

    # Si el título elegido sigue teniendo ruido (score < 1), caer a regex.
    # Umbral 1 en lugar de 0 para capturar falsos positivos como "Frieren 1080p".
    if not _titulo_es_usable(titulo_elegido):
        logger.debug(f"[parsing] título no usable ({titulo_elegido!r}), fallback a regex")
        return _fallback_regex(stem)

    # Episodio: primer valor no-None gana (aniparse tiene prioridad)
    temporada = temp_combinada
    episodio  = ep_a if ep_a is not None else ep_b

    resultado = ParsedAnime(
        titulo=titulo_elegido,
        temporada=temporada,
        episodio=episodio,
        fuente=fuente,
    )
    logger.debug(
        f"[parsing] {Path(ruta_video).name!r} → "
        f"aniparse={titulo_a!r} | anitopy={titulo_b!r} → "
        f"final={resultado.titulo!r} "
        f"(fuente={resultado.fuente}, T={resultado.temporada}, E={resultado.episodio})"
    )
    return resultado


# Wrappers de compatibilidad — mantienen la firma anterior intacta
# para no reescribir el ResolverWorker de golpe.

def quitar_sufijo_episodio(s: str) -> str:
    """Quita un sufijo ' - NN' de un título canónico (no de nombre de archivo)."""
    return re.sub(r"(?i)\s*-\s*\d{1,3}(?:v\d+)?\s*$", "", (s or "").strip())

def quitar_marcador_temporada(s: str) -> str:
    """Quita marcadores de temporada textual de un título canónico."""
    x = (s or "").strip()
    x = re.sub(r"(?i)\b(\d+)\s*(st|nd|rd|th)\s*season\b", "", x)
    x = re.sub(r"(?i)\bseason[_\s\-]*\d+\b", "", x)
    x = re.sub(r"\s+", " ", x).strip(" -_:")
    return x

def inferir_consulta_desde_nombre_archivo(ruta_video: str) -> str:
    return parsear_nombre_archivo(ruta_video).titulo

def _extraer_temporada_desde_slug_o_nombre(s: str) -> Optional[int]:
    if not s:
        return None
    x = s.casefold()
    for pat in [
        r"\b(\d+)(?:st|nd|rd|th)_season\b",
        r"\bseason[_\s\-]*(\d+)\b",
        r"(?:^|[_\-\s])s(\d+)(?:$|[_\-\s])",
        r"\b(\d+)(?:st|nd|rd|th)\b",
    ]:
        m = re.search(pat, x)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def _preferir_resultados_por_temporada(
    resultados: List[dict],
    temporada:  Optional[int],
) -> List[dict]:
    if not resultados or not temporada or temporada <= 1:
        return resultados

    def temporada_item(it: dict) -> Optional[int]:
        slug = it.get("slug") or ""
        name = (it.get("name") or "").strip()
        t    = _extraer_temporada_desde_slug_o_nombre(slug)
        return t if t is not None else _extraer_temporada_desde_slug_o_nombre(name)

    exactos, desconocidos, otros = [], [], []
    for it in resultados:
        t = temporada_item(it)
        if t is None:        desconocidos.append(it)
        elif t == temporada: exactos.append(it)
        else:                otros.append(it)

    if exactos:      return exactos + desconocidos + otros
    if desconocidos: return desconocidos + otros
    return sorted(resultados, key=lambda it: (
        abs(temporada_item(it) - temporada) if temporada_item(it) is not None else 999
    ))
