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
_RUIDO_TOKENS: frozenset = frozenset({
    "1080p", "2160p", "720p", "480p", "4k", "8k",
    "10bit", "10-bit", "8bit", "hi10p", "hi10",
    "x264", "x265", "hevc", "av1", "h264", "h265",
    "webrip", "webdl", "web-dl", "bdrip", "bluray", "blu-ray", "dvd",
    "hdr", "hdr10", "dv", "atmos",
    "aac", "flac", "opus", "eac3", "ac3", "ddp",
    "jpn", "eng", "spa", "lat", "sub", "subs", "msubs", "multisub", "multi",
    "uncensored", "censored", "repack", "proper", "remux",
    "amzn", "nf", "dsnp", "adn",
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
    r"|aac\d*\.?\d*|flac|opus|eac3|ac3|ddp?\d*\.?\d*"
    r"|jpn|eng|spa|lat|msubs?|multisub|multi|dual[- ]?audio"
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
    r"|aac|flac|opus|eac3|ac3|ddp"
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


def _parsed_dict_a_campos(d: dict) -> tuple:
    """Extrae (titulo, temporada, episodio) de un dict de aniparse/anitopy."""
    titulo    = (d.get("anime_title") or "").strip()
    temporada = _safe_int(d.get("anime_season"))
    episodio  = _safe_int(d.get("episode_number"))
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

    titulo_a, temp_a, ep_a = _parsed_dict_a_campos(a) if a else ("", None, None)
    titulo_b, temp_b, ep_b = _parsed_dict_a_campos(b) if b else ("", None, None)

    # Normalizar puntos/underscores en títulos de scene releases (e.g. 'HELL.MODE.The...')
    titulo_a = _normalizar_titulo_parser(titulo_a)
    titulo_b = _normalizar_titulo_parser(titulo_b)

    # Merge consciente de temporada: si una biblioteca detectó season pero la otra
    # dejó el número pegado al título (e.g. "Kingdom 5" cuando season=5), limpiarlo.
    temp_combinada = temp_a if temp_a is not None else temp_b
    if temp_combinada:
        titulo_a = re.sub(rf"\s{temp_combinada}$", "", titulo_a).strip()
        titulo_b = re.sub(rf"\s{temp_combinada}$", "", titulo_b).strip()

    # Elegir el mejor título por score de limpieza
    score_a = _score_titulo(titulo_a)
    score_b = _score_titulo(titulo_b)

    if score_b > score_a:
        titulo_elegido = titulo_b
        fuente         = "anitopy" if b and not a else "aniparse+anitopy"
    else:
        titulo_elegido = titulo_a
        fuente         = "aniparse" if a and not b else "aniparse+anitopy"

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
