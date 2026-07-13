"""Identificacion de anime via trace.moe (busqueda por fotogramas,
consenso por lotes) y resolucion de titulo via AniList. Movido sin
cambios desde chapterizen.py (monolito original, v0.0.7)."""
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from .config import _http, _reintento_http, TRACE_ENDPOINT
from .modelos import AnimeDetectado
from .parsing import parsear_nombre_archivo
from .anilist import anilist_titulo_por_id


@_reintento_http
def trace_buscar_por_bytes(img_bytes: bytes) -> dict:
    """
    Consulta trace.moe subiendo la imagen directamente con POST multipart.
    Más rápido que el flujo anterior (Litterbox → GET) porque elimina el
    round-trip extra de un host intermediario. Re-subir la misma imagen
    en un reintento es inocuo, así que el POST completo puede reintentarse.
    """
    r = _http.post(
        TRACE_ENDPOINT,
        files={"image": ("frame.jpg", img_bytes, "image/jpeg")},
    )
    r.raise_for_status()
    return r.json()

_TRACE_UMBRAL_RAPIDO = 0.95  # similitud suficiente para no probar más fotogramas

def _consenso_trace(
    tops: List[dict],
    require_mayoria: bool = False,
) -> Tuple[Optional[dict], float]:
    """
    Elige el resultado más votado por anilist_id. Si varios frames coinciden
    en el mismo anime, ese gana aunque otro frame haya tenido mayor similitud.
    Con require_mayoria=True devuelve (None, -1.0) si el ganador no supera
    el 50% de los frames — evita salidas tempranas cuando los frames
    identifican animes distintos (p. ej. 1:1:1 con 3 frames).
    Devuelve (top1_dict, similitud_máxima_del_ganador).
    """
    if not tops:
        return None, -1.0
    conteo: Dict[int, int] = {}
    for t in tops:
        aid = t.get("anilist")
        if aid is not None:
            conteo[aid] = conteo.get(aid, 0) + 1
    if not conteo:
        mejor = max(tops, key=lambda t: float(t.get("similarity", 0)))
        return mejor, float(mejor.get("similarity", 0))
    id_ganador, max_votos = max(conteo.items(), key=lambda x: x[1])
    if require_mayoria and max_votos <= len(tops) / 2:
        return None, -1.0
    candidatos  = [t for t in tops if t.get("anilist") == id_ganador]
    mejor       = max(candidatos, key=lambda t: float(t.get("similarity", 0)))
    return mejor, float(mejor.get("similarity", 0))


def _consenso_episodio(
    candidatos: List[dict],
    log_fn=None,
) -> Optional[int]:
    """
    Vota por mayoría el episodio entre los frames del anime ganador.
    Frames sin campo 'episode' se excluyen (abstención, no voto nulo).
    Mayoría estricta (>50% de frames con episodio): devuelve ese episodio.
    Sin mayoría: loguea la dispersión con votos individuales y devuelve None.
    """
    _warn = log_fn if log_fn is not None else logger.warning

    votos: List[int] = []
    for t in candidatos:
        raw = t.get("episode")
        try:
            ep = int(raw) if raw is not None else None
        except (ValueError, TypeError):
            ep = None
        if ep is not None:
            votos.append(ep)

    if not votos:
        return None

    conteo: Dict[int, int] = {}
    for ep in votos:
        conteo[ep] = conteo.get(ep, 0) + 1

    ep_max, votos_max = max(conteo.items(), key=lambda x: x[1])

    if votos_max > len(votos) / 2:
        if len(conteo) == 1:
            _warn(f"  - Número de episodio confirmado: {ep_max}")
        else:
            votos_str = ", ".join(
                f"{v} {'indicó' if v == 1 else 'indicaron'} episodio {ep}"
                for ep, v in sorted(conteo.items(), key=lambda x: x[1], reverse=True)
            )
            _warn(f"  - Dispersión entre fotogramas: {votos_str}")
            _warn(f"  - Número de episodio confirmado (por mayoría): {ep_max}")
        return ep_max

    votos_str = ", ".join(
        f"{v} {'indicó' if v == 1 else 'indicaron'} episodio {ep}"
        for ep, v in sorted(conteo.items(), key=lambda x: x[1], reverse=True)
    )
    _warn(f"  - Dispersión entre fotogramas: {votos_str}")
    _warn(
        f"  - ⚠️ Sin coincidencia clara de episodio — "
        f"usando número del nombre de archivo"
    )
    return None


def identificar_anime_con_fotogramas(
    rutas_fotogramas: List[Path],
    log_fn=None,
) -> Tuple["AnimeDetectado", int]:
    """
    Consulta trace.moe en lotes: [centro], [±1], [±2], [±3], [±4].
    Requiere al menos 2 lotes (3 frames) antes de activar la salida temprana,
    y usa voto mayoritario por anilist_id para evitar falsos positivos de un
    único frame que coincida accidentalmente con otro anime.
    Devuelve (AnimeDetectado, n_fotogramas_enviados).
    """
    todos_tops: List[dict] = []
    n_usados = 0

    # Lotes: [centro], [centro-1, centro+1], [centro-2, centro+2], …
    lotes: List[List[Path]] = []
    if rutas_fotogramas:
        lotes.append([rutas_fotogramas[0]])
        i = 1
        while i < len(rutas_fotogramas):
            lotes.append(list(rutas_fotogramas[i:i + 2]))
            i += 2

    _serie_resuelta = False

    for num_lote, lote in enumerate(lotes):
        tops_pre = len(todos_tops)
        n_pre    = n_usados

        with ThreadPoolExecutor(max_workers=len(lote)) as ex:
            futuros = {ex.submit(trace_buscar_por_bytes, fp.read_bytes()): fp for fp in lote}
            for fut in as_completed(futuros):
                n_usados += 1
                try:
                    data = fut.result()
                    res  = data.get("result") or []
                    if res:
                        todos_tops.append(res[0])
                except Exception:
                    pass

        logger.debug(
            f"[lote {num_lote}] {n_usados - n_pre} frame(s) enviados a trace.moe → "
            f"{len(todos_tops) - tops_pre} devolvieron resultado utilizable "
            f"(suma de todos los lotes hasta aquí: {len(todos_tops)} con resultado / {n_usados} enviados)"
        )

        # Mínimo 2 lotes (3 frames) antes de aceptar: un único frame puede
        # coincidir accidentalmente con otro anime aunque tenga 100% de similitud.
        if num_lote == 0:
            continue

        # Fase 1: esperar a que la serie alcance el umbral de confianza.
        if not _serie_resuelta:
            _, sim_consenso = _consenso_trace(todos_tops, require_mayoria=True)
            if sim_consenso < _TRACE_UMBRAL_RAPIDO:
                logger.debug(
                    f"[lote {num_lote}] Fase 1 — serie en dispersión: "
                    f"sim={sim_consenso:.3f} < {_TRACE_UMBRAL_RAPIDO} → continuar"
                )
                continue
            logger.debug(f"[lote {num_lote}] Fase 1 — serie resuelta: sim={sim_consenso:.3f}")
            _serie_resuelta = True

        # Fase 2: serie resuelta — verificar si el episodio también tiene consenso.
        # Si el episodio sigue en dispersión con datos disponibles, continuar
        # acumulando lotes (hasta el máximo de n_fotogramas) en vez de salir.
        mejor_temp, _ = _consenso_trace(todos_tops)
        id_temp = None
        if mejor_temp is not None:
            try:
                id_temp = int(mejor_temp.get("anilist")) if mejor_temp.get("anilist") is not None else None
            except Exception:
                pass
        cands_temp = (
            [t for t in todos_tops if t.get("anilist") == id_temp]
            if id_temp is not None else list(todos_tops)
        )
        ep_temp     = _consenso_episodio(cands_temp, log_fn=lambda _: None)
        has_ep_data = any(t.get("episode") is not None for t in cands_temp)

        logger.debug(
            f"[lote {num_lote}] Fase 2 — episodio_provisional={ep_temp!r}, hay_datos_episodio={has_ep_data}"
        )
        if ep_temp is not None or not has_ep_data:
            logger.debug(
                f"[lote {num_lote}] salida — "
                + ("episodio resuelto" if ep_temp is not None else "sin datos de episodio")
            )
            break
        logger.debug(f"[lote {num_lote}] episodio en dispersión con datos → continuar")

    if log_fn is not None:
        log_fn(f"  - Enviados {n_usados}/{len(rutas_fotogramas)} fotograma(s) a trace.moe")

    if not todos_tops:
        raise RuntimeError("trace.moe no pudo identificar el anime con los fotogramas.")

    if log_fn is not None and len(todos_tops) < n_usados:
        log_fn(f"  - Resultados utilizables: {len(todos_tops)}/{n_usados}")

    mejor, mejor_sim = _consenso_trace(todos_tops)
    if mejor is None:
        raise RuntimeError("trace.moe no pudo identificar el anime con los fotogramas.")

    anilist_id = mejor.get("anilist")
    try:
        anilist_id = int(anilist_id) if anilist_id is not None else None
    except Exception:
        anilist_id = None

    candidatos_ganador = (
        [t for t in todos_tops if t.get("anilist") == anilist_id]
        if anilist_id is not None
        else list(todos_tops)
    )
    logger.debug(
        f"[episodio] frames que coincidieron con la serie ganadora (anilist_id={anilist_id}): "
        f"{len(candidatos_ganador)} de {len(todos_tops)} — estos son los que votan por episodio"
    )
    ep = _consenso_episodio(candidatos_ganador, log_fn=log_fn)

    filename = mejor.get("filename") or ""

    titulo = None
    if anilist_id is not None:
        try:
            titulo = anilist_titulo_por_id(anilist_id)
        except Exception:
            pass

    # Parsear el filename del frame top solo si algún fallback sigue siendo necesario.
    # Si AniList ya resolvió el título Y el consenso ya resolvió el episodio,
    # este parseo no aporta nada y su log [parsing] solo confundiría la traza.
    parsed = None
    if titulo is None or ep is None:
        parsed = parsear_nombre_archivo(filename) if filename else None

    if not titulo:
        titulo = (parsed.titulo if parsed and parsed.titulo else None) or filename or "Anime"
    ep = ep if ep is not None else (parsed.episodio if parsed else None)

    return AnimeDetectado(
        titulo=titulo,
        anilist_id=anilist_id,
        episodio=ep,
        similitud=float(mejor_sim),
    ), n_usados
