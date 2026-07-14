"""Integracion con AnimeThemes: busqueda de series, descarga y cacheo
de audios de OP/ED. Movido sin cambios desde chapterizen.py (monolito
original, v0.0.7)."""
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from .config import (
    _http,
    _reintento_http,
    get_api_cache,
    _TTL_API_DAYS,
    _TTL_THEMES_DAYS,
    _THEMES_DIR,
    ANIMETHEMES_SEARCH,
    ANIMETHEMES_ANIME,
)
from .ffmpeg_utils import extraer_audio_wav_mono_16k


@_reintento_http
def buscar_anime_en_animethemes(nombre_busqueda: str) -> List[dict]:
    clave = f"at_search:{nombre_busqueda.strip().casefold()}"
    cached = get_api_cache().get(clave)
    if cached is not None:
        return cached
    r = _http.get(
        ANIMETHEMES_SEARCH,
        params={"fields[search]": "anime", "q": nombre_busqueda},
    )
    r.raise_for_status()
    js      = r.json()
    result  = (((js or {}).get("search") or {}).get("anime") or [])
    get_api_cache().set(clave, result, expire=_TTL_API_DAYS * 86400)
    return result

def obtener_anime_de_animethemes(slug: str) -> dict:
    clave  = f"at_anime:{slug}"
    cached = get_api_cache().get(clave)
    if cached is not None:
        return cached

    url      = f"{ANIMETHEMES_ANIME}/{slug}"
    intentos = [
        {"include": "animethemes.song.artists,animethemes.animethemeentries.videos.audio"},
        {"include": "animethemes.animethemeentries.videos.audio"},
        {},
    ]
    ultimo = None

    @_reintento_http
    def _get(params):
        r = _http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    for params in intentos:
        try:
            js = _get(params)
            if isinstance(js, dict) and "anime" in js and isinstance(js["anime"], dict):
                result = js["anime"]
            elif isinstance(js, dict) and ("animethemes" in js or "name" in js or "slug" in js):
                result = js
            else:
                result = {}
            get_api_cache().set(clave, result, expire=_TTL_THEMES_DAYS * 86400)
            return result
        except httpx.HTTPStatusError as e:
            ultimo = e
            if e.response.status_code == 422:
                continue
            raise
    raise RuntimeError(f"AnimeThemes: no pude obtener /anime/{slug}. Último error: {ultimo}")

def construir_mapa_mostrar_temas(anime_json: dict) -> Dict[str, str]:
    salida: Dict[str, str] = {}
    for tema in (anime_json.get("animethemes") or []):
        slug_tema_raw = (tema.get("slug") or "").strip()
        slug_tema     = re.sub(r"v\d+$", "", slug_tema_raw, flags=re.I)
        if not slug_tema:
            continue
        cancion  = tema.get("song") or {}
        titulo   = (cancion.get("title") or "").strip()
        artistas = [
            (a.get("name") or "").strip()
            for a in (cancion.get("artists") or [])
            if (a.get("name") or "").strip()
        ]
        if not titulo or not artistas:
            continue
        artista  = ", ".join(artistas)
        etiqueta = None
        if slug_tema.upper().startswith("OP"):
            etiqueta = f'Opening: "{titulo}" por {artista}'
        elif slug_tema.upper().startswith("ED"):
            etiqueta = f'Ending: "{titulo}" por {artista}'
        if etiqueta:
            salida[slug_tema]     = etiqueta
            salida[slug_tema_raw] = etiqueta
    return salida

def nombre_archivo_seguro(name: str) -> str:
    s = str(name)
    s = re.sub(r'"(?=\w)', "“", s)   # " antes de palabra → "
    s = re.sub(r'(?<=\w)"', "”", s)  # " después de palabra → "
    s = s.replace(":", "꞉").replace("?", "？")
    s = re.sub(r'[<>/\\|*\x00-\x1F]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s

@_reintento_http
def descargar_archivo(url: str, ruta_salida: str):
    """Descarga con streaming para no cargar archivos grandes en memoria."""
    with _http.stream("GET", url) as r:
        r.raise_for_status()
        Path(ruta_salida).write_bytes(r.read())

def _nombres_serie_iguales(a: str, b: str) -> bool:
    return (a or "").strip().casefold() == (b or "").strip().casefold()

def _episodio_en_entrada(episodio: int, entry: dict) -> bool:
    """True si el episodio está cubierto por el rango declarado en esta entry."""
    if episodio <= 0:
        return True
    eps_str = (entry.get("episodes") or "").strip()
    if not eps_str:
        return True
    for parte in re.split(r'[,;]', eps_str):
        parte = parte.strip()
        if not parte:
            continue
        if '-' in parte:
            try:
                lo, hi = parte.split('-', 1)
                if int(lo.strip()) <= episodio <= int(hi.strip()):
                    return True
            except ValueError:
                return True
        else:
            try:
                if episodio == int(parte):
                    return True
            except ValueError:
                return True
    return False

def _tema_cubre_episodio(tema: dict, episodio: int) -> bool:
    """True si alguna entry del tema cubre el episodio dado."""
    if episodio <= 0:
        return True
    entries = tema.get("animethemeentries") or []
    if not entries:
        return True
    return any(_episodio_en_entrada(episodio, e) for e in entries)

def construir_cache_temas(slug: str, anime_json: dict, log, episodio: int = 0) -> Tuple[Path, set]:
    """
    Descarga y convierte los audios de los temas de AnimeThemes.
    - Los archivos OGG/WAV se guardan en _THEMES_DIR/<slug>/wav/
    - Los metadatos de cada tema se guardan en diskcache con TTL de 30 días
    - Si episodio > 0, solo descarga los temas cuyas entries cubren ese episodio
    Devuelve (wav_dir, slugs_relevantes): directorio WAV y set de slugs
    que corresponden al episodio (vacío = incluir todos).
    """
    series_dir = _THEMES_DIR / nombre_archivo_seguro(slug)
    wav_dir    = series_dir / "wav"
    series_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(exist_ok=True)

    series_name = anime_json.get("name") or anime_json.get("slug") or "series"
    clave_serie = f"themes_meta:{slug}"
    meta_cached = get_api_cache().get(clave_serie)

    # Si la serie cambió de nombre, limpiar archivos del directorio
    if meta_cached and not _nombres_serie_iguales(meta_cached.get("nombre_serie", ""), series_name):
        for p in series_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
        for p in wav_dir.glob("*.wav"):
            try:
                p.unlink()
            except Exception:
                pass
        get_api_cache().delete(clave_serie)
        meta_cached = None

    temas_meta: dict = (meta_cached or {}).get("temas", {})

    pendientes: List[Tuple[str, str]] = []
    slugs_relevantes: set = set()
    temas = anime_json.get("animethemes") or []

    for tema in temas:
        if not _tema_cubre_episodio(tema, episodio):
            continue

        base_slug = tema.get("slug") or ""
        cur_theme = (
            base_slug
            if (base_slug and base_slug[-1].isdigit())
            else (base_slug + "1" if base_slug else "")
        )
        if not cur_theme:
            continue

        slugs_relevantes.add(cur_theme)

        audio_version       = 1
        links_audio_vistos: set = set()

        for entry in (tema.get("animethemeentries") or []):
            full_cur_theme = cur_theme
            if audio_version > 1:
                full_cur_theme += f"v{audio_version}"

            for video in (entry.get("videos") or []):
                if (video.get("overlap") or "None") != "None":
                    continue
                audio      = video.get("audio") or {}
                link       = audio.get("link")
                updated_at = audio.get("updated_at")
                if not link or link in links_audio_vistos:
                    continue

                links_audio_vistos.add(link)

                ogg_path = series_dir / f"{full_cur_theme}.ogg"
                wav_path = wav_dir / f"{full_cur_theme}.wav"
                rec      = temas_meta.get(full_cur_theme)

                if (
                    rec
                    and rec.get("updated_at") == updated_at
                    and ogg_path.exists()
                    and wav_path.exists()
                ):
                    log(f"  - {full_cur_theme}: en caché ✓")
                    audio_version += 1
                    break

                temas_meta[full_cur_theme] = {"updated_at": updated_at, "link": link}
                pendientes.append((full_cur_theme, link))
                audio_version += 1
                break

    if pendientes:
        log(f"• Descargando {len(pendientes)} temas desde AnimeThemes (paralelo)…")

    def _bajar_y_convertir(item: Tuple[str, str]) -> Tuple[str, Optional[str]]:
        theme_name, link = item
        ogg_path = series_dir / f"{theme_name}.ogg"
        wav_path = wav_dir / f"{theme_name}.wav"
        try:
            descargar_archivo(link, str(ogg_path))
            extraer_audio_wav_mono_16k(str(ogg_path), str(wav_path))
            return theme_name, None
        except Exception as e:
            return theme_name, str(e)

    if pendientes:
        max_workers = min(4, len(pendientes))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futuros = {ex.submit(_bajar_y_convertir, item): item for item in pendientes}
            for fut in as_completed(futuros):
                theme_name, err = fut.result()
                if err:
                    log(f"  - ⚠️ {theme_name}: error al descargar/convertir: {err}")
                else:
                    log(f"  - ✅ {theme_name}: listo")

    # Persistir metadatos actualizados en diskcache
    get_api_cache().set(
        clave_serie,
        {"nombre_serie": series_name, "temas": temas_meta},
        expire=_TTL_THEMES_DAYS * 86400,
    )
    return wav_dir, slugs_relevantes
