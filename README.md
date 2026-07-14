# ChapteriZen

Generador automático de capítulos (chapters) para episodios de anime. A partir de un video, identifica la serie y el episodio, localiza el opening/ending correspondiente en [AnimeThemes](https://animethemes.moe/) y hace *matching* de audio contra el episodio para ubicar exactamente dónde empiezan y terminan — sin necesidad de marcarlo a mano. El resultado es un XML de capítulos listo para usar con **mkvmerge/MKVToolNix**.

## Qué hace (pipeline)

1. **Parseo del nombre de archivo** (`aniparse` con `anitopy` de respaldo, más un fallback por regex) — extrae título, temporada y episodio, y limpia tags de release (resolución, codec, grupo, subtítulos, etc.).
2. **Identificación del anime**, con varias capas de resiliencia:
   - Si el título no es reconocible en el nombre de archivo, se identifica por **fotogramas vía trace.moe** (consenso por lotes sobre varios frames).
   - El título se resuelve/verifica contra **Jikan (MyAnimeList)**; si Jikan está caído (reintentos agotados), cae automáticamente a **AniList** como respaldo — búsqueda de título, navegación de secuela/temporada y conteo de episodios funcionan igual por cualquiera de las dos fuentes.
   - Si Jikan y trace.moe/AniList no coinciden, se abre un selector manual para que el usuario decida.
3. **Resolución del slug en AnimeThemes** — búsqueda por título (con reintentos usando títulos alternativos/variantes oficiales), con selector manual si el resultado es ambiguo.
4. **Descarga y cacheo de audio** de los openings/endings de la serie desde AnimeThemes.
5. **Matching de audio FFT → DTW** contra el video del episodio, para ubicar el inicio y fin exactos del OP/ED.
6. **Generación del XML de capítulos** (formato mkvmerge), guardado junto al video o en la carpeta que se indique.

Si el matching exacto no está disponible o se desactiva, el programa genera capítulos heurísticos (introducción / episodio / conclusión) como respaldo.

## Requisitos

- **Python 3.10+**
- **ffmpeg y ffprobe** disponibles en el `PATH` del sistema (se invocan como comandos externos vía `subprocess` — no son paquetes de Python, hay que instalarlos aparte).
- Dependencias de Python listadas en `requirements.txt` (PyQt6, httpx, numpy, librosa, scipy, rapidfuzz, pydantic, diskcache, loguru, tenacity, aniparse, anitopy, entre otras).

## Instalación

```bash
git clone https://github.com/RcUchiha/ChapteriZen.git
cd ChapteriZen
pip install -r requirements.txt
```

## Uso

```bash
python -m chapterizen
```

(o, si se instaló el paquete con `pip install -e .`, también queda disponible el comando `chapterizen` directamente, gracias al entry point declarado en `pyproject.toml`.)

Es una aplicación de escritorio (PyQt6), sin flags de línea de comandos. En la ventana:

- **Video** — el archivo a procesar (extensiones soportadas: `.mkv .mp4 .avi .webm .mov .m2ts .ts .wmv .vob`).
- **Carpeta de salida** — opcional; si se deja vacía, el XML se guarda junto al video. Con la casilla **"Guardar en carpeta Chapters"** se guarda en una subcarpeta `Chapters/` en vez de la carpeta del video.
- **OP/ED exactos (AnimeThemes + coincidencia de audio)** — activado por defecto; si se desactiva, se generan capítulos heurísticos sin identificar el anime ni descargar audio.
- **Búsqueda en AnimeThemes (opcional)** — para forzar manualmente el término de búsqueda en vez de que se infiera del nombre de archivo.
- **Parámetros de coincidencia** — ajustes finos del matching de audio: submuestreo, porción del theme a comparar (0.5–1.0) y umbral mínimo de puntuación (0.10–0.40). Los valores por defecto funcionan bien en la mayoría de los casos.

Durante el proceso, si hace falta desambiguar (varios resultados posibles en Jikan/AniList/AnimeThemes, o una discrepancia entre fuentes), se abre un selector para elegir manualmente — queda registrado en el log tanto la apertura del selector como la opción elegida.

## Estructura del proyecto

```
chapterizen/
├── __main__.py           # punto de entrada, ventana principal (Qt)
├── config.py              # constantes, endpoints, cliente HTTP, caché en disco, logging, reintentos
├── modelos.py              # dataclasses y modelos pydantic compartidos
├── parsing.py              # parseo de nombres de archivo y detección de ruido/tags de release
├── ffmpeg_utils.py          # extracción de fotogramas y audio vía ffmpeg/ffprobe
├── trace_moe.py             # identificación de anime por fotogramas (trace.moe)
├── jikan.py                  # integración con Jikan/MyAnimeList
├── anilist.py                 # integración con AniList (fallback de Jikan, navegación de secuela)
├── animethemes.py              # búsqueda y descarga de audio OP/ED (AnimeThemes)
├── audio_matching.py            # pipeline de matching de audio (FFT → DTW)
├── chapters_xml.py               # generación del XML de capítulos (mkvmerge)
├── naming.py                      # construcción de la ruta de salida
└── gui/
    ├── pickers.py                  # diálogo de selección manual (desambiguación)
    ├── resolver_worker.py           # QThread: resolución de título/slug (ResolverWorker)
    └── chapterizer_worker.py         # QThread: generación de capítulos (ChapterizerWorker)
```

`chapterizen.py`, en la raíz del repo, es el monolito original previo a esta modularización. Se mantiene intencionalmente en el repo como referencia mientras se termina de validar el paquete `chapterizen/` en uso real — se eliminará en una versión futura una vez confirmada la estabilidad.

## Tests

```bash
python -m pytest tests/
```

La suite (pytest + [respx](https://lundberg.github.io/respx/) para mockear las llamadas HTTP a Jikan/AniList/AnimeThemes/trace.moe) cubre lógica pura, red mockeada, e integración completa del flujo de `ResolverWorker.run()` — incluyendo los distintos caminos de fallback y los selectores manuales.

Para chequeo estático:

```bash
python -m pyflakes chapterizen/
```

## Estado actual

- **Versión**: 0.0.8 — ver [`docs/CHANGELOG.md`](docs/CHANGELOG.md) para el historial completo de cambios por versión.
- **Limitaciones conocidas**: ver [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md). Actualmente hay un caso sin cubrir (nombres de archivo con el título en dos idiomas concatenados en un mismo string, ej. `"Chained Soldier (Mato Seihei no Slave)"`), que degrada de forma segura al selector manual sin romper el flujo.
- El paquete `chapterizen/` (modularizado) es la versión activa y mantenida; `chapterizen.py` es solo referencia temporal, como se explica arriba.

## Licencia

MIT — ver [`LICENSE`](LICENSE).
