## [0.0.6] — 03-04-2026

### Rendimiento

- **Precarga de WAVs de temas en memoria**. Los archivos WAV de OP/ED ahora se leen del disco una sola vez antes de iniciar la búsqueda con ventana deslizante. En la versión anterior, cada ventana del sliding window relanzaba una lectura de disco por tema, lo que podía significar 80+ lecturas innecesarias por episodio.
- **Resampleo movido a la precarga**. Si un WAV de tema tiene una tasa de muestreo distinta a 16kHz (edge case), el resampleo ahora ocurre una sola vez al precargar, no en cada iteración del loop de ventanas.
- **Features de temas precalculadas**. Las features MFCC+chroma de cada tema se calculan (o recuperan de caché) una sola vez al precargar, y se almacenan en el objeto `TemaAudio`. En la versión anterior, `obtener_features_con_cache` se llamaba dentro del loop DTW por cada ventana × candidato.
- `frames_t` **precalculado en** `TemaAudio`. La longitud en frames de cada tema se calcula una vez y viaja con el objeto, eliminando el recálculo en `_buscar_con_ventana`.

### Correcciones de bugs

- `tuple index out of range` **al buscar OP/ED**. El refactor que eliminó `ruta_wav` del tuple interno de candidatos FFT dejó referencias a `c[5]` que ya no existía (el índice correcto pasó a ser `c[4])`. Corregido en cuatro puntos: el sort, el cálculo de percentiles, el threshold dinámico y el log de candidatos.
- `_ANIPARSE_OK` **no definido al arrancar**. Los guards de importación de `aniparse` y `anitopy` se escribieron correctamente en el archivo pero en una sesión anterior no llegaron a persistir, causando `NameError en runtime`. Corregido.

### Refactors

- `_BaseWorker(QThread)` **como clase base**. `ResolverWorker` y `ChapterizerWorker` ahora heredan de `_BaseWorker`, que centraliza el método `_log` (antes duplicado en ambas clases) y las señales `log` y `progress`.
- `TemaAudio` **dataclass**. Reemplaza el tuple `(nombre, y_th, hz_th, frames_t)` en todo el pipeline de matching. Acceso por atributo nombrado en lugar de índice numérico.
- `CandidatoFFT` **dataclass**. Reemplaza el tuple interno de la fase FFT `(nombre, audio, inicio, fin, score_fft)`. Elimina por completo la clase de bug de índice incorrecto que causó el error anterior. `CandidatoFFT` además guarda una referencia al `TemaAudio` origen en lugar de copiar el array de audio.
- **Nuevo pipeline de parsing de nombres de archivo**. Se integran las bibliotecas `aniparse` (principal) y `anitopy` (respaldo) para parsear nombres de releases de anime. La función central `parsear_nombre_archivo()` implementa una estrategia de merge: elige el título con mejor score de limpieza entre ambos parsers, es consciente del número de temporada (evita que "Kingdom 5" se envíe a Jikan con el `5` pegado al título), y cae a un fallback regex si ambas bibliotecas fallan o producen un resultado con ruido residual.
- **Eliminación de código de limpieza manual redundante**. Las constantes `_RUIDO`, `_BRACKET_BLOCK` y las funciones `recortar_a_nombre_serie`, `_limpiar_nombre_release` y `_extraer_temporada_textual` fueron eliminadas, reemplazadas por el nuevo pipeline de parsing. Las funciones `quitar_sufijo_episodio` y `quitar_marcador_temporada` se conservaron porque operan sobre títulos canónicos de Jikan, no sobre nombres de archivo.
- **Eliminación de código muerto**. Se eliminaron la función standalone `mejor_coincidencia` (~115 líneas, duplicado inactivo de `_coincidencia_con_features`) y las constantes `OP_WINDOW_SEC` / `ED_WINDOW_SEC` (reemplazadas por `_SLIDE_OP_MAX` / `_SLIDE_ED_MAX` en una versión anterior).
- **Bug visual corregido en** `_buscar_con_ventana`. Un separador de sección `# GUI` estaba incrustado dentro de la clase `ChapterizerWorker`, haciendo que `_buscar_con_ventana` pareciera estar fuera de la clase al leer el código.

---

## [0.0.5] — 29-03-2026

### Added
- Integración más completa con AnimeThemes para obtención de openings/endings.
- Mejora en el pipeline de análisis de audio (incluyendo DTW para matching).
- Sistema más robusto de extracción de características de audio (MFCC, chroma).
- Manejo mejorado de selección manual cuando hay resultados ambiguos.
- Uso extendido de caché en disco para evitar reprocesos innecesarios.

### Changed
- Refactor importante del flujo de resolución de anime (ResolverWorker).
- Mejora en la precisión del matching de audio frente a versiones anteriores.
- Optimización del procesamiento de audio usando FFT y scipy.
- Mejor normalización y limpieza de nombres de archivo.
- Ajustes en la lógica de detección de temporada y episodio.
- Mejora en la interacción con APIs externas (Jikan, AnimeThemes).
- Sistema de logging migrado/mejorado para mayor claridad y depuración.

### Fixed
- Casos donde el matching de audio fallaba o daba resultados inconsistentes.
- Problemas en parsing de nombres complejos de episodios.
- Errores en respuestas incompletas o inválidas de APIs externas.
- Fallos en descarga o procesamiento de audio en ciertos escenarios.
