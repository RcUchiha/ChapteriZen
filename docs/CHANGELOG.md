## [0.0.5] - 2026-03-29

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
