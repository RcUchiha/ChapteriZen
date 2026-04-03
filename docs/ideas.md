1. Mejorar el parsing de nombres empleando las siguientes bibliotecas, abriendo la posibilidad de mejorar la sanitización del filename, y así hacer más poderosa, por ejemplo, la función de "inferir_consulta_desde_nombre_archivo":
  - aniparse (principal)
  - anitopy (respaldo)

Además, estas bibliotecas podrían ayudar con la tarea de resolución de metadatos del ResolverWorker, ya que también extraen datos como temporada y episodio, etc.

---

2. Estructurar el log en etapas para pipeline más legible.

Piensa así:

[ETAPA]
  → acción
    → detalle fino

Y separa:

- 🟢 INFO → progreso
- 🔵 DEBUG → detalles internos
- 🟡 WARN → cosas raras
- 🔴 ERROR → fallos

Estructura propuesta:
```
[Resolver]
  INFO  Analizando nombre de archivo
  DEBUG temporada=2 episodio=6

  INFO  Buscando en Jikan (base)
  DEBUG query='Hime-sama Goumon no Jikan desu'
  WARN  Secuela no encontrada → usando base

  INFO  Resolviendo AnimeThemes
  DEBUG resultados=3 priorizados=3

[Preparación]
  INFO  Duración video: 1425.11s
  INFO  Archivo salida: Chapters.xml

  INFO  Cargando temas
  DEBUG OP1 cache=OK
  DEBUG ED1 cache=OK

  INFO  Precargando audio de temas

[Análisis]
  INFO  Buscando OP (0s–300s)

  [OP][win0 @0s]
    DEBUG FFT top: OP1 (0.184)
    DEBUG DTW=0.26 → score=0.752
    INFO  match 00:00:06 → 00:01:35

  [OP][win1 @15s]
    ...

  INFO  Mejor OP: 00:00:50 → 00:02:19 (0.806)

  INFO  Buscando ED (1125s–1425s)
  ...

[Resultado]
  INFO  Chapters generados correctamente
```

---

4. Ver si lo del cache en AppData se borra después de tiempo o si se almacenaría.

5. Cuando AnimeThemes no tenga un anime en su base de datos, hacer que ChapteriZen pueda tomar las canciones desde los videos que carguemos. Solo bastaría con indicar nosotros los tiempos en que inicia y termina el Opening o el Ending.
Ya sea poniendo marcadores en una barra de desplazamiento de video. O también cargando un archivo de audio de la canción para hacer el match offset.

6. Ver si las opciones de parámetros de coincidencia como submuestreo, porción del theme y umbral de puntuación siguen siendo útiles. Y si es posible, ver la manera de que no sean necesarias, ya que siendo realistas, es dificil que un usuario sepa para qué sirven o como utilizarlas. Además, también no creo que el usuario sepa para qué es la opción de "OP/ED exactos" y cuando marcarla o desmarcarla.

7. Hacer que los placeholders sean en cursiva y del mismo color tenue. Porque noto que la cadena "Selecciona el archivo de video…" es más tenue que las otras. ¿Se puede hacer un tono intermedio entre ambos colores?

8. Agregar fallbacks a otras APIs:
- **Problema**: Solo usa Jikan (MAL). Si MAL no tiene datos o hay rate limits, falla.
- **Mejora**: Agregar fallback a AniList (otra API popular) si Jikan falla. AniList tiene mejor soporte para relaciones de series.
- **Implementación**: Crear función buscar_en_anilist similar a jikan_buscar_anime, y llamarla si Jikan retorna vacío.
- **Beneficio**: Más confiabilidad, menos "Slug vacío" errors.

9. Jikan como fallback — la observación es correcta: AnimeThemes ya resolvió bien sin Jikan en ese caso. Pero cambiar la arquitectura de Jikan ahora es más invasivo. Lo más pragmático por ahora es mejorar la sanitización del filename y el formato, y dejar la refactorización de Jikan para cuando hagamos la separación en módulos.

10. A tener en cuenta: cuando un nombre de anime trae consigo solo el nombre de la primera temporada (sin "2ns seadon", "season 2", etc.) y contiene un número de episodio que supera el total de episodios que oficialmente tiene esa temporada, significa que probablemente es de la segunda o tercera temporada. Por ejemplo, "Jigokuraku - 25":
  - parse_filename: title='Jigokuraku', season=1, episode=25, source='aniparse'
  - parsed: temporada=1, episodio=25

Primero debió verificar la cantidad de episodios de la primera temporada. Y al ver que no tiene el total de 25, debió hacer una suma, contemplando los 13 episodios de la primera temporada y luego 12 de la segunda temporada (si es que la segunda tiene 12) para dar un total de 25. Luego debió parsear con la segunda temporada y determinar que el episodio del que se trata es del 12. 

Así pues, las líneas entes mencionadas debieron dar este resultado:
  - parse_filename: title='Jigokuraku 2nd Season', season=2, episode=12, source='aniparse'
  - parsed: temporada=2, episodio=12

Si fuera el caso de que las temporadas fueran de 10 episodios y hubiera una tercera, debería parsear en la tercera, siendo el episodio 5 el resultante.

---

11. Mejoras de UX pendientes — directorios persistentes por campo (como en SincroNyaa) y drag & drop en el campo de video.
