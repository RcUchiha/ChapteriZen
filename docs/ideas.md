- Mejoras de UX pendientes — directorios persistentes por campo (como en SincroNyaa) y drag & drop en el campo de video.

- Optimizar la información que recibimos del log, para que me arroje nombres alternativos de los animes tanto en inglés como en romaji.
  - Que me arroje el enlace del anime detectado en MAL. 
  - Poner en español algunas cadenas como "parse filename", "parsed", "(resolver sequel season)"

- Ver si lo del cache en AppData se borra después de tiempo o si se almacenaría.

- Cuando AnimeThemes no tenga un anime en su base de datos, hacer que ChapteriZen pueda tomar las canciones desde los videos que carguemos. Solo bastaría con indicar nosotros los tiempos en que inicia y termina el Opening o el Ending.
Ya sea poniendo marcadores en una barra de avance de video. O cargando un archivo de audio de la canción para hacer el match offset.

- Ver si las opciones de parámetros de coincidencia como submuestreo, porción del theme y umbral de puntuación siguen siendo útiles. Y si es posible, ver la manera de que no sean necesarias, ya que siendo realistas, es dificil que un usuario sepa para qué sirven o como utilizarlas. Además, también no creo que el usuario sepa para qué es la opción de "OP/ED exactos" y cuando marcarla o desmarcarla.

- (".mkv", ".mp4", ".avi", ".webm", ".mov", ".m2ts") > Añade estas: .ts, .wmv, .vob

- Hacer que los textos flotantes de los textbox sean en cursiva y del mismo color tenue. Porque noto que "Selecciona el archivo de video…" es más tenue que los otros dos. ¿Se puede hacer un tono intermedio entre ambos colores?

3. Jikan como fallback — la observación es correcta: AnimeThemes ya resolvió bien sin Jikan en ese caso. Pero cambiar la arquitectura de Jikan ahora es más invasivo. Lo más pragmático por ahora es mejorar la sanitización del filename y el formato, y dejar la refactorización de Jikan para cuando hagamos la separación en módulos.
