# Ideas

Ideas y propuestas todavía no implementadas, a distinto nivel de
madurez (desde una observación suelta hasta un diseño ya pensado).

## 1. [Observabilidad] Estructurar el log en etapas para un pipeline más legible

Piensa así:

```
[ETAPA]
  → acción
    → detalle fino
```

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

## 2. [Funcionalidad] Marcado manual de OP/ED cuando AnimeThemes no tiene la serie

Cuando AnimeThemes no tenga un anime en su base de datos, hacer que
ChapteriZen pueda tomar las canciones desde los videos que carguemos.
Solo bastaría con indicar nosotros los tiempos en que inicia y termina
el Opening o el Ending. Ya sea poniendo marcadores en una barra de
desplazamiento de video, o también cargando un archivo de audio de la
canción para hacer el match offset.

---

## 3. [UX] Directorios persistentes por campo y drag & drop

Mejoras de UX pendientes — directorios persistentes por campo (como en
SincroNyaa) y drag & drop en el campo de video.

---

## 4. [Rendimiento] Memoria de resolución entre episodios de la misma tanda

Hoy cada video repite desde cero todo el pipeline de identificación
(parseo de filename → trace.moe si hace falta → Jikan/AniList →
cross-verificación → resolución de temporada → slug de AnimeThemes),
aunque la serie ya se resolvió con el episodio anterior de la misma
temporada. Lo único que cambia episodio a episodio (salvo cruce de
temporada, Camino B) es el número de episodio y el matching de audio
puntual.

Propuesta: permitir reutilizar `(titulo_usado, slug)` ya resueltos para
los siguientes episodios de la misma tanda, saltando directo a
`ChapterizerWorker` con solo el episodio nuevo parseado del filename.
Forma más simple sugerida: guardar `(carpeta_del_video, titulo_usado,
slug)` como estado de sesión en la ventana principal (persiste mientras
la GUI esté abierta, no hace falta persistencia en disco) tras un
`ChapterizerWorker` exitoso, y un checkbox "Reusar identificación
anterior" habilitado solo cuando el video nuevo está en la misma
carpeta que el anterior.

Deliberadamente sin detección automática de cruce de temporada (Camino
B) al reusar — eso duplicaría de nuevo la lógica de resolución que se
busca evitar ejecutar. El usuario desmarca el checkbox manualmente al
empezar una temporada nueva.
