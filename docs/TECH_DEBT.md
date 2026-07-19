# Tech Debt

Deuda técnica interna: código o estructura que funciona correctamente hoy
(no afecta el comportamiento observable para el usuario — para eso está
`docs/KNOWN_LIMITATIONS.md`) pero que amerita limpieza o reorganización
futura.

## [RESUELTO] `chapterizen.py` (monolito legacy en la raíz) diverge del paquete

`chapterizen.py` en la raíz del repo era el monolito original (v0.0.7) que
antecedía al paquete `chapterizen/`. Ya no era el código que ejecutaba la
GUI (`chapterizen/__main__.py` es el entry point desde la modularización),
pero seguía presente en el repo como referencia.

Ese archivo usaba su propia copia de `_aplicar_canon` (versión vieja, sin
el chequeo de variantes oficiales de título) en sus 3 call sites — la
versión equivalente en el paquete (`ResolverWorker._aplicar_canon_multivariante`,
`gui/resolver_worker.py`) sí prueba esas variantes desde el commit `895b51e`.
Es decir, monolito y paquete ya producían resultados distintos ante el
mismo input en el caso de variantes de idioma.

**Resuelto en la versión 0.1.0**: confirmado por grep que nada en el
código activo (paquete `chapterizen/`, tests, scripts) importa, ejecuta
o depende de `chapterizen.py` — los `from chapterizen import ...` de los
tests siempre resuelven al paquete, nunca al archivo plano (no es un path
de import válido). Se eliminó el archivo del repo; la divergencia queda
sin objeto.

## [CORREGIDO] El sink de loguru se configuraba al importar `config.py`, sin distinguir GUI de tests/scripts

`config.py` llamaba a `logger.remove()` + `logger.add(_LOG_DIR / "chapterizen_{time:YYYY-MM-DD}.log", ...)`
directamente en el cuerpo del módulo, así que se ejecutaba apenas alguien
hacía `import chapterizen` (o cualquier import que disparara
`chapterizen/__init__.py`) — no solo cuando arrancaba la GUI real. Confirmado:
correr `pytest` escribía entradas DEBUG reales (incluyendo datos sintéticos
de fixtures, ej. títulos de prueba como "Attack on Titan"/"Shingeki no
Kyojin") en el mismo archivo rotativo de producción que usa la GUI
(`%LOCALAPPDATA%\ChapteriZen\ChapteriZen\Logs\` en Windows). Cualquier
script standalone que importaba el paquete tenía el mismo problema salvo
que reemplazara el sink explícitamente después del import (ver
`scripts/calibrar_trace_moe.py` y `scripts/simular_decision_resolver.py`,
que hacían justamente eso).

Mismo patrón que el bug de aislamiento de `_API_CACHE` (ver commit
`33e5611`, `get_api_cache()`), aplicado a logging en vez de caché: un
efecto secundario de proceso configurado incondicionalmente al importar un
módulo, sin diferenciar quién importa.

**Corregido:** la configuración del sink se movió a una función explícita,
`configurar_logging_produccion()` (`chapterizen/config.py`), que ya no se
ejecuta en el cuerpo del módulo — solo la llama `__main__.main()` al
arrancar la GUI real. Importar `chapterizen.config` (tests, scripts,
exploración en REPL) ya no tiene ningún efecto secundario sobre logging;
sin sink configurado explícitamente, loguru simplemente usa su sink de
stderr por defecto. Los 2 scripts existentes que ya se redirigían
manualmente después del import (`calibrar_trace_moe.py`,
`simular_decision_resolver.py`) siguen funcionando sin cambios de
comportamiento — sus comentarios se actualizaron para no seguir
describiendo un bug que ya no existe.

Cubierto por `tests/test_config_logging.py`: confirma que importar
`chapterizen.config` no deja ningún sink de archivo activo, y que
`configurar_logging_produccion()` sí lo agrega cuando se llama
explícitamente.

## `_TRACE_UMBRAL_CONFIANZA_ALTA` (punto 5 del análisis de resolver_worker.py) evaluado y NO implementado

Propuesta original: agregar un segundo umbral (distinto de
`_TRACE_UMBRAL_RAPIDO` en `trace_moe.py`, que gobierna el corte temprano
del loop de fotogramas) para que, en la cross-verificación
`"id_reutilizado"` de `_verificar_y_resolver_discrepancia`
(`resolver_worker.py`), se confíe automáticamente en el `anilist_id` ya
detectado sin abrir el picker de discrepancia cuando la similitud cae en
la banda 0.85–0.95 (candidatos evaluados: 0.88 / 0.90 / 0.92).

**Evaluación empírica (2026-07-15):** se corrió `scripts/calibrar_trace_moe.py`
contra 42 episodios reales de la librería del usuario (41 en una primera
tanda + 1 exitoso al inicio de una segunda tanda que se frenó por
agotamiento de cuota de trace.moe, ver `docs/KNOWN_LIMITATIONS.md`),
llamando directamente a `identificar_anime_con_fotogramas()` sin sesgo
hacia casos "difíciles" — selección diversa (títulos populares y oscuros,
temporada explícita en idioma distinto al oficial, distintas releases/
fansubs de un mismo episodio para chequeo cruzado).

**Resultado:** la banda 0.85–0.95 no apareció ni una sola vez en los 42
casos. Distribución marcadamente bimodal: 41/42 casos entre 96.1% y
99.99% de similitud (verificados manualmente contra evidencia
independiente de trace.moe — nombre de archivo, carpeta contenedora,
releases hermanas de otro fansub convergiendo al mismo `anilist_id` — 39
confirmados correctos, 1 marcado "sin verificar" por baja confianza
propia en el título exacto de una obra adulta oscura, ninguno
descartado por conveniencia), y exactamente 1 caso claramente mal
identificado en 81.33% (Ingoku Danchi identificado como Kuroshitsuji,
obra completamente distinta), muy por debajo de la banda de interés.

**Decisión:** no implementar el segundo umbral. No hay evidencia real de
que la banda 0.85–0.95 ocurra con frecuencia suficiente en uso real como
para justificar la complejidad de una constante adicional — el problema
que motivó la propuesta parece más raro en la práctica de lo asumido
inicialmente. Esta conclusión se basa en datos reales de la librería del
usuario (no en una muestra construida a propósito para forzar casos
límite), y queda documentada explícitamente para que quede claro que no
es una suposición.

Si en el futuro aparece evidencia de que esta banda sí ocurre con más
frecuencia (contenido de peor calidad de imagen, escenas más genéricas,
un catálogo distinto), reabrir esta evaluación corriendo
`scripts/calibrar_trace_moe.py` de nuevo en vez de asumir el valor a ojo.

## `_parsed_dict_a_campos` nunca extrae nada útil de aniparse — usa el schema de anitopy

`chapterizen/parsing.py` integra aniparse como parser "principal" y anitopy
como "respaldo" (`parsear_nombre_archivo`, línea ~279), pero `_parsed_dict_a_campos`
(línea ~200) extrae los campos con las claves `anime_title` / `anime_season` /
`episode_number` — ese es el schema plano de **anitopy**. La versión de
aniparse pineada en `requirements.txt` (`aniparse==2.0.0`) devuelve un
schema completamente distinto y anidado: `{"series": [{"title": ...,
"season": [{"number": ...}], "episode": [{"number": ...}]}], ...}`. Ninguna
de las claves que busca `_parsed_dict_a_campos` existe en ese dict.

Confirmado en runtime (no solo leyendo el código): corriendo `aniparse.parse(...)`
directo sobre 82 nombres de archivo reales (ver
`scripts/comparacion_parsers.txt`, gitignoreado — nombres de la librería
del usuario), `aniparse.parse()` sí identifica título/temporada/episodio
correctamente en varios casos (ej. "Chained Soldier - S02E01" →
`series[0] = {"title": "Chained Soldier", "season": [{"number": 2}],
"episode": [{"number": 1}]}`), pero `_parsed_dict_a_campos` siempre
devuelve `("", None, None)` para ese resultado porque busca las claves
equivocadas. En las 82/82 filas de la muestra, `titulo` de aniparse salió
vacío — sin una sola excepción.

**Por qué no afecta el comportamiento observable hoy (por eso va acá y no
en KNOWN_LIMITATIONS.md):** `parsear_nombre_archivo` elige el título con
mejor `_score_titulo` entre aniparse y anitopy, y un título vacío siempre
pierde. anitopy ha estado cargando el 100% de la extracción real desde
que esto se rompió (o desde que se introdujo, no hay forma de saberlo sin
revisar cuándo se fijó la versión de aniparse), sin que el merge lo note
— por diseño el sistema tolera que una de las dos fuentes falle. El
fallback a regex tampoco se activó ni una vez en la muestra de 82,
consistente con que anitopy solo alcanza para producir un título usable
en casi todos los casos.

**Lo que sí se pierde:** la redundancia que el diseño original buscaba
(dos parsers independientes verificándose / complementándose) no existe
en la práctica — es anitopy en solitario con un colega que nunca habla.
Si anitopy alguna vez falla en un caso donde aniparse sí hubiera
acertado, hoy no hay ninguna red de contención ahí.

**Actualización (2026-07-15) — evaluación empírica sobre 204 archivos
reales, corpus completo de la librería del usuario, sin llamadas de
red:** se corrió un harness aislado que llama a `aniparse.parse()` y
`anitopy.parse()` directo, leyendo el schema **correcto** de cada uno
(no el bug de arriba), y replica la lógica de merge de
`parsear_nombre_archivo` con el mapeo de aniparse ya corregido, para
medir qué pasaría si se arreglara `_parsed_dict_a_campos` tal cual.
Resultado desglosado por tipo de patrón (temporada explícita, episodio
absoluto sin marca de temporada, título dual-idioma, releases con
puntuación sin limpiar):

- **Episodio:** el merge corregido gana — 204/204 (100%) vs 203/204
  (99.5%) de anitopy solo. Los dos parsers tienen debilidades
  complementarias (ver causa raíz 4 abajo) y el merge las cubre sin
  introducir ninguna regresión nueva.
- **Temporada:** el merge corregido **empata** con anitopy solo en
  agregado (178/204, 87.3% ambos) pero eso esconde una **regresión real
  y reproducible**, no hipotética: en la categoría "episodio absoluto"
  (sin marcador de temporada), el merge corregido baja a 101/102 (99.0%)
  contra 102/102 (100%) de anitopy solo. La razón es concreta:

  ```
  Golden Kamuy Final Season - 07  (ground truth: temporada=None, episodio=7)
    aniparse : temporada=7,    episodio=None   ← lee el "07" como TEMPORADA, no episodio
    anitopy  : temporada=None, episodio=7       ← correcto
    merge    : temporada=7,    episodio=7       ← INCORRECTO -- prioridad ciega a aniparse
  ```

  La lógica de merge actual (`temp_combinada = sa if sa is not None else
  sb`) confía en la temporada de aniparse sin validar si tiene sentido.
  Arreglar solo el mapeo de claves, sin revisar también esta prioridad,
  cambia un acierto de anitopy por un error nuevo.
- **Título:** empatado 204/204 (100%) entre aniparse, anitopy y el
  merge — sin diferencia real (la brecha nominal de la primera pasada de
  este análisis era un artefacto del ground truth, no de los parsers).

**Las 17 discrepancias entre aniparse y anitopy en la muestra se reducen
a 4 causas raíz** (no son 17 problemas independientes):

1. **Confusión temporada/episodio** (Golden Kamuy, 1 archivo) — aniparse
   interpreta el número de episodio como temporada cuando el título dice
   "Final Season" sin dígito propio. Causa la regresión de arriba.
2. **Pérdida de "Nth Season" textual** (Hime-sama, Medalist, Vigilante —
   3 series) — cuando la temporada viene escrita como texto ("2nd
   Season") en vez de tag `SxxExx`, aniparse la deja pegada al título en
   vez de extraerla como número; anitopy sí la separa correctamente.
3. **Truncado de "Android" en el título** (serie "Does It Count If You
   Lose Your Innocence to an Android", 10 archivos) — aniparse corta
   sistemáticamente esa palabra del título en todos los episodios de
   esta serie (posible palabra en su wordlist interno tratada como
   ruido). No afecta episodio/temporada, solo el texto del título.
4. **Caso complementario Tojima/Toujima Kamen Rider** (2 archivos,
   mismo episodio real, dos releases) — anitopy falla completo
   (temporada=episodio=None) en la release con título tipo oración larga
   con puntos; aniparse falla completo en la release con guiones
   limpios invertidos. Cada parser cubre la debilidad del otro — este es
   el caso que sí justifica la redundancia por la que se adoptaron los
   dos parsers en primer lugar.

**Implementado.** El diseño final combina tres correcciones, cada una
validada por simulación aislada (individualmente y las tres juntas)
contra los 204 archivos reales antes de tocar `parsing.py`:

1. `_parsed_dict_a_campos` se separó en `_campos_desde_aniparse` (schema
   anidado correcto) y `_campos_desde_anitopy` (schema plano, sin cambios).
2. **Temporada** (Opción 2): se desconfía de la temporada de aniparse si
   coincide con el episodio que leyó anitopy (evita la regresión de la
   causa raíz 1 — confirmado, ya no ocurre).
3. **Título**: desempate invertido hacia anitopy — aniparse solo gana un
   empate de `_score_titulo` si lo supera estrictamente (evita el
   truncado de "Android", causa raíz 3).
4. **Episodio**: no se confía en el episodio de aniparse si su título
   quedó vacío (evita el caso "12345.mkv" — aniparse interpretaba el
   nombre de archivo puramente numérico como episodio, con su propia
   confianza interna en 0.0; ver `test_puramente_numerico_devuelve_titulo_numerico`
   en `tests/test_parsing.py`). Se investigó `_confidence` como señal
   general antes de esta decisión y se descartó: no se correlaciona de
   forma confiable con acierto/error en los 204 archivos reales (el
   promedio de confidence es incluso más alto en los casos incorrectos
   de temporada que en los correctos).

Simulación final combinada de las tres correcciones juntas sobre los 204
reales: **solo 1 archivo cambia** (Tojima, mejora limpia de temporada y
episodio, título sin cambios) — los otros 203 dan exactamente el mismo
resultado antes y después. Cubierto por
`tests/test_parsing.py::TestCaracterizacionFixAniparseSchemaYDesempate`
(Golden Kamuy y Android como guardas de no-regresión, Tojima como el
único caso que cambia, Frieren como representante de los 203 sin
cambios) y por la actualización de dos tests preexistentes que
resultaron afectados (`test_sin_temporada_con_tag_pegado_hevc10bit...`,
mejora real no anticipada; `test_puramente_numerico_devuelve_titulo_numerico`,
ahora protegido explícitamente en vez de "por accidente").

## Jikan estuvo caído (504) durante toda la simulación de 204 archivos del atajo por ID de AnimeThemes

Corriendo `scripts/simular_atajo_por_id_animethemes.py` sobre los 204
archivos reales del corpus (ver `docs/KNOWN_LIMITATIONS.md`, entrada del
atajo de resolución por ID externo), `GET https://api.jikan.moe/v4/anime`
devolvió `504 BadResponseException` ("Jikan failed to connect to
MyAnimeList") de forma persistente durante **toda** la corrida —
confirmado revisando la caché de la corrida (`get_api_cache()`): 0 claves
`jikan_search:*`, 44 claves `anilist_search:*`. Los 204 archivos
resolvieron título vía el respaldo de AniList (`jikan_resolver_titulo`
agotando reintentos → `anilist_buscar_titulo`), no vía Jikan directo.

No es parte de este cambio (el atajo por ID funcionó igual usando el
`picked_base` de AniList) y no se investigó si fue una caída puntual o
un patrón — pero es una señal empírica a favor de priorizar el ítem
pendiente de `docs/ideas.md` (#8/#9: fallback a AniList cuando Jikan
devuelve vacío, hoy solo cubre el caso de error de Jikan, no el de
respuesta vacía) más adelante: si Jikan cae con esta frecuencia en uso
real, ese gap importa más de lo que parecía cuando se documentó
originalmente.

## El .exe de PyInstaller debe construirse con un venv limpio, no con el Python de desarrollo

`ChapteriZen.spec` arma el ejecutable a partir de `run.py` (necesario
porque `chapterizen/__main__.py` usa imports relativos y PyInstaller no
puede apuntar directo a un módulo dentro de un paquete). Al construir la
primera vez contra el Python global de la máquina de desarrollo (con
`torch`, `pandas`, `scikit-learn`, `sqlalchemy`, `matplotlib`, `sympy`,
`dask`, `lightning`, `aiohttp` instalados para otros proyectos, ninguno
de ellos en `requirements.txt`), el `.exe` resultó de **294.8 MB**.

Causa confirmada: el hook de `librosa` de `pyinstaller-hooks-contrib`
usa `collect_submodules("librosa")`, que fuerza el análisis de **todos**
los submódulos de librosa aunque la app no los use — incluye
submódulos con imports condicionales opcionales (soporte experimental
de tensores vía `try: import torch`, decomposición NMF vía
`sklearn.decomposition`, etc.). Si esos paquetes están instalados
(aunque sean de otro proyecto), PyInstaller los arrastra enteros con su
cadena transitiva completa — en este caso incluyendo intentos fallidos
de bundlear DLLs de CUDA (`nvrtc64_120_0.dll`, `nvcuda.dll`).

Reconstruyendo con un venv nuevo (`python -m venv`) con **solo**
`requirements.txt` + `pyinstaller` instalados, el mismo `.spec` produjo
un `.exe` de **139.4 MB** — sin torch/pandas/sqlalchemy/matplotlib/
sympy/dask/lightning/aiohttp, y con el flujo completo (red + ffmpeg +
FFT/DTW real) verificado end-to-end sin diferencias de resultado
contra la build "sucia". `scikit-learn` sí quedó incluido en ambas —
es una dependencia transitiva real de librosa (confirmada por
`pip install -r requirements.txt` instalándola sola), no parte de la
contaminación.

**Implicación práctica:** cualquier rebuild futuro del `.exe` debe
partir de un venv nuevo con únicamente `requirements.txt` +
`pyinstaller` instalados — nunca del intérprete de desarrollo con
librerías acumuladas de otros proyectos.

## Excluir `scipy.optimize._highspy` del `.exe` rompe `librosa.sequence.dtw()` — no es una exclusión segura

Con el desglose de tamaño de `ChapteriZen.exe` (build `--onedir` de
diagnóstico), se identificó `scipy.optimize._highspy` (solver de
programación lineal/entera de `scipy.optimize.linprog`, ~7.4 MB) como
candidato a excluir vía `excludes=` en `ChapteriZen.spec` — ni
`audio_matching.py` ni las funciones de librosa que se llaman (`mfcc`,
`chroma_stft`, `sequence.dtw`, `util.normalize`) hacen programación
lineal, así que en teoría no debería hacer falta.

**Probado y revertido tras confirmar con la prueba de flujo completo
real** (video real → AnimeThemes → FFT/DTW → XML) que la exclusión
rompe el matching: con `scipy.optimize._highspy` excluido,
`librosa.sequence.dtw()` falla en tiempo de ejecución con
`ModuleNotFoundError: No module named 'scipy.optimize._highspy'`,
capturado por el `try/except` existente en `chapterizer_worker.py` que
cae a un score solo-FFT (peor, sin verificación cruzada por DTW) —
en la corrida de prueba, el OP directamente no se detectó (antes sí,
con score 0.766) y el ED bajó de score 0.778 a 0.287 (el mismo valor
que da el FFT solo, confirmando que el paso DTW nunca corrió).

**Causa:** `scipy/optimize/__init__.py` importa todo el árbol de
submódulos de `scipy.optimize` al inicializarse (no son imports
perezosos por función) — excluir cualquier submódulo individual de
`scipy.optimize` (no solo `_highspy`) tira abajo la inicialización del
subpaquete completo, aunque el código que realmente se ejecuta nunca
toque esa pieza puntual. Esto es distinto del caso de `librosa`
(collect_submodules fuerza el análisis pero cada submódulo se importa
independiente) — acá es el propio `scipy.optimize` el que no tolera
exclusiones parciales.

**Decisión:** no excluir ningún submódulo de `scipy.optimize` en
`ChapteriZen.spec`. Documentado explícitamente en el comentario del
`.spec` para que nadie reintente esta misma exclusión sin saber por
qué falla. `sklearn` (excluido en el mismo commit, ~12 MB) sí se
confirmó seguro con la misma prueba — no tiene esta clase de problema
de inicialización de paquete.
