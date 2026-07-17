# Known Limitations

## [CORREGIDO] Fallback de AniList + temporada explícita en filename

Si Jikan estaba caído (agotaba reintentos) Y el nombre de archivo traía
temporada explícita (ej. "S02E05"), el resultado de AniList (claves
`id`/`idMal`) se pasaba a `jikan_resolver_temporada_por_sequel` /
`jikan_navegar_por_episodio`, que esperan el shape de respuesta de Jikan
(clave `mal_id`) — esto rompía con un `KeyError` en ese cruce específico.
No cubierto en el fallback original (commit f2ff3c5) porque requería una
fase aparte para portar la navegación de secuela a AniList también, como
se discutió inicialmente para esta feature.

Corregido: se agregaron `anilist_avanzar_a_secuela`,
`anilist_resolver_temporada_por_sequel` y `anilist_navegar_por_episodio`
(chapterizen/anilist.py), análogos a sus pares de Jikan pero usando la
query GraphQL de `relations` de AniList (`relationType == "SEQUEL"`,
confirmado que distingue de `SPIN_OFF`/`SIDE_STORY` igual que el campo
`relation` de Jikan). En `gui/workers.py`, los dos caminos de resolución
de temporada (Camino A: temporada explícita; Camino B: detección por
conteo de episodios) branchean por `"mal_id" in picked_base` para elegir
la función de Jikan o de AniList según corresponda.

Importante: ambas funciones de AniList son puramente mecánicas, igual
que sus pares de Jikan — no validan el título resultante internamente.
La decisión de aceptar o rechazar el canon resuelto
(`_aceptar_canon_sin_perder_tokens`) sigue viviendo únicamente en
`gui/workers.py`, en el mismo punto donde ya vivía para Jikan, así que el
mensaje de usuario (`"⚠️ Ignorando canon de temporada por recorte…"`) es
textualmente idéntico sin importar qué fuente resolvió el título.

Cubierto por `tests/test_anilist.py` (funciones de navegación en
aislamiento) y `tests/test_resolver_worker_integration.py`
(`test_jikan_caido_anilist_navega_secuela_por_variante_ingles_pero_adopta_romaji`
y `test_canon_de_secuela_anilist_rechazado_por_recorte_log_identico_a_jikan`).

## [CORREGIDO] Títulos alternativos corruptos al reintentar búsqueda en AnimeThemes

Cuando el título base venía de AniList (fallback de Jikan agotado, item sin
`mal_id`) y `_resolver_slug_con_picker` reintentaba la búsqueda en
AnimeThemes con "títulos alternativos", el código usaba siempre
`jikan_titulos_desde_item`, que asume que `item["title"]` es un string. En
AniList `item["title"]` es un dict (`{romaji, english, native,
userPreferred}`), así que el texto enviado a la búsqueda terminaba siendo
el `repr()` completo del dict (ej. `"{'romaji': 'Mato Seihei no Slave',
'english': ...}"`) en vez de una lista de títulos limpios.

Comportamiento observado en validación manual con este bug presente: la
búsqueda con ese texto corrupto simplemente no encontraba match exacto en
AnimeThemes, así que el flujo caía de forma segura al picker manual y el
usuario podía completar la selección sin problema — no "rompía
silenciosamente". El riesgo real no era un crash sino texto de búsqueda
corrupto sin garantía de que nunca coincidiera por accidente con algo en
otro caso.

Corregido: `_resolver_slug_con_picker` ahora distingue el shape del item
(`"mal_id" in jikan_item` → Jikan; si no, AniList) y usa
`anilist_titulos_desde_item` (chapterizen/anilist.py) para items de AniList,
que extrae romaji/english/native/userPreferred y sinónimos como strings
individuales limpios. Cubierto por
`test_titulos_alternativos_de_item_anilist_son_strings_limpios` en
tests/test_resolver_worker_anilist_fallback.py.

## Filenames con título dual-idioma en un solo string

Cuando el nombre de archivo incluye el título en dos idiomas concatenados
dentro del mismo string parseado (ej. "Chained Soldier (Mato Seihei no
Slave)", donde el parser no distingue el paréntesis como alt-título), la
búsqueda inicial en AniList/Jikan con ese texto completo no encuentra
resultados (confirmado en vivo contra AniList real: 0 resultados). Como
picked_base queda None, ni el gate multivariante (895b51e) ni la
navegación de secuela llegan a activarse — el flujo degrada con
seguridad (titulo_confiable=False, sin crashear) pero probablemente
termina en el picker manual con usar_exacto=True, perdiendo la
resolución automática.

No cubierto todavía. Posible solución futura: detectar el patrón
"Título (Título Alterno)" durante el parsing y probar cada mitad como
consulta separada, o extraer el paréntesis como alt-título candidato en
vez de tratarlo como parte literal del título.

## [CORREGIDO] Filenames con puntuación sin limpiar pasan `_titulo_es_usable` y no disparan el fallback a trace.moe

Cuando el nombre de archivo es una release con puntos como separador y
texto tipo "oración" pegado al título (ej. release real confirmada:
"Tojima.Wants.to.Be.a.Kamen.Rider.S01E19.I.Have.No.Regrets.Dying.as.a.Kamen.Rider..."),
anitopy no logra extraer temporada ni episodio (ambos quedan `None`) y el
título resultante conserva basura sin limpiar (ej. fragmentos como
"S01E19.I Have No Regrets..." pegados al título). Confirmado corriendo
`parsear_nombre_archivo()` directo sobre esta release real (ver
`scripts/comparacion_parsers.txt`, gitignoreado): el título sucio de todas
formas pasa `_titulo_es_usable()` (es largo, no coincide con tokens de
ruido conocidos), así que `ResolverWorker.run()` **no** activa el
fallback a trace.moe (ese fallback solo se dispara si el título se
considera inutilizable) — el query sucio se manda directo a Jikan/AniList
con `episodio=0` por defecto (ya que no se pudo extraer ninguno).

Contraejemplo en la misma muestra: la release japonesa del mismo episodio
("Toujima.Tanzaburou.wa.Kamen.Rider.ni.Naritai.S01E19...") sí parsea
correctamente (temporada=1, episodio=19, título limpio) — el problema es
específico del patrón "título en inglés como oración completa con puntos
como separador", no de todas las releases con puntos.

Corregido (commit `faff480`): se agregó `_titulo_tiene_artefacto_pegado`
(`chapterizen/parsing.py`), que detecta un dígito y una letra pegados sin
separador en el título (ej. "19.I" en el caso de arriba) — señal de que
un tag técnico quedó sin limpiar del todo tras normalizar
(`_normalizar_titulo_parser` solo convierte a espacio los puntos entre
DOS LETRAS, así que un punto entre un dígito y una letra no matchea ese
regex y queda pegado). `ResolverWorker.run()` ahora llama a esta función
además de `_titulo_es_usable()` para decidir si activar el fallback a
trace.moe.

Se descartó deliberadamente agregar este chequeo directamente dentro de
`_titulo_es_usable()`: esa función también la usa `parsear_nombre_archivo()`
internamente para decidir si confiar en el título elegido entre
aniparse/anitopy o caer a su propio regex de respaldo. Un primer intento
de tocar `_titulo_es_usable()` pasó la simulación sobre 204 archivos
reales (0 falsos positivos) pero generó una regresión real no capturada
por esa simulación: para el archivo de prueba, el título con el patrón
detectado desviaba a `parsear_nombre_archivo()` hacia su regex de
respaldo interno, que producía un título igual de imperfecto pero **sin
ningún dígito pegado** (le dejaba pegado el tag de plataforma/release en
vez del de episodio) — evitando que el chequeo se disparara de nuevo en
`ResolverWorker.run()` y frustrando el objetivo real de activar trace.moe.
La función separada evita ese efecto secundario por completo.

Cubierto por `tests/test_parsing.py::TestTituloTieneArtefactoPegado`
(detecta el caso real, confirma que `parsear_nombre_archivo()` sigue
devolviendo el resultado del merge sin caer a `fallback`, y no-regresión
con un título real sin dígitos pegados).

## Cuota de búsquedas de trace.moe: anónima, por IP, sin reseteo documentado

trace.moe limita las búsquedas anónimas por dirección IP (o prefijo /64
en IPv6). Confirmado consultando `GET https://api.trace.moe/me`, que
devuelve `{id, priority, concurrency, quota, quotaUsed}`. El 2026-07-15,
tras una sesión de calibración con volumen alto (~82 videos entre dos
tandas, ver `docs/TECH_DEBT.md` — evaluación de `_TRACE_UMBRAL_CONFIANZA_ALTA`),
devolvió `quota: 100, quotaUsed: 100` — cuota agotada. El endpoint `/me`
**no incluye ningún campo de reseteo** (`resetAt`, `period`, o similar) —
no hay forma de saber desde la API si la cuota es diaria, mensual, o de
otro tipo; no se debe asumir ninguno de esos mecanismos sin evidencia.

Cuando la cuota se agota, `/search` devuelve `402 Payment Required` (no
`429`) — `_es_error_transitorio` (`config.py`) no lo trata como
transitorio, así que `_reintento_http` no reintenta y el fallo es
inmediato. Con el fix de logging de `identificar_anime_con_fotogramas`
(commit `7f733ae`), este 402 ahora queda registrado a nivel DEBUG cuando
ocurre (antes se descartaba en silencio con `except Exception: pass`,
sin dejar ningún rastro de la causa real).

Relevancia para un usuario real: en uso normal de la GUI (un video a la
vez, no un batch de decenas) es poco probable agotar la cuota — este
límite se descubrió durante una calibración deliberadamente intensiva,
no en uso típico. Pero si un usuario reporta que trace.moe "dejó de
identificar nada" después de procesar muchos episodios en poco tiempo,
este es el primer lugar para revisar: consultar `/me` manualmente y
cruzar contra el log de DEBUG para confirmar si la causa real fue 402.

## `_THEMES_DIR` (audio OP/ED descargado) crece sin ningún límite

`_THEMES_DIR` (`%LOCALAPPDATA%\ChapteriZen\Cache\themes` en Windows) guarda
el audio OGG/WAV de cada tema descargado de AnimeThemes, organizado por
slug de serie (`construir_cache_temas`, `chapterizen/animethemes.py`). A
diferencia de la caché de API (`get_api_cache()`, ver entrada de arriba
sobre la cuota de trace.moe — esa sí está acotada, ver más abajo), esta
carpeta **no tiene ninguna política de limpieza por tamaño ni por
antigüedad**.

El único `unlink()` que existe en `construir_cache_temas` borra los
archivos de una serie puntual solo cuando su nombre cambió en
AnimeThemes (invalidación de metadata desactualizada, no limpieza por
espacio). No hay TTL, no hay límite de tamaño total, no hay eviction de
series que ya no se vuelven a procesar — cada serie nueva que pasa por
`usar_exacto=True` agrega su OP/ED a esta carpeta para siempre.

Estimación (sin medición exhaustiva, orden de magnitud): ~3 temas
promedio por serie (OGG + WAV, ~90s c/u) ≈ **15-20 MB por serie
procesada**, sin techo. Para uso normal (decenas de series a lo largo de
meses) esto es modesto (cientos de MB); para un usuario muy activo con
cientos de series distintas, podría acumular varios GB con el tiempo sin
que nada lo purgue.

**`get_api_cache()` (el diskcache de respuestas JSON de Jikan/AniList/
AnimeThemes + features de audio precalculadas) NO tiene este problema**:
`Cache(_DC_PATH)` (`config.py`) usa los defaults de `diskcache` sin
sobreescribirlos — confirmado en el código real de la librería instalada
(`DEFAULT_SETTINGS`), no asumido: `size_limit=1073741824` (1 GiB),
`eviction_policy='least-recently-stored'`, `cull_limit=10`, y
`Cache.set()` invoca `_cull(...)` automáticamente en cada escritura
(borra expirados primero, y por política LRU si se supera el tamaño
límite). No hace falta ningún cambio ahí.

No cubierto todavía. Si se decide atacar, es una investigación aparte
específica para `_THEMES_DIR` (política de limpieza por tamaño total o
antigüedad de uso) — no relacionada con `get_api_cache()`, que ya
resuelve esto por sí solo vía `diskcache`.

## El atajo de resolución por ID externo de AnimeThemes no protege contra una identificación previa equivocada

`_resolver_slug_con_picker` (`gui/resolver_worker.py`) intenta primero
resolver el slug consultando AnimeThemes directo por recurso externo
(`filter[has]=resources&filter[site]=MyAnimeList|Anilist&filter[external_id]=...`,
vía `animethemes_buscar_por_id_externo`), usando el `mal_id`/`id` de
AniList que Jikan o AniList ya resolvieron (`picked_base`, o
`detectado_anilist_id` cuando viene de trace.moe con alta confianza).
Si hay exactamente un resultado y su nombre no pierde tokens contra
alguno de los títulos que Jikan/AniList ya conocen para ese mismo item
(`_token_ok_contra_titulos_conocidos`, incluye título japonés/native),
se acepta sin picker ni búsqueda de texto.

**Esta validación protege un caso específico: que el recurso externo
esté mal enlazado *dentro de AnimeThemes* (el `external_id` apunta a una
página de anime distinta de la que Jikan/AniList conocen para ese
mismo ID) — una inconsistencia de datos entre dos fuentes
independientes.** Confirmado con simulación real sobre 204 archivos
(`scripts/simular_atajo_por_id_animethemes.py`): la validación contra
títulos conocidos de `picked_base` (en vez de contra el texto crudo del
filename) rescata el 100% de los 95 falsos rechazos por diferencia de
idioma que producía una primera versión más simple de la validación,
sin ninguna regresión.

**Lo que esta validación NO cubre**: si `picked_base` ya viene mal
identificado desde una capa *anterior* del pipeline (el escenario que
motivó pedir esta validación — ej. una identificación de baja confianza
vía trace.moe que resolvió la serie equivocada, como el caso real de
Ingoku Danchi documentado más abajo en la cuota de trace.moe), el
nombre que devuelve AnimeThemes y los "títulos conocidos" usados para
validar provienen del mismo `picked_base` erróneo — coinciden entre sí
igual, sin que esta validación lo note. En ese escenario, el atajo por
ID aceptaría con la misma confianza (falsa) que antes se hubiera visto
en el propio `picked_base`, sin ningún picker de por medio que le diera
al usuario la oportunidad de corregirlo manualmente — mientras que el
camino de texto anterior, al menos, podía terminar en un picker si
AnimeThemes no encontraba un match exacto.

Confirmado empíricamente que esta brecha no tiene evidencia real
todavía: en la simulación de 204 archivos, cero casos tenían un
`picked_base` genuinamente mal identificado (todos los filenames eran
legibles y se resolvieron bien por texto) — de hecho, Jikan estuvo
caído durante toda esa corrida (ver `docs/TECH_DEBT.md`), así que los
204 pasaron por AniList como respaldo sin ningún caso de identificación
incorrecta que probar. No cubierto todavía. Si se decide atacar, es una
investigación aparte sobre cómo validar `picked_base` en sí mismo antes
de que llegue a esta función (una capa anterior del pipeline, no
relacionada con `_resolver_slug_con_picker`).

## Releases que omiten el apóstrofe en el filename pueden hacer que Jikan/AniList no encuentren match

Confirmado con un caso real: el filename
`You.Cant.Be.In.a.Rom-Com.with.Your.Childhood.Friends.S01E07...mkv`
genera la consulta `'You Cant Be In a Rom-Com with Your Childhood
Friends'` (parsing correcto y fiel al filename — no hay ningún bug de
`parsear_nombre_archivo()` acá). El título oficial real es *"You **Can't**
Be In a Rom-Com with Your Childhood Friends!"* — el filename del grupo
de release ya omite el apóstrofe (convención común para evitar
caracteres especiales en nombres de archivo).

Probado en vivo contra AniList: la consulta sin apóstrofe devuelve 0
resultados; restaurando únicamente el apóstrofe ("Can't" en vez de
"Cant", sin agregar el "!" final) encuentra el match exacto e
inequívoco de inmediato. Es decir, la pérdida de un solo carácter en el
filename alcanza para que la búsqueda de AniList no encuentre nada.

No se considera un bug a corregir: el picker de AnimeThemes (15
resultados, selección manual) ya cubre este caso como red de
seguridad — el usuario resolvió el episodio sin problema. Una
heurística de reinserción de apóstrofes sería frágil (requeriría un
diccionario de contracciones inglesas, con beneficio angosto frente al
riesgo de falsos positivos) y no se implementa. Esta nota es
informativa, para quien audite el log más adelante y se pregunte por
qué ese episodio puntual necesitó selección manual.
