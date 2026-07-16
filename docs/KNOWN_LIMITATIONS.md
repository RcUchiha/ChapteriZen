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
