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
