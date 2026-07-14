# Known Limitations

## Fallback de AniList + temporada explícita en filename

Si Jikan está caído (agota reintentos) Y el nombre de archivo trae
temporada explícita (ej. "S02E05"), el resultado de AniList (claves
`id`/`idMal`) se pasa a jikan_resolver_temporada_por_sequel, que
espera el shape de respuesta de Jikan (clave `mal_id`) — esto
rompería en ese cruce específico. No cubierto en el fallback actual
(commit f2ff3c5) porque requiere una fase aparte para traducir entre
los dos formatos o para portar la navegación de secuela a AniList
también (como se discutió inicialmente para esta feature).

Este límite sigue abierto y es distinto del bug de títulos alternativos
descrito abajo (ya corregido) — este otro cruce (`mal_id` en
jikan_resolver_temporada_por_sequel) todavía no se toca.

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
