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
