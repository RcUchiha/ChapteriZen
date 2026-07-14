# Tech Debt

Deuda técnica interna: código o estructura que funciona correctamente hoy
(no afecta el comportamiento observable para el usuario — para eso está
`docs/KNOWN_LIMITATIONS.md`) pero que amerita limpieza o reorganización
futura.

## `_aplicar_canon` (chapterizen/jikan.py) es código muerto

Desde el commit `895b51e` ("fix: aceptar canon de temporada/título si
cualquier variante oficial pasa el chequeo de tokens..."), `ResolverWorker`
(`gui/resolver_worker.py`) reemplazó sus 3 llamadas a `_aplicar_canon` por
`self._aplicar_canon_multivariante(...)`, que además de lo que hacía
`_aplicar_canon` prueba las variantes oficiales de título (romaji/english/
native/userPreferred) antes de rechazar el canon.

Nada en el código llama a `_aplicar_canon` hoy (confirmado con grep). Sigue
definida en `jikan.py` con un comentario inline señalando esto, para que
nadie la modifique pensando que afecta el comportamiento real de la app.

**Candidata a eliminar** en un futuro commit de limpieza, una vez se
confirme que ningún test ni código externo depende de ella.
