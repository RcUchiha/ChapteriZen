# Tech Debt

Deuda técnica interna: código o estructura que funciona correctamente hoy
(no afecta el comportamiento observable para el usuario — para eso está
`docs/KNOWN_LIMITATIONS.md`) pero que amerita limpieza o reorganización
futura.

## `chapterizen.py` (monolito legacy en la raíz) diverge del paquete

`chapterizen.py` en la raíz del repo es el monolito original (v0.0.7) que
antecede al paquete `chapterizen/`. Ya no es el código que ejecuta la GUI
(`chapterizen/__main__.py` es el entry point actual), pero sigue presente
en el repo.

Ese archivo todavía usa su propia copia de `_aplicar_canon` (versión vieja,
sin el chequeo de variantes oficiales de título) en sus 3 call sites — la
versión equivalente en el paquete (`ResolverWorker._aplicar_canon_multivariante`,
`gui/resolver_worker.py`) sí prueba esas variantes desde el commit `895b51e`.
Es decir, monolito y paquete ya producen resultados distintos ante el mismo
input en el caso de variantes de idioma (ver `_variante_oficial_que_acepta`
en `resolver_worker.py`). La versión muerta de `_aplicar_canon` que vivía en
`chapterizen/jikan.py` (sin callers en el paquete, confirmado con grep) ya
se eliminó de ahí.

Esta divergencia es un motivo más (además de ser código duplicado sin
mantenimiento) para eliminar `chapterizen.py` cuando corresponda. No se
toca ahora — requiere confirmar antes que nada externo lo importe como
módulo.

## El sink de loguru se configura al importar `config.py`, sin distinguir GUI de tests/scripts

`config.py` llama a `logger.remove()` + `logger.add(_LOG_DIR / "chapterizen_{time:YYYY-MM-DD}.log", ...)`
directamente en el cuerpo del módulo (línea ~66), así que se ejecuta apenas
alguien hace `import chapterizen` (o cualquier import que dispare
`chapterizen/__init__.py`) — no solo cuando arranca la GUI real. Confirmado:
correr `pytest` escribe entradas DEBUG reales (incluyendo datos sintéticos
de fixtures, ej. títulos de prueba como "Attack on Titan"/"Shingeki no
Kyojin") en el mismo archivo rotativo de producción que usa la GUI
(`%LOCALAPPDATA%\ChapteriZen\ChapteriZen\Logs\` en Windows). Cualquier
script standalone que importe el paquete tiene el mismo problema salvo que
reemplace el sink explícitamente después del import (ver
`scripts/calibrar_trace_moe.py`, que hace justamente eso).

Mismo patrón que el bug de aislamiento de `_API_CACHE` (ver commit
`33e5611`, `get_api_cache()`), aplicado a logging en vez de caché: un
efecto secundario de proceso configurado incondicionalmente al importar un
módulo, sin diferenciar quién importa. No se toca ahora — requiere decidir
si `config.py` debe seguir configurando el sink en el import (y ofrecer un
modo de override explícito) o si esa responsabilidad debería moverse a
`__main__.py` (solo se configuraría al arrancar la GUI real).
