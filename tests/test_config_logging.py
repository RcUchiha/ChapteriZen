"""
Confirma el fix del bug documentado en docs/TECH_DEBT.md: importar
chapterizen.config por si solo ya no configura ningun sink de archivo de
loguru -- eso quedo movido a configurar_logging_produccion(), que solo
llama __main__.main() al arrancar la GUI real.

Introspecciona logger._core.handlers (API privada de loguru, pero es el
unico modo real de preguntar "que sinks estan activos ahora mismo" sin
recurrir a reload/subprocess -- el propio test suite de loguru usa el
mismo patron) en vez de reload/subprocess: chapterizen.config ya fue
importado por conftest.py (y transitivamente por el resto de la suite)
mucho antes de que este test corra, asi que si el bug de import-time
existiera todavia, el sink de produccion ya estaria activo a esta altura
sin que este test necesite forzar un import fresco.
"""
from pathlib import Path

from loguru import logger

from chapterizen import config as cz_config


def _sink_de_archivo_bajo(directorio: Path):
    """Devuelve el FileSink activo cuyo path cae bajo `directorio`, o
    None si no hay ninguno."""
    directorio = directorio.resolve()
    for handler in logger._core.handlers.values():
        ruta = getattr(handler._sink, "_path", None)
        if ruta is None:
            continue
        try:
            if Path(ruta).resolve().is_relative_to(directorio):
                return handler._sink
        except (OSError, ValueError):
            continue
    return None


def test_importar_config_no_configura_ningun_sink_de_archivo():
    assert _sink_de_archivo_bajo(cz_config._LOG_DIR) is None


def test_configurar_logging_produccion_agrega_sink_de_archivo(tmp_path, monkeypatch):
    monkeypatch.setattr(cz_config, "_LOG_DIR", tmp_path)
    try:
        cz_config.configurar_logging_produccion()
        sink = _sink_de_archivo_bajo(tmp_path)
        assert sink is not None
        assert "chapterizen_" in sink._path
    finally:
        # No dejar el sink de este test activo para el resto de la suite.
        logger.remove()
