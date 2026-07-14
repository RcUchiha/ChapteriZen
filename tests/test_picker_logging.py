"""
_pedir_pick es el unico punto de entrada real a un picker (need_pick.emit +
espera bloqueante) -- los tres tipos de picker (discrepancia Jikan/trace.moe,
seleccion de AnimeThemes, seleccion de Jikan) y sus 4 call sites en
ResolverWorker pasan todos por aqui, asi que instrumentar el log ahi cubre
los tres de forma uniforme sin duplicar logging en cada call site.
"""
from chapterizen.modelos import ParametrosTrabajo, PickRequest
from chapterizen.gui.resolver_worker import ResolverWorker


def _worker(tmp_path, interactivo=True):
    video = tmp_path / "Test Anime - 01.mkv"
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida="",
        crear_subcarpeta=False,
        usar_exacto=False,
        search_override="",
    )
    w = ResolverWorker(None, params, interactivo=interactivo)
    logs = []
    w.log.connect(logs.append)
    return w, logs


def _req(filas):
    return PickRequest(
        kind="jikan",
        titulo="Selecciona el anime correcto (Jikan/MAL)",
        subtitulo="Se encontraron múltiples resultados. Elige el correcto:",
        columnas=[("Título", 620), ("Tipo", 80)],
        filas=filas,
        payload=[{} for _ in filas],
    )


def test_pedir_pick_loguea_apertura_y_seleccion(tmp_path, monkeypatch):
    worker, logs = _worker(tmp_path)
    req = _req([["Anime A", "TV"], ["Anime B", "TV"]])

    # _wait_pick encapsula el bloqueo real via QWaitCondition (solo se
    # desbloquea cuando la UI llama a entregar_pick) -- lo reemplazamos aca
    # para probar el logging de _pedir_pick de forma sincronica y aislada.
    monkeypatch.setattr(worker, "_wait_pick", lambda: 1)

    idx = worker._pedir_pick(req)

    assert idx == 1
    assert any(
        l == "🖱️ Picker abierto: Selecciona el anime correcto (Jikan/MAL) (2 opciones)"
        for l in logs
    )
    assert any(l == "🖱️ Selección: Anime B | TV" for l in logs)
    # El log de apertura debe preceder al de selección.
    idx_abierto  = next(i for i, l in enumerate(logs) if l.startswith("🖱️ Picker abierto"))
    idx_eleccion = next(i for i, l in enumerate(logs) if l.startswith("🖱️ Selección"))
    assert idx_abierto < idx_eleccion


def test_pedir_pick_loguea_cancelacion(tmp_path, monkeypatch):
    worker, logs = _worker(tmp_path)
    req = _req([["Anime A", "TV"], ["Anime B", "TV"]])

    monkeypatch.setattr(worker, "_wait_pick", lambda: None)

    idx = worker._pedir_pick(req)

    assert idx is None
    assert any(l.startswith("🖱️ Picker abierto:") for l in logs)
    assert "🖱️ Picker cancelado por el usuario." in logs
    # No debe quedar log de "Selección" cuando se cancela.
    assert not any(l.startswith("🖱️ Selección") for l in logs)
