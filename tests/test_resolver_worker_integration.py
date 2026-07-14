"""
Suite de integracion de ResolverWorker.run() sobre el patron ya usado en
test_resolver_worker_anilist_fallback.py: respx mockeando las APIs reales
(Jikan/AniList/AnimeThemes) + worker.run() completo y sincronico -- no
funciones sueltas. Objetivo: que una regresion como el bug del repr(dict)
de AniList (corregido en 67b306b) se atrape en CI, no solo probando a
mano con videos reales.

Como se resuelve _wait_pick() sin bloquear (mecanismo, confirmado leyendo
_pedir_pick en gui/workers.py):

  _pedir_pick() hace self.need_pick.emit(req) y LUEGO llama a
  self._wait_pick(), que bloquea en una QWaitCondition hasta que alguien
  llama a self.entregar_pick(idx) (eso es lo que hace _on_need_pick en
  __main__.py cuando el usuario elige en el dialogo real).

  Los 3 tests existentes en test_resolver_worker_anilist_fallback.py NUNCA
  llegan a ejecutar _wait_pick(): usan usar_exacto=False (evita por
  completo el picker de AnimeThemes, que solo se alcanza si usar_exacto es
  True) y construyen la respuesta de Jikan para que no cruce el umbral de
  cross-verificacion (ts1_base >= 0.85) que dispara el picker de
  discrepancia. O sea, esos 3 tests no tienen un mecanismo para resolver
  picks -- simplemente nunca abren un picker.

  Para los escenarios nuevos que SI necesitan resolver un picker real, el
  truco es que pyqtSignal.connect() con un callable de Python, sin
  QApplication corriendo un event loop y sin cruzar threads (run() se
  llama de forma sincronica, no via .start()), usa conexion directa: el
  slot conectado se ejecuta DENTRO de la llamada a .emit(), antes de que
  esta retorne. Entonces:

      worker.need_pick.connect(lambda req: worker.entregar_pick(idx_deseado))

  hace que entregar_pick(idx_deseado) corra sincronicamente en cuanto
  _pedir_pick() llama a self.need_pick.emit(req) -- ANTES de que
  _pedir_pick() llegue a llamar a self._wait_pick(). Cuando _wait_pick()
  por fin se ejecuta, _pick_ready ya es True y el while no bloquea. Mismo
  patron ya usado (sin nombrarlo asi) por test_picker_logging.py, pero ahi
  se parcheaba _wait_pick() directamente en vez de pasar por need_pick.
  Aca se prefiere conectar need_pick porque es mas fiel a como lo hace
  __main__.py con el dialogo real.
"""
import httpx
import respx

from chapterizen.modelos import ParametrosTrabajo, AnimeDetectado
from chapterizen.gui.workers import ResolverWorker

JIKAN_ANIME        = "https://api.jikan.moe/v4/anime"
ANILIST_GRAPHQL    = "https://graphql.anilist.co"
ANIMETHEMES_SEARCH = "https://api.animethemes.moe/search"


def _worker(tmp_path, video_name, usar_exacto, interactivo=True):
    video = tmp_path / video_name
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida="",
        crear_subcarpeta=False,
        usar_exacto=usar_exacto,
        search_override="",
    )
    worker = ResolverWorker(None, params, interactivo=interactivo)

    logs = []
    worker.log.connect(logs.append)

    resultado = {}
    worker.resolved.connect(lambda p: resultado.update(ok=True, params=p))
    worker.failed.connect(lambda msg: resultado.update(ok=False, error=msg))

    return worker, logs, resultado


@respx.mock
def test_happy_path_sin_picker_ni_fallback(tmp_path):
    """Jikan devuelve un unico resultado confiable, AnimeThemes encuentra
    slug exacto en el primer intento -- no debe abrirse ningun picker ni
    activarse el respaldo de AniList/Jikan-via-picker."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 1, "title": "Attack on Titan", "type": "TV", "episodes": 25, "score": 8.5},
    ]}))
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Attack on Titan", "year": 2013, "season": None, "slug": "attack-on-titan"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv", usar_exacto=True)
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "attack-on-titan"
    assert resultado["params"].titulo_usado == "Attack on Titan"
    assert resultado["params"].episodio == 5

    assert not any("🖱️" in l for l in logs)
    assert not any("usando AniList como respaldo" in l for l in logs)
    assert not any("Respaldo: Jikan" in l for l in logs)
    assert not any("Reintentando con título alternativo" in l for l in logs)


@respx.mock
def test_discrepancia_jikan_trace_moe_abre_picker_y_usa_seleccion(tmp_path, monkeypatch):
    """Jikan devuelve 2 candidatos con el mismo texto (ts1=ts2=1.0 -> no
    confiable por falta de margen, pero ts1 >= 0.85 dispara la
    cross-verificacion con trace.moe). trace.moe/AniList identifican un
    titulo distinto -> discrepancia real -> se abre el picker; el usuario
    elige la opcion de Jikan (idx=0)."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 1, "title": "Attack on Titan", "title_english": "Attack on Titan",
         "score": 8.5, "type": "TV", "episodes": 25},
        {"mal_id": 2, "title": "Attack on Titan", "title_english": "Shingeki no Kyojin: Chronicle",
         "score": 7.0, "type": "Movie", "episodes": 1},
    ]}))
    respx.post(ANILIST_GRAPHQL).mock(return_value=httpx.Response(200, json={
        "data": {"Media": {"title": {"romaji": "Shingeki no Kyojin"}}}
    }))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv", usar_exacto=False)

    # _identificar_con_trace_moe hace extraccion real de fotogramas via ffmpeg --
    # fuera de alcance para esta suite orientada a red (HTTP). Se fija el
    # resultado que normalmente vendria de trace.moe + AniList.
    monkeypatch.setattr(
        worker, "_identificar_con_trace_moe",
        lambda video: AnimeDetectado(titulo="no-usado-en-este-call-site", anilist_id=999, episodio=5, similitud=0.93),
    )
    worker.need_pick.connect(lambda req: worker.entregar_pick(0))  # elige la fila de Jikan

    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].titulo_usado == "Attack on Titan"
    assert resultado["params"].episodio == 5

    assert "  - ⚠️ Verificación no concluyente: Jikan='Attack on Titan' / AniList='Shingeki no Kyojin'" in logs
    assert "🖱️ Picker abierto: Verificación de título — Jikan vs trace.moe (2 opciones)" in logs
    assert "🖱️ Selección: Jikan / MAL | Attack on Titan | ts1 = 100.00%" in logs


@respx.mock
def test_discrepancia_cancelada_propaga_failed_con_mensaje_claro(tmp_path, monkeypatch):
    """Mismo escenario que el anterior, pero el usuario cancela el picker
    de discrepancia. Comportamiento real confirmado leyendo el codigo y
    ejecutandolo (no asumido): _verificar_y_resolver_discrepancia lanza
    RuntimeError("Selección cancelada.") cuando _pedir_pick devuelve None;
    ese RuntimeError se re-lanza explicitamente (no se traga) en el
    try/except de la cross-verificacion dentro de run(), y llega intacto
    al try/except exterior de run(), que hace failed.emit(str(e)). Osea:
    NO es un resolved.emit() con valor por defecto ni un re-pregunta --
    es un failed.emit() limpio y explicito con el mismo mensaje en los
    3 pickers (ver los otros 2 "Selección cancelada." en workers.py)."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 1, "title": "Attack on Titan", "title_english": "Attack on Titan",
         "score": 8.5, "type": "TV", "episodes": 25},
        {"mal_id": 2, "title": "Attack on Titan", "title_english": "Shingeki no Kyojin: Chronicle",
         "score": 7.0, "type": "Movie", "episodes": 1},
    ]}))
    respx.post(ANILIST_GRAPHQL).mock(return_value=httpx.Response(200, json={
        "data": {"Media": {"title": {"romaji": "Shingeki no Kyojin"}}}
    }))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv", usar_exacto=False)
    monkeypatch.setattr(
        worker, "_identificar_con_trace_moe",
        lambda video: AnimeDetectado(titulo="no-usado-en-este-call-site", anilist_id=999, episodio=5, similitud=0.93),
    )
    worker.need_pick.connect(lambda req: worker.entregar_pick(None))  # cancela

    worker.run()

    assert resultado.get("ok") is False
    assert resultado.get("error") == "Selección cancelada."

    assert "🖱️ Picker abierto: Verificación de título — Jikan vs trace.moe (2 opciones)" in logs
    assert "🖱️ Picker cancelado por el usuario." in logs
    assert not any(l.startswith("🖱️ Selección") for l in logs)


@respx.mock
def test_anilist_fallback_con_animethemes_ambiguo_usa_titulos_limpios(tmp_path):
    """Regresion directa del bug corregido en 67b306b: Jikan agota
    reintentos (503 persistente), AniList responde con un item SIN
    'mal_id'. AnimeThemes no da match exacto para ninguna consulta ->
    se abre el picker de AnimeThemes con titulos alternativos. Antes del
    fix, esas consultas alternativas eran el repr() del dict de titulo de
    AniList; ahora deben ser strings limpios (romaji/english/synonyms)."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(503, json={"error": "down"}))
    respx.post(ANILIST_GRAPHQL).mock(return_value=httpx.Response(200, json={"data": {"Page": {"media": [
        {
            "id": 12345, "idMal": 999999,
            "title": {
                "romaji":        "Mato Seihei no Slave",
                "english":       "Slave of the Magic Capital's Guardian Fairy",
                "native":        None,
                "userPreferred": "Mato Seihei no Slave",
            },
            "synonyms": ["Guardian Fairy Slave"],
            "format": "TV", "status": "FINISHED", "episodes": 12,
            "startDate": {"year": 2023, "month": 1, "day": 1},
        }
    ]}}}))

    consultas_animethemes = []

    def _at_side_effect(request):
        q = request.url.params.get("q", "")
        consultas_animethemes.append(q)
        # Ninguna de estas 2 opciones matchea exacto ninguna consulta
        # (base ni alternativas) -- fuerza a abrir el picker.
        return httpx.Response(200, json={"search": {"anime": [
            {"name": "Mato Seihei no Slave TV",     "year": 2023, "season": None, "slug": "mato-seihei-no-slave-tv"},
            {"name": "Mato Seihei no Slave (2023)", "year": 2023, "season": None, "slug": "mato-seihei-no-slave-2023"},
        ]}})
    respx.get(ANIMETHEMES_SEARCH).mock(side_effect=_at_side_effect)

    worker, logs, resultado = _worker(tmp_path, "Mato Seihei no Slave - 01.mkv", usar_exacto=True)
    worker.need_pick.connect(lambda req: worker.entregar_pick(0))

    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "mato-seihei-no-slave-tv"
    assert resultado["params"].titulo_usado == "Mato Seihei no Slave TV"
    assert resultado["params"].episodio == 1

    assert any("Jikan no disponible, usando AniList como respaldo" in l for l in logs)
    assert "🖱️ Picker abierto: Selecciona el anime correcto (AnimeThemes) (2 opciones)" in logs
    assert "🖱️ Selección: Mato Seihei no Slave TV | 2023 |  | mato-seihei-no-slave-tv" in logs

    # El corazon de la regresion: ninguna consulta real enviada a
    # AnimeThemes debe ser el repr() de un dict.
    assert len(consultas_animethemes) >= 2
    assert not any(q.startswith("{") for q in consultas_animethemes), (
        f"consulta corrupta (repr de dict) enviada a AnimeThemes: {consultas_animethemes!r}"
    )
    assert "Mato Seihei no Slave" in consultas_animethemes
    assert "Slave of the Magic Capital's Guardian Fairy" in consultas_animethemes
    assert "Guardian Fairy Slave" in consultas_animethemes


@respx.mock
def test_jikan_y_anilist_agotan_reintentos_propaga_failed(tmp_path):
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(503, json={"error": "down"}))
    respx.post(ANILIST_GRAPHQL).mock(return_value=httpx.Response(503, json={"error": "also down"}))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv", usar_exacto=False)
    worker.run()

    assert resultado.get("ok") is False
    assert resultado.get("error")
    assert "503" in resultado["error"]
    assert any("Jikan no disponible, usando AniList como respaldo" in l for l in logs)
