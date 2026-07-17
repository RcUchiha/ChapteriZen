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
  llegan a ejecutar _wait_pick(): construyen la respuesta de Jikan para que
  no cruce el umbral de cross-verificacion (ts1_base >= 0.85) que dispara
  el picker de discrepancia, y mockean AnimeThemes con un resultado unico
  no ambiguo para que la resolucion de slug (que corre siempre desde que
  se elimino el checkbox "OP/ED exactos" y el campo usar_exacto) tampoco
  abra picker. O sea, esos 3 tests no tienen un mecanismo para resolver
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
import json

import httpx
import respx

from chapterizen.modelos import ParametrosTrabajo, AnimeDetectado
from chapterizen.gui.resolver_worker import ResolverWorker, _variante_oficial_que_acepta

JIKAN_ANIME        = "https://api.jikan.moe/v4/anime"
ANILIST_GRAPHQL    = "https://graphql.anilist.co"
ANIMETHEMES_SEARCH = "https://api.animethemes.moe/search"


def _worker(tmp_path, video_name, interactivo=True):
    video = tmp_path / video_name
    video.write_bytes(b"")
    params = ParametrosTrabajo(
        video=str(video),
        carpeta_salida="",
        crear_subcarpeta=False,
        search_override="",
    )
    worker = ResolverWorker(None, params, interactivo=interactivo)

    logs = []
    worker.log.connect(logs.append)

    resultado = {}
    worker.resolved.connect(lambda p: resultado.update(ok=True, params=p))
    worker.failed.connect(lambda msg: resultado.update(ok=False, error=msg))
    worker.cancelado.connect(lambda: resultado.update(ok=False, cancelado=True))

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

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv")
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
    # Mock minimo de AnimeThemes: resultado unico no ambiguo para "Attack on
    # Titan" (titulo elegido en el picker de discrepancia), para que la
    # resolucion de slug -- que ahora corre siempre -- no abra un segundo
    # picker.
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Attack on Titan", "year": 2013, "season": None, "slug": "attack-on-titan"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv")

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

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv")
    monkeypatch.setattr(
        worker, "_identificar_con_trace_moe",
        lambda video: AnimeDetectado(titulo="no-usado-en-este-call-site", anilist_id=999, episodio=5, similitud=0.93),
    )
    worker.need_pick.connect(lambda req: worker.entregar_pick(None))  # cancela

    worker.run()

    # La cancelacion ya no viaja por failed -- tiene su propia senal
    # cancelado(), separada para que la UI no muestre un popup de error
    # ante una accion intencional del usuario (ver __main__.py._on_cancelado).
    assert resultado.get("ok") is False
    assert resultado.get("cancelado") is True
    assert "error" not in resultado

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

    worker, logs, resultado = _worker(tmp_path, "Mato Seihei no Slave - 01.mkv")
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

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan - 05.mkv")
    worker.run()

    assert resultado.get("ok") is False
    assert resultado.get("error")
    assert "503" in resultado["error"]
    assert any("Jikan no disponible, usando AniList como respaldo" in l for l in logs)


def _mock_anilist_search_y_relations(media_busqueda: list, relaciones_por_id: dict):
    """Un solo mock de POST a ANILIST_GRAPHQL que enruta segun el shape de
    las variables del body: {'search': ...} -> query de busqueda
    (anilist_buscar_titulo), {'id': ...} -> query de relations
    (anilist_avanzar_a_secuela). Necesario porque ambas queries pegan al
    mismo endpoint y en estos tests se ejercitan las dos en la misma
    corrida de run()."""
    def _side_effect(request: httpx.Request) -> httpx.Response:
        body      = json.loads(request.content)
        variables = body.get("variables") or {}
        if "search" in variables:
            return httpx.Response(200, json={"data": {"Page": {"media": media_busqueda}}})
        if "id" in variables:
            edges = relaciones_por_id.get(variables["id"], [])
            return httpx.Response(200, json={"data": {"Media": {"relations": {"edges": edges}}}})
        return httpx.Response(200, json={"data": {}})
    respx.post(ANILIST_GRAPHQL).mock(side_effect=_side_effect)


def _anilist_media(anilist_id, romaji, episodes=12):
    return {
        "id": anilist_id,
        "idMal": anilist_id + 900000,
        "title": {"romaji": romaji, "english": None, "native": None, "userPreferred": romaji},
        "synonyms": [],
        "format": "TV", "status": "FINISHED", "episodes": episodes,
        "startDate": {"year": 2024, "month": 1, "day": 1},
    }


@respx.mock
def test_jikan_caido_anilist_navega_secuela_por_variante_ingles_pero_adopta_romaji(tmp_path):
    """Caso real reportado: Jikan agota reintentos, AniList resuelve el
    titulo base y el filename declara temporada explicita (S02) -> Camino
    A navega la cadena de secuelas. El filename usa el titulo
    occidentalizado ('Chained Soldier'), pero el canon de temporada 2 que
    devuelve AniList viene en romaji ('Mato Seihei no Slave 2') --
    AnimeThemes/AniList/MAL tratan el romaji como fuente de verdad, asi
    que el chequeo de tokens directo fallaria por idioma. Pero la
    variante inglesa del MISMO item ('Chained Soldier Season 2') si
    comparte tokens con el filename, asi que el canon se acepta via esa
    variante -- y sin embargo el titulo que se ADOPTA sigue siendo el
    romaji (nunca la variante que hizo pasar el chequeo), porque
    AnimeThemes indexa por romaji: adoptar la variante inglesa generaria
    mas reintentos fallidos alli, no menos. Debe quedar un log visible
    explicando por que se acepto un canon que a simple vista no comparte
    tokens con el filename, y no debe hacer falta ningun picker."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(503, json={"error": "down"}))
    _mock_anilist_search_y_relations(
        media_busqueda=[_anilist_media(100, "Mato Seihei no Slave", episodes=12)],
        relaciones_por_id={
            100: [{
                "relationType": "SEQUEL",
                "node": {
                    "id": 200, "idMal": 900200,
                    "title": {
                        "romaji":        "Mato Seihei no Slave 2",
                        "english":       "Chained Soldier Season 2",
                        "native":        None,
                        "userPreferred": "Mato Seihei no Slave 2",
                    },
                    "synonyms": [], "format": "TV", "status": "FINISHED", "episodes": 12,
                },
            }],
        },
    )
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Mato Seihei no Slave 2", "year": 2025, "season": None, "slug": "mato-seihei-no-slave-2"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Chained Soldier S02E03.mkv")
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].slug == "mato-seihei-no-slave-2"
    assert resultado["params"].titulo_usado == "Mato Seihei no Slave 2"
    assert resultado["params"].episodio == 3


@respx.mock
def test_camino_a_no_actualiza_picked_base_a_la_temporada_resuelta_CARACTERIZACION(tmp_path, monkeypatch):
    """Caracterizacion del comportamiento ACTUAL (bug, no un requisito):
    en Camino A, la navegacion de secuela guarda el resultado en
    `picked_season` (variable local de run()) pero nunca reasigna
    `picked_base` -- a diferencia de Camino B, que si lo hace
    (jikan_navegar_por_episodio/anilist_navegar_por_episodio reasignan
    picked_base explicitamente). Consecuencia real: `jikan_item`, que se
    arma como `picked_base if not override else None` y se pasa a
    _resolver_slug_con_picker para los titulos alternativos de respaldo,
    sigue siendo el item de la TEMPORADA 1 despues de que Camino A ya
    resolvio y adopto el titulo de la temporada 2.

    Mismo escenario real que el test anterior (Chained Soldier S02 ->
    Mato Seihei no Slave, secuela real id=200) pero interceptando
    _resolver_slug_con_picker para capturar el jikan_item con el que
    seria llamado, en vez de dejar que se resuelva contra AnimeThemes.

    Este test debe quedar en verde ANTES del fix (documenta el bug tal
    cual esta) y se actualiza cuando se agregue `picked_base =
    picked_season` en Camino A (gui/resolver_worker.py) -- en ese punto,
    jikan_item debe pasar a ser el de la temporada 2 (id=200), no el de
    la temporada 1 (id=100)."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(503, json={"error": "down"}))
    _mock_anilist_search_y_relations(
        media_busqueda=[_anilist_media(100, "Mato Seihei no Slave", episodes=12)],
        relaciones_por_id={
            100: [{
                "relationType": "SEQUEL",
                "node": {
                    "id": 200, "idMal": 900200,
                    "title": {
                        "romaji":        "Mato Seihei no Slave 2",
                        "english":       "Chained Soldier Season 2",
                        "native":        None,
                        "userPreferred": "Mato Seihei no Slave 2",
                    },
                    "synonyms": [], "format": "TV", "status": "FINISHED", "episodes": 12,
                },
            }],
        },
    )

    worker, logs, resultado = _worker(tmp_path, "Chained Soldier S02E03.mkv")

    llamadas = []

    def _capturar(consulta, temporada, jikan_item=None, **kwargs):
        llamadas.append((consulta, jikan_item))
        return "slug-no-usado", "nombre-no-usado"

    monkeypatch.setattr(worker, "_resolver_slug_con_picker", _capturar)

    worker.run()

    assert resultado.get("ok") is True

    assert len(llamadas) == 1
    consulta_recibida, jikan_item = llamadas[0]
    # consulta_base (el titulo) SI se actualizo a la temporada 2 -- eso ya
    # funciona hoy, via _aplicar_canon_multivariante.
    assert consulta_recibida == "Mato Seihei no Slave 2"

    assert jikan_item is not None
    # Bug actual: jikan_item (derivado de picked_base) sigue siendo la
    # temporada 1 (id=100), no la temporada 2 recien resuelta (id=200) --
    # pese a que el titulo ya refleja la temporada 2 correctamente.
    assert jikan_item["id"] == 100

    assert any("Jikan no disponible, usando AniList como respaldo" in l for l in logs)
    assert "• Resolviendo temporada de secuela…" in logs
    assert not any("🖱️" in l for l in logs)
    assert not any(l.startswith("  - ⚠️ Ignorando canon de temporada por recorte") for l in logs)
    assert (
        "  - Título del archivo coincide por variante inglés: "
        "'Chained Soldier Season 2' → adoptando título romaji/principal: 'Mato Seihei no Slave 2'"
    ) in logs


@respx.mock
def test_canon_de_secuela_anilist_rechazado_por_recorte_log_identico_a_jikan(tmp_path):
    """El SEQUEL de AniList existe y se navega correctamente, pero su
    titulo no comparte tokens significativos con el titulo base -- el
    gate vive en gui/workers.py (no en anilist.py, ver fix de esta misma
    fase), asi que el mensaje debe ser TEXTUALMENTE igual al que ya
    produce el camino de Jikan para el mismo caso.

    Importante (por que las 4 variantes de titulo van pobladas con texto
    genuinamente distinto, no None como en _anilist_media): con la logica
    multivariante (fix de esta misma fase), un campo en None se SALTA sin
    evaluarse -- si english/native/userPreferred quedaran en None aqui,
    este test pasaria por una razon equivocada (esas variantes nunca se
    habrian probado de verdad, no que se probaron y fallaron). Poblarlas
    con contenido no relacionado y verificar _variante_oficial_que_acepta
    directamente confirma que las 3 variantes no-romaji SI se evaluaron y
    NINGUNA coincidio -- no solo que el resultado final fue rechazo."""
    node_sin_relacion = {
        "id": 999, "idMal": 900999,
        "title": {
            "romaji":        "Completely Different Show",
            "english":       "Some Other Anime Entirely",
            "native":        "Mattaku Kankei Nai Bangumi",
            "userPreferred": "Some Other Anime Entirely",
        },
        "synonyms": [], "format": "TV", "status": "FINISHED", "episodes": 12,
    }
    # Fixture invalida si alguna de estas 3 variantes quedara vacia --
    # _variante_oficial_que_acepta las salta sin evaluarlas, y la
    # asercion de "ninguna coincide" de abajo dejaria de ser una prueba real.
    assert all(node_sin_relacion["title"][campo] for campo in ("english", "native", "userPreferred"))
    assert _variante_oficial_que_acepta("Attack on Titan", node_sin_relacion) is None

    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(503, json={"error": "down"}))
    _mock_anilist_search_y_relations(
        media_busqueda=[_anilist_media(100, "Attack on Titan", episodes=25)],
        relaciones_por_id={
            100: [{"relationType": "SEQUEL", "node": node_sin_relacion}],
        },
    )
    # Mock minimo de AnimeThemes: resultado unico no ambiguo para "Attack on
    # Titan" (el canon se rechaza por recorte, asi que el titulo usado sigue
    # siendo el base), para que la resolucion de slug -- que ahora corre
    # siempre -- no abra picker (este test no tiene need_pick cableado, asi
    # que un picker real quedaria esperando entregar_pick() para siempre).
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Attack on Titan", "year": 2013, "season": None, "slug": "attack-on-titan"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan S02E03.mkv")
    worker.run()

    assert resultado.get("ok") is True
    assert resultado["params"].titulo_usado == "Attack on Titan"  # canon rechazado -- se queda con el titulo base
    assert resultado["params"].episodio == 3

    assert (
        "  - ⚠️ Ignorando canon de temporada por recorte: "
        "'Attack on Titan' → 'Completely Different Show'"
    ) in logs


@respx.mock
def test_camino_a_navega_secuela_pese_a_jikan_ambiguo(tmp_path):
    """Test de caracterizacion (documenta el comportamiento ACTUAL, no un
    requisito) para la discrepancia encontrada al comparar el gate de
    Camino A contra el de _aplicar_canon_multivariante:

    _aplicar_canon_multivariante exige `titulo_confiable and titulo_resuelto`
    como primera condicion (jikan.py / resolver_worker.py) antes de
    siquiera considerar adoptar un canon. El gate de entrada a Camino A
    (linea ~365 de resolver_worker.py: "temporada >= 2 and picked_base and
    not temporada_fue_default") NO exige titulo_confiable -- asi que hoy,
    si el nombre de archivo declara temporada explicita (S02) pero Jikan
    devolvio resultados ambiguos (titulo_confiable=False) para el titulo
    base, Camino A igual navega la cadena de secuelas y aplica el canon
    de esa temporada, sin haber confirmado antes que el titulo base era
    siquiera el anime correcto.

    Este test debe seguir en verde despues de unificar Camino A con
    _aplicar_canon_multivariante, siempre que ese refactor fije
    titulo_confiable=True explicitamente en la llamada unificada (en vez
    de propagar el titulo_confiable real, que aqui es False) -- de lo
    contrario se estaria cambiando el comportamiento real de la app, no
    solo el estilo del codigo.

    Setup: Jikan busca "Attack on Titan" y devuelve 2 candidatos sin
    relacion alguna con el texto buscado (ts1 << 0.72 con ambos ->
    confiable=False garantizado, y por debajo de 0.85 asi que tampoco
    dispara la cross-verificacion con trace.moe). Ambos candidatos
    apuntan (via /relations) a la misma secuela mal_id=600, cuyo titulo
    "Attack on Titan Season 2" SI preserva los tokens del archivo -- el
    canon se acepta directo, sin picker ni variante de idioma de por medio."""
    respx.get(JIKAN_ANIME).mock(return_value=httpx.Response(200, json={"data": [
        {"mal_id": 501, "title": "Unrelated Show Alpha", "type": "TV", "episodes": 25, "score": 7.0},
        {"mal_id": 502, "title": "Unrelated Show Beta",  "type": "TV", "episodes": 24, "score": 6.9},
    ]}))
    relacion_sequel = {"data": [
        {"relation": "Sequel", "entry": [{"mal_id": 600, "type": "anime"}]},
    ]}
    respx.get(f"{JIKAN_ANIME}/501/relations").mock(return_value=httpx.Response(200, json=relacion_sequel))
    respx.get(f"{JIKAN_ANIME}/502/relations").mock(return_value=httpx.Response(200, json=relacion_sequel))
    respx.get(f"{JIKAN_ANIME}/600").mock(return_value=httpx.Response(200, json={"data": {
        "mal_id": 600, "title": "Attack on Titan Season 2", "episodes": 12,
    }}))
    # Mock minimo de AnimeThemes: resultado unico no ambiguo para "Attack on
    # Titan Season 2" (canon de secuela aceptado), para que la resolucion de
    # slug -- que ahora corre siempre -- no abra picker (este test tampoco
    # tiene need_pick cableado).
    respx.get(ANIMETHEMES_SEARCH).mock(return_value=httpx.Response(200, json={"search": {"anime": [
        {"name": "Attack on Titan Season 2", "year": 2017, "season": None, "slug": "attack-on-titan-season-2"},
    ]}}))

    worker, logs, resultado = _worker(tmp_path, "Attack on Titan S02E03.mkv")
    worker.run()

    assert resultado.get("ok") is True
    # El canon de temporada 2 se adopto pese a que Jikan quedo ambiguo --
    # esto es lo que se quiere confirmar, no necesariamente lo deseable.
    assert resultado["params"].titulo_usado == "Attack on Titan Season 2"
    assert resultado["params"].episodio == 3

    assert "  - ⚠️ Se encontraron varios animes con nombre similar — el resultado puede no ser exacto." in logs
    assert "• Resolviendo temporada de secuela…" in logs
    assert not any("🖱️" in l for l in logs)
    assert not any(l.startswith("  - ⚠️ Ignorando canon de temporada por recorte") for l in logs)
