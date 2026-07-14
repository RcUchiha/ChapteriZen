"""Workers de QThread: resolucion de titulo/slug (ResolverWorker) y
generacion de chapters via matching de audio (ChapterizerWorker). Movido
sin cambios desde chapterizen.py (monolito original, v0.0.7)."""
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from ..modelos import (
    ParametrosTrabajo,
    PickRequest,
    AnimeDetectado,
    TemaAudio,
    CandidatoFFT,
    ResultadoCoincidencia,
)
from ..ffmpeg_utils import (
    log_clv,
    asegurar_ffmpeg,
    duracion_con_ffprobe,
    extraer_fotogramas_centrado,
    extraer_audio_wav_mono_16k,
    leer_pcm16_mono_wav,
)
from ..config import _es_error_transitorio
from ..trace_moe import _TRACE_UMBRAL_RAPIDO, identificar_anime_con_fotogramas
from ..anilist import anilist_buscar_titulo, anilist_titulo_por_id, anilist_titulos_desde_item
from ..animethemes import (
    buscar_anime_en_animethemes,
    obtener_anime_de_animethemes,
    construir_mapa_mostrar_temas,
    construir_cache_temas,
)
from ..audio_matching import (
    _W_DTW,
    _W_FFT,
    _SR_FEATURES,
    _HOP_LENGTH,
    _TOP_K_FFT,
    _FFT_PRUNING_MIN,
    _SLIDE_WIN_SEC,
    _SLIDE_STEP_SEC,
    _SLIDE_OP_MAX,
    _SLIDE_ED_MAX,
    _fft_score,
    _dtw_score,
    obtener_features_con_cache,
    formatear_tiempo,
    _tiempo_sin_ms,
)
from ..chapters_xml import guardar_chapters, chapters_heuristicos
from ..parsing import (
    inferir_consulta_desde_nombre_archivo,
    quitar_sufijo_episodio,
    quitar_marcador_temporada,
    _titulo_es_usable,
    _preferir_resultados_por_temporada,
)
from ..jikan import (
    extraer_temporada_y_episodio_desde_nombre_archivo,
    jikan_resolver_titulo,
    jikan_resolver_temporada_por_sequel,
    jikan_navegar_por_episodio,
    jikan_buscar_anime,
    jikan_titulos_desde_item,
    _aceptar_canon_sin_perder_tokens,
    _aplicar_canon,
    _comparar_titulos_para_verificacion,
    animethemes_coincidencia_exacta_por_titulo,
    filtrar_por_token_obligatorio,
)
from ..naming import construir_ruta_salida


class _BaseWorker(QThread):
    """Base compartida para todos los workers — provee _log con nivel automático."""
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)

    def _log(self, s: str):
        self.log.emit(s)
        if s.startswith("  - ⚠️") or s.startswith("⚠️"):
            logger.warning(s)
        elif s.startswith("❌"):
            logger.error(s)
        else:
            logger.info(s)


class ResolverWorker(_BaseWorker):
    """Fase de resolución de nombre/slug, ejecutada en un QThread separado.

    Mezcla tres fases de forma intencional: (1) inferencia del título desde el
    filename, (2) cross-verificación con Jikan/AniList/trace.moe y (3) resolución
    del slug de AnimeThemes. Comparten estado intermedio (consulta_base,
    picked_base, titulo_confiable) que se construye progresivamente entre fases.

    Para añadir una tercera fuente de verificación: extender el literal `origen`
    en _verificar_y_resolver_discrepancia y añadir la rama if/else correspondiente.
    """
    need_pick  = pyqtSignal(object)
    resolved   = pyqtSignal(object)
    failed     = pyqtSignal(str)

    def __init__(self, ventana, params: ParametrosTrabajo, interactivo: bool = True):
        super().__init__(ventana)
        self.ventana     = ventana
        self.params      = params
        self.interactivo = interactivo

        self._mx     = QMutex()
        self._cv     = QWaitCondition()
        self._cancel = False

        self._pick_index: Optional[int] = None
        self._pick_ready: bool          = False

    def cancelar(self):
        self._mx.lock()
        self._cancel = True
        self._cv.wakeAll()
        self._mx.unlock()

    def entregar_pick(self, idx: Optional[int]):
        self._mx.lock()
        self._pick_index = idx
        self._pick_ready = True
        self._cv.wakeAll()
        self._mx.unlock()

    def _wait_pick(self) -> Optional[int]:
        self._mx.lock()
        while not self._pick_ready and not self._cancel:
            self._cv.wait(self._mx)
        idx    = self._pick_index
        cancel = self._cancel
        self._mx.unlock()
        return None if cancel else idx

    def _pedir_pick(self, req: PickRequest) -> Optional[int]:
        self._mx.lock()
        self._pick_index = None
        self._pick_ready = False
        self._mx.unlock()
        self.need_pick.emit(req)
        return self._wait_pick()

    def run(self):
        try:
            p     = self.params
            video = p.video

            _sep = "=" * 64
            logger.debug(_sep)
            logger.debug(f"NUEVO EPISODIO: {Path(video).stem}")
            logger.debug(_sep)

            temporada_raw, ep = extraer_temporada_y_episodio_desde_nombre_archivo(video)
            episodio           = int(ep or 0)
            temporada_fue_default = temporada_raw is None
            temporada          = 1 if temporada_raw is None else int(temporada_raw)

            self._log(f"• Analizando: {Path(video).name}")
            self.progress.emit(5)
            log_clv(logger.debug, "parsed", temporada=temporada, episodio=episodio)

            override             = (p.search_override or "").strip()
            anilist_confirmado   = False
            detectado_anilist_id = None

            if override:
                consulta_base = override
                self._log("• Usando nombre de búsqueda (anulación) desde interfaz…")
                log_clv(logger.debug, "override", q=consulta_base)
            else:
                consulta_base    = inferir_consulta_desde_nombre_archivo(video)
                consulta_base    = quitar_sufijo_episodio(consulta_base)
                _via_trace_moe   = False

                if not _titulo_es_usable(consulta_base) or re.fullmatch(r'\d+', consulta_base.strip()):
                    self._log("⚠️ Nombre de archivo sin título reconocible. Identificando con trace.moe…")
                    detectado      = self._identificar_con_trace_moe(video)
                    consulta_base  = quitar_sufijo_episodio(detectado.titulo)
                    _via_trace_moe = True
                    if detectado.episodio and episodio == 0:
                        episodio = detectado.episodio
                    self._log(f"  - Anime identificado: {consulta_base!r} (similitud={detectado.similitud:.2%})")
                    if detectado.similitud < 0.85:
                        self._log("  - ⚠️ Similitud baja. Verifica que el resultado sea correcto.")
                    if detectado.anilist_id is not None:
                        if detectado.similitud >= _TRACE_UMBRAL_RAPIDO:
                            anilist_confirmado   = True
                            detectado_anilist_id = detectado.anilist_id
                        else:
                            self._log(
                                f"  - AniList ID {detectado.anilist_id} disponible"
                                " pero no usado (confianza insuficiente) — continuando con Jikan"
                            )

                consulta_jikan = quitar_marcador_temporada(consulta_base)

                # Valores por defecto para cuando Jikan se omite
                titulo_resuelto  = consulta_base
                picked_base = None
                titulo_confiable     = False

                if anilist_confirmado:
                    # anilist_confirmado solo es True cuando similitud >= _TRACE_UMBRAL_RAPIDO
                    # (actualmente 0.95). Por debajo de ese umbral, picked_base queda en None
                    # pero el flujo igual pasa por Jikan — no se omite por baja confianza.
                    # Limitación conocida: cuando se omite Jikan, picked_base es None y el
                    # bloque de resolución de secuelas (jikan_resolver_temporada_por_sequel)
                    # tampoco se ejecuta. Hoy es aceptable porque trace.moe se activa cuando
                    # el filename no tiene título reconocible, así que `temporada` suele ser
                    # 1 por defecto. Si en el futuro un archivo lleva temporada en el nombre
                    # Y cae a trace.moe por otro motivo, ese caso quedará sin resolución de
                    # secuela automática.
                    self._log(
                        f"  - Título confirmado por AniList (ID {detectado_anilist_id})"
                        " — sin búsqueda adicional de nombre con Jikan"
                    )
                else:
                    self._log("• Jikan (base)…")
                    try:
                        titulo_resuelto, picked_base, titulo_confiable, ts1_base = jikan_resolver_titulo(consulta_jikan)
                    except Exception as e:
                        if not _es_error_transitorio(e):
                            raise
                        self._log("  - ⚠️ Jikan no disponible, usando AniList como respaldo…")
                        titulo_resuelto, picked_base, titulo_confiable, ts1_base = anilist_buscar_titulo(consulta_jikan)
                    log_clv(logger.debug, "jikan_query", q=consulta_jikan, from_base=consulta_base,
                            origen="trace_moe" if _via_trace_moe else "filename")
                    log_clv(
                        logger.debug, "jikan_base_result",
                        canon=titulo_resuelto, ok=titulo_confiable,
                        mal_id=(picked_base or {}).get("mal_id"),
                    )
                    if picked_base:
                        self._log(
                            f"  - Título del anime: {titulo_resuelto!r}"
                            f" ({'confirmado' if titulo_confiable else 'por confirmar'})"
                        )
                        if not titulo_confiable:
                            self._log(
                                "  - ⚠️ Se encontraron varios animes con nombre similar"
                                " — el resultado puede no ser exacto."
                            )
                    else:
                        self._log("  - Jikan no encontró resultados.")

                    # Los dos bloques de cross-verificación que siguen son intencionalmente
                    # paralelos y no se unifican: su paso de recopilación de datos difiere
                    # fundamentalmente (_via_trace_moe=False llama trace.moe en el acto;
                    # _via_trace_moe=True reutiliza el anilist_id ya disponible sin nueva red).

                    # Cross-verificación con trace.moe — solo cuando:
                    # · Jikan encontró algo (ts1 alto) pero no fue confiable (varios candidatos cercanos)
                    # · El filename tenía título reconocible (_via_trace_moe=False)
                    if (
                        not titulo_confiable
                        and not _via_trace_moe
                        and ts1_base >= 0.85
                        and picked_base is not None
                    ):
                        try:
                            self._log("  - Verificando automáticamente con trace.moe…")
                            detectado_xv   = self._identificar_con_trace_moe(video)
                            titulo_anilist = None
                            if detectado_xv.anilist_id is not None:
                                try:
                                    titulo_anilist = anilist_titulo_por_id(detectado_xv.anilist_id)
                                except Exception:
                                    pass
                            if titulo_anilist:
                                titulo_resuelto, picked_base, titulo_confiable = (
                                    self._verificar_y_resolver_discrepancia(
                                        titulo_resuelto, picked_base, ts1_base,
                                        titulo_anilist, detectado_xv.similitud,
                                        "cross_verification",
                                    )
                                )
                        except RuntimeError:
                            raise
                        except Exception as e:
                            self._log(
                                f"  - ⚠️ Cross-verificación con trace.moe falló: {e}"
                                " — manteniendo resultado Jikan."
                            )

                    # Verificación por ID reutilizado — solo cuando trace.moe ya fue
                    # llamado (confianza media, < _TRACE_UMBRAL_RAPIDO) y Jikan quedó
                    # ambiguo. No hace una segunda llamada a trace.moe: reutiliza el
                    # anilist_id del detectado original y consulta AniList por título
                    # (operación liviana, cacheada). Cubre el caso donde trace.moe
                    # identificó el anime correcto pero con similitud insuficiente para
                    # anilist_confirmado, y Jikan después eligió la entrada equivocada.
                    if (
                        not titulo_confiable
                        and _via_trace_moe
                        and picked_base is not None
                        and detectado.anilist_id is not None
                    ):
                        try:
                            titulo_anilist_rv = anilist_titulo_por_id(detectado.anilist_id)
                            if titulo_anilist_rv:
                                titulo_resuelto, picked_base, titulo_confiable = (
                                    self._verificar_y_resolver_discrepancia(
                                        titulo_resuelto, picked_base, ts1_base,
                                        titulo_anilist_rv, detectado.similitud,
                                        "id_reutilizado",
                                    )
                                )
                        except RuntimeError:
                            raise
                        except Exception as e:
                            self._log(
                                f"  - ⚠️ Verificación por ID AniList falló: {e}"
                                " — manteniendo resultado Jikan."
                            )

                    # Resolución de temporada — dos caminos mutuamente excluyentes.
                    if temporada >= 2 and picked_base and not temporada_fue_default:
                        # Camino A: filename declaró temporada explícita → navegar
                        # la cadena de secuelas hasta llegar a ese número de temporada.
                        try:
                            self._log("• Jikan (resolviendo temporada secuela)…")
                            picked_season = jikan_resolver_temporada_por_sequel(picked_base, temporada)
                            canon_season  = (
                                (picked_season.get("title") or "").strip()
                                or titulo_resuelto or consulta_base
                            )
                            if canon_season and _aceptar_canon_sin_perder_tokens(consulta_base, canon_season):
                                consulta_base = canon_season
                            else:
                                self._log(
                                    f"  - ⚠️ Ignorando canon de temporada por recorte: "
                                    f"{consulta_base!r} → {canon_season!r}"
                                )
                        except Exception as e:
                            self._log(f"  - ⚠️ Secuela falló: {e}. Usando canon base si está disponible.")
                            consulta_base = _aplicar_canon(consulta_base, titulo_resuelto, titulo_confiable)
                    elif temporada_fue_default and picked_base is not None and episodio > 0:
                        # Camino B: filename no declaró temporada → detectar por conteo
                        # de episodios. Si ep supera el total de S1, avanzar por secuelas.
                        eps_temporada = (picked_base or {}).get("episodes")
                        try:
                            eps_temporada = int(eps_temporada) if eps_temporada else 0
                        except (TypeError, ValueError):
                            eps_temporada = 0
                        if eps_temporada > 0 and episodio > eps_temporada:
                            try:
                                self._log(
                                    f"  - ℹ️ Ep. {episodio} supera los {eps_temporada} episodios de la primera temporada"
                                    " — detectando temporada automáticamente…"
                                )
                                picked_base, episodio, temporada = jikan_navegar_por_episodio(
                                    picked_base, episodio
                                )
                                titulo_resuelto = (picked_base.get("title") or "").strip() or titulo_resuelto
                                self._log(
                                    f"  - Reasignado a temporada {temporada}, episodio {episodio} — {titulo_resuelto!r}"
                                )
                            except Exception as e:
                                self._log(
                                    f"  - ⚠️ Navegación por episodio fallida: {e}"
                                    " — usando episodio original."
                                )
                        consulta_base = _aplicar_canon(consulta_base, titulo_resuelto, titulo_confiable)
                    else:
                        consulta_base = _aplicar_canon(consulta_base, titulo_resuelto, titulo_confiable)

            if not p.usar_exacto:
                p.slug         = ""
                p.episodio     = episodio
                p.titulo_usado = consulta_base
                self.progress.emit(30)
                self.resolved.emit(p)
                return

            # Pasar el item de Jikan para que _resolver_slug_con_picker pueda
            # buscar con títulos alternativos si la consulta base no da resultado.
            # Limitación conocida: cuando anilist_confirmado es True, picked_base es None
            # (Jikan fue omitido), así que jikan_item también es None. Si el título que
            # devuelve AniList no encuentra slug en AnimeThemes, se irá directo al picker
            # interactivo sin haber intentado los títulos alternativos que Jikan hubiera
            # aportado (jikan_titulos_desde_item). En el camino normal (sin AniList) ese
            # respaldo sí existe.
            jikan_item = picked_base if not override else None
            slug, titulo_usado = self._resolver_slug_con_picker(
                consulta_base, temporada, jikan_item=jikan_item
            )
            self.progress.emit(30)
            p.slug         = slug
            p.titulo_usado = titulo_usado
            p.episodio     = episodio
            self.resolved.emit(p)

        except Exception as e:
            self.failed.emit(str(e))

    def _identificar_con_trace_moe(self, video: str) -> AnimeDetectado:
        asegurar_ffmpeg()
        self._log("  - Extrayendo fotogramas…")
        with tempfile.TemporaryDirectory() as dir_tmp:
            duracion = duracion_con_ffprobe(video)
            frames   = extraer_fotogramas_centrado(video, dir_tmp, duracion)
            if not frames:
                raise RuntimeError("No se pudieron extraer fotogramas del video.")
            resultado, _ = identificar_anime_con_fotogramas(frames, log_fn=self._log)
            return resultado

    def _verificar_y_resolver_discrepancia(
        self,
        titulo_resuelto:   str,
        picked_base:       Optional[dict],
        ts1_base:          float,
        titulo_anilist:    str,
        confianza_anilist: float,
        origen:            str,
    ) -> Tuple[str, Optional[dict], bool]:
        """
        Compara titulo_resuelto contra titulo_anilist. Si coinciden, confirma.
        Si no, abre el picker interactivo y devuelve la elección del usuario.
        Lanza RuntimeError si el picker es cancelado.
        Devuelve (nuevo_titulo_resuelto, nuevo_picked_base, titulo_confiable).

        origen debe ser uno de dos literales exactos: "cross_verification"
        (Jikan vs trace.moe fresco) o "id_reutilizado" (Jikan vs AniList
        reutilizando el ID del detectado inicial). El método usa un if/else
        binario sobre este valor — no es un campo libre: agregar un tercer
        origen requiere modificar el cuerpo del método, no solo pasar un
        string nuevo.
        """
        acuerdo, motivo_acuerdo = _comparar_titulos_para_verificacion(titulo_resuelto, titulo_anilist)
        if acuerdo is True:
            if motivo_acuerdo == "igualdad_exacta":
                self._log(f"  - ✅ Título confirmado: {titulo_resuelto!r}")
            else:
                self._log(
                    f"  - ✅ Título confirmado: {titulo_resuelto!r}"
                    f" (variante equivalente: {titulo_anilist!r})"
                )
            return titulo_resuelto, picked_base, True

        if origen == "cross_verification":
            etiqueta_anilist = "trace.moe + AniList"
            picker_titulo    = "Verificación de título — Jikan vs trace.moe"
            picker_subtitulo = (
                "Jikan y trace.moe identificaron animes distintos. "
                "Elige cuál es el correcto para este episodio:"
            )
            sufijo_aviso = ""
        else:
            etiqueta_anilist = "AniList (trace.moe ID)"
            picker_titulo    = "Verificación de título — Jikan vs AniList"
            picker_subtitulo = (
                "Jikan y trace.moe (via AniList) identificaron"
                " animes distintos. Elige cuál es el correcto:"
            )
            sufijo_aviso = " (ID reutilizado)"

        motivo = "Discrepancia" if acuerdo is False else "Verificación no concluyente"
        self._log(
            f"  - ⚠️ {motivo}{sufijo_aviso}: "
            f"Jikan={titulo_resuelto!r} / AniList={titulo_anilist!r}"
        )

        if not self.interactivo:
            return titulo_resuelto, picked_base, False

        req = PickRequest(
            kind="discrepancia",
            titulo=picker_titulo,
            subtitulo=picker_subtitulo,
            columnas=[
                ("Fuente",    140),
                ("Título",    500),
                ("Confianza", 200),
            ],
            filas=[
                [
                    "Jikan / MAL",
                    titulo_resuelto,
                    f"ts1 = {ts1_base:.2%}",
                ],
                [
                    etiqueta_anilist,
                    titulo_anilist,
                    f"similitud = {confianza_anilist:.2%}",
                ],
            ],
            payload=[
                {"fuente": "jikan",   "titulo": titulo_resuelto, "item": picked_base},
                {"fuente": "anilist", "titulo": titulo_anilist,  "item": None},
            ],
        )
        idx_elegido = self._pedir_pick(req)
        if idx_elegido is None:
            raise RuntimeError("Selección cancelada.")
        titulo_jikan_orig = titulo_resuelto
        if idx_elegido == 1:
            titulo_resuelto = titulo_anilist
            picked_base     = None
        log_clv(
            logger.debug, "discrepancia_resuelta",
            origen=origen,
            motivo=motivo,
            titulo_jikan=titulo_jikan_orig,
            titulo_anilist=titulo_anilist,
            elegido="anilist" if idx_elegido == 1 else "jikan",
        )
        return titulo_resuelto, picked_base, True

    def _pedir_pick_animethemes(self, resultados: list, consulta: str) -> dict:
        """Construye y despacha el picker de AnimeThemes; devuelve el dict crudo elegido.
        Lanza RuntimeError('Selección cancelada.') si el usuario cancela.
        No extrae slug/name ni decide qué hacer si slug está vacío — eso
        corresponde a cada caller según su propia lógica de error."""
        filas = [
            [
                it.get("name") or "(sin nombre)",
                str(it.get("year") or ""),
                str(it.get("season") or ""),
                it.get("slug") or "",
            ]
            for it in resultados
        ]
        req = PickRequest(
            kind="animethemes",
            titulo="Selecciona el anime correcto (AnimeThemes)",
            subtitulo=(
                f"AnimeThemes devolvió múltiples resultados para: {consulta!r}. "
                "Elige el correcto:"
            ),
            columnas=[("Nombre", 520), ("Año", 70), ("Temporada", 110), ("Slug", 260)],
            filas=filas,
            payload=resultados,
        )
        idx = self._pedir_pick(req)
        if idx is None:
            raise RuntimeError("Selección cancelada.")
        return resultados[int(idx)]

    def _resolver_slug_con_picker(
        self,
        consulta:    str,
        temporada:   int,
        jikan_item:  Optional[dict] = None,
    ) -> Tuple[str, str]:
        """
        Resuelve el slug de AnimeThemes para una consulta dada.
        Si jikan_item está disponible, busca con todos sus títulos alternativos
        antes de abrir el selector interactivo. jikan_item puede venir de Jikan
        (shape con 'mal_id') o de AniList (fallback cuando Jikan agota
        reintentos, shape con 'id'/'idMal' y sin 'mal_id') -- cada shape usa
        su propia función de extracción de títulos porque tienen forma distinta
        (ver anilist_titulos_desde_item vs jikan_titulos_desde_item).
        """
        self._log("• AnimeThemes (resolviendo slug)…")

        # Construir lista de consultas a intentar:
        # 1. La consulta base (título del archivo limpio)
        # 2. Todos los títulos alternativos que la fuente (Jikan o AniList)
        #    conoce para este anime
        consultas_a_intentar: List[str] = [consulta]
        if jikan_item:
            titulos_alt = (
                jikan_titulos_desde_item(jikan_item)
                if "mal_id" in jikan_item
                else anilist_titulos_desde_item(jikan_item)
            )
            for t in titulos_alt:
                if t and t != consulta:
                    consultas_a_intentar.append(t)

        # Intentar cada consulta hasta encontrar un match exacto o resultado único
        for idx_c, q in enumerate(consultas_a_intentar):
            if idx_c > 0:
                self._log(f"  - Reintentando con título alternativo: {q!r}")
            resultados = buscar_anime_en_animethemes(q)
            resultados = filtrar_por_token_obligatorio(q, resultados)
            raw        = list(resultados)
            resultados = _preferir_resultados_por_temporada(resultados, temporada)

            logger.debug(
                f"  - Resultados: crudos={len(raw)} → priorizados={len(resultados)} "
                f"(temporada={temporada})"
            )

            if not resultados:
                continue

            if len(resultados) == 1:
                it   = resultados[0]
                slug = (it.get("slug") or "").strip()
                name = (it.get("name") or q).strip()
                if slug:
                    return slug, name

            exacto = animethemes_coincidencia_exacta_por_titulo(resultados, q)
            if exacto:
                slug = (exacto.get("slug") or "").strip()
                name = (exacto.get("name") or q).strip()
                if slug:
                    return slug, name

        # Ninguna consulta dio match exacto — usar los resultados de la consulta base
        # para el picker (o los de la última que devolvió algo)
        resultados_picker: List[dict] = []
        for q in consultas_a_intentar:
            r = buscar_anime_en_animethemes(q)  # viene de caché, no hace red
            r = filtrar_por_token_obligatorio(q, r)
            r = _preferir_resultados_por_temporada(r, temporada)
            if r:
                resultados_picker = r
                break

        if not resultados_picker:
            return self._resolver_via_jikan_con_picker(consulta)

        resultados = resultados_picker

        if not self.interactivo:
            raise RuntimeError("AnimeThemes ambiguo (modo no interactivo)")

        elegido = self._pedir_pick_animethemes(resultados, consulta)
        slug    = (elegido.get("slug") or "").strip()
        name    = (elegido.get("name") or consulta).strip()
        if not slug:
            raise RuntimeError("AnimeThemes: seleccionado sin slug.")
        return slug, name

    def _resolver_via_jikan_con_picker(self, consulta: str) -> Tuple[str, str]:
        self._log("• Respaldo: Jikan…")
        resultados = jikan_buscar_anime(consulta, limite=10)
        if not resultados:
            raise RuntimeError("Jikan no devolvió resultados.")

        elegido = None
        if len(resultados) == 1:
            elegido = resultados[0]
        else:
            if not self.interactivo:
                raise RuntimeError("Jikan ambiguo (modo no interactivo)")

            filas = [
                [
                    el.get("title") or "(sin título)",
                    el.get("type") or "",
                    str(el.get("year") or ""),
                    "" if el.get("episodes") is None else str(el["episodes"]),
                    "" if el.get("score") is None else f"{float(el['score']):.2f}",
                ]
                for el in resultados
            ]
            req = PickRequest(
                kind="jikan",
                titulo="Selecciona el anime correcto (Jikan/MAL)",
                subtitulo=(
                    f"Se encontraron múltiples resultados para: {consulta!r}. "
                    "Elige el correcto:"
                ),
                columnas=[("Título", 620), ("Tipo", 80), ("Año", 70), ("Eps", 60), ("Puntaje", 80)],
                filas=filas,
                payload=resultados,
            )
            idx = self._pedir_pick(req)
            if idx is None:
                raise RuntimeError("Selección cancelada.")
            elegido = resultados[int(idx)]

        for cand in jikan_titulos_desde_item(elegido):
            at = buscar_anime_en_animethemes(cand)
            if not at:
                continue
            if len(at) == 1:
                slug = (at[0].get("slug") or "").strip()
                name = (at[0].get("name") or cand).strip()
                if slug:
                    return slug, name

            if self.interactivo:
                it   = self._pedir_pick_animethemes(at, cand)
                slug = (it.get("slug") or "").strip()
                name = (it.get("name") or cand).strip()
                if slug:
                    return slug, name

        raise RuntimeError("No encontré la serie en AnimeThemes vía Jikan.")


class ChapterizerWorker(_BaseWorker):
    terminado = pyqtSignal(str)
    fallo     = pyqtSignal(str)

    def __init__(self, ventana, params: ParametrosTrabajo):
        super().__init__(ventana)
        self.ventana = ventana
        self.params  = params

    def run(self):
        try:
            p     = self.params
            video = p.video

            asegurar_ffmpeg()

            self.progress.emit(35)
            dur = duracion_con_ffprobe(video)
            self._log(f"• Video: {_tiempo_sin_ms(dur)}")

            slug         = (p.slug or "").strip()
            titulo_usado = (p.titulo_usado or "").strip() or "Anime"
            episodio     = int(p.episodio or 0)

            ruta_salida = construir_ruta_salida(
                video_path=video,
                carpeta_salida=p.carpeta_salida,
                crear_subcarpeta=p.crear_subcarpeta,
                titulo_anime=titulo_usado,
                episodio=episodio,
            )
            if not p.usar_exacto:
                chapters = chapters_heuristicos(dur)
                guardar_chapters(ruta_salida, chapters)
                self.progress.emit(100)
                self._log("• Chapters:")
                for _t, _nom in sorted(chapters, key=lambda x: x[0]):
                    self._log(f"    [{_tiempo_sin_ms(_t)}]  {_nom}")
                self._log(f"✅ Completado (heurístico): {ruta_salida}")
                self.terminado.emit(ruta_salida)
                return

            if not slug:
                raise RuntimeError(
                    "Slug vacío. La serie no fue resuelta en el hilo principal "
                    "(no se puede hacer coincidencia exacta sin AnimeThemes)."
                )

            self.progress.emit(40)
            anime_json         = obtener_anime_de_animethemes(slug)
            mapa_titulos_temas = construir_mapa_mostrar_temas(anime_json)

            wav_dir, slugs_relevantes = construir_cache_temas(slug, anime_json, self._log, episodio=episodio)

            # Precargar todos los WAVs de temas UNA sola vez — evita re-leer disco
            # en cada ventana del sliding window (potencialmente 20+ lecturas por tema).
            # También se resamplea aquí si es necesario (edge case) y se precalcula
            # frames_t para no repetirlo en cada llamada a _buscar_con_ventana.
            _SR_TARGET = _SR_FEATURES  # 16 000 Hz — mismo hz que el audio del episodio
            wavs_temas: List[TemaAudio] = []
            for ruta in sorted(wav_dir.glob("*.wav")):
                if not ruta.stem.upper().startswith(("OP", "ED")):
                    continue
                if slugs_relevantes:
                    base = re.sub(r'(?i)v\d+$', '', ruta.stem)
                    if base not in slugs_relevantes:
                        continue
                try:
                    y_th, hz_th = leer_pcm16_mono_wav(str(ruta))
                    if hz_th != _SR_TARGET:
                        self._log(f"  - ⚠️ {ruta.stem}: resampleando {hz_th}Hz → {_SR_TARGET}Hz…")
                        razon     = _SR_TARGET / hz_th
                        nuevo_len = int(len(y_th) * razon)
                        x_orig    = np.linspace(0, len(y_th) - 1, len(y_th))
                        x_nuevo   = np.linspace(0, len(y_th) - 1, nuevo_len)
                        y_th      = np.interp(x_nuevo, x_orig, y_th).astype(np.float32)
                        hz_th     = _SR_TARGET
                    feat_th = obtener_features_con_cache(y_th, hz_th)
                    wavs_temas.append(TemaAudio(
                        nombre=ruta.stem,
                        audio=y_th,
                        hz=hz_th,
                        frames=int(len(y_th) / _HOP_LENGTH),
                        features=feat_th,
                    ))
                except Exception as e:
                    self._log(f"  - ⚠️ {ruta.stem}: no se pudo cargar ({e}), omitido")

            if not wavs_temas:
                raise RuntimeError(
                    "No encontré WAVs de OP/ED en caché. "
                    "(¿AnimeThemes no trae audios?)"
                )

            tiene_op = any(t.nombre.upper().startswith("OP") for t in wavs_temas)
            tiene_ed = any(t.nombre.upper().startswith("ED") for t in wavs_temas)

            self._log("• Buscando temas en el episodio…")
            self.progress.emit(55)

            # ── Extraer audio de las zonas UNA sola vez ──────────────────
            op_zona_inicio = 0.0
            op_zona_fin    = min(_SLIDE_OP_MAX, dur * 0.6)
            ed_zona_inicio = max(0.0, dur - _SLIDE_ED_MAX)
            ed_zona_fin    = dur

            with tempfile.TemporaryDirectory() as dir_tmp:
                tmp = Path(dir_tmp)

                op_wav = str(tmp / "zona_op.wav")
                ed_wav = str(tmp / "zona_ed.wav")

                extraer_audio_wav_mono_16k(
                    video, op_wav,
                    ss=op_zona_inicio,
                    duracion=op_zona_fin - op_zona_inicio,
                )
                extraer_audio_wav_mono_16k(
                    video, ed_wav,
                    ss=ed_zona_inicio,
                    duracion=ed_zona_fin - ed_zona_inicio,
                )

                y_op, hz_op = leer_pcm16_mono_wav(op_wav)
                y_ed, hz_ed = leer_pcm16_mono_wav(ed_wav)

            # ── Extraer features globales de cada zona ────────────────────
            logger.debug("Extrayendo features OP y ED")
            feat_op = obtener_features_con_cache(y_op, hz_op)
            feat_ed = obtener_features_con_cache(y_ed, hz_ed)

            mejor_op = self._buscar_con_ventana(
                y_zona=y_op, feat_zona=feat_op, hz=hz_op,
                wavs_temas=wavs_temas, objetivo="OP",
                zona_offset=op_zona_inicio,
                zona_dur=op_zona_fin - op_zona_inicio,
                params=p,
            )
            self.progress.emit(82)

            mejor_ed = self._buscar_con_ventana(
                y_zona=y_ed, feat_zona=feat_ed, hz=hz_ed,
                wavs_temas=wavs_temas, objetivo="ED",
                zona_offset=ed_zona_inicio,
                zona_dur=ed_zona_fin - ed_zona_inicio,
                params=p,
            )

            # Fallback de cobertura: si AnimeThemes no tiene un rol catalogado para
            # este episodio, intentar el rol opuesto en esa zona. Solo se activa cuando
            # el rol falta en wavs_temas (ausencia de datos), no cuando el matching
            # normal falló con datos presentes (problema de audio — caso distinto).
            if mejor_ed is None and tiene_op and not tiene_ed:
                self._log(
                    "  - Sin temas ED para este episodio"
                    " — buscando OP en zona final…"
                )
                mejor_ed = self._buscar_con_ventana(
                    y_zona=y_ed, feat_zona=feat_ed, hz=hz_ed,
                    wavs_temas=wavs_temas, objetivo="OP",
                    zona_offset=ed_zona_inicio,
                    zona_dur=ed_zona_fin - ed_zona_inicio,
                    params=p,
                    es_fallback=True,
                )
                if mejor_ed:
                    self._log(
                        f"    → Coincidencia de respaldo: {mejor_ed.nombre_tema}"
                        f" (score {mejor_ed.puntuacion:.3f})"
                    )

            if mejor_op is None and tiene_ed and not tiene_op:
                self._log(
                    "  - Sin temas OP para este episodio"
                    " — buscando ED en zona inicial…"
                )
                mejor_op = self._buscar_con_ventana(
                    y_zona=y_op, feat_zona=feat_op, hz=hz_op,
                    wavs_temas=wavs_temas, objetivo="ED",
                    zona_offset=op_zona_inicio,
                    zona_dur=op_zona_fin - op_zona_inicio,
                    params=p,
                    es_fallback=True,
                )
                if mejor_op:
                    self._log(
                        f"    → Coincidencia de respaldo: {mejor_op.nombre_tema}"
                        f" (score {mejor_op.puntuacion:.3f})"
                    )

            self.progress.emit(90)

            PRE_OP  = "Introducción"
            EPISODE = "Episodio"
            POST_ED = "Conclusión"

            if not mejor_op and not mejor_ed:
                self._log("⚠️ No pude coincidir con OP/ED. Usando modo heurístico.")
                chapters = chapters_heuristicos(dur)
            else:
                marcas_tiempo: List[float] = []
                if mejor_op:
                    marcas_tiempo.extend([mejor_op.inicio, mejor_op.fin])
                if mejor_ed:
                    marcas_tiempo.extend([mejor_ed.inicio, mejor_ed.fin])
                marcas_tiempo = sorted(marcas_tiempo)

                def cerca_del_inicio(t: float) -> bool: return t < 4.0
                def cerca_del_final(t: float)  -> bool: return t > dur - 4.0

                ajusta_inicio = cerca_del_inicio(marcas_tiempo[0]) if marcas_tiempo else False
                ajusta_final  = cerca_del_final(marcas_tiempo[-1])  if marcas_tiempo else False
                solo_ed       = bool(marcas_tiempo and marcas_tiempo[0] > (dur / 2.0))

                titulo_op = (
                    mapa_titulos_temas.get(mejor_op.nombre_tema)
                    or f"Opening ({mejor_op.nombre_tema})"
                ) if mejor_op else "Opening"

                titulo_ed = (
                    mapa_titulos_temas.get(mejor_ed.nombre_tema)
                    or f"Ending ({mejor_ed.nombre_tema})"
                ) if mejor_ed else "Ending"

                chapters: List[Tuple[float, str]] = [(0.0, PRE_OP)]

                if marcas_tiempo:
                    if ajusta_inicio and not solo_ed:
                        chapters[0] = (0.0, titulo_op)
                        chapters.append((marcas_tiempo[1], EPISODE))
                        if len(marcas_tiempo) == 4:
                            chapters.append((marcas_tiempo[2], titulo_ed))
                            if not ajusta_final:
                                chapters.append((marcas_tiempo[3], POST_ED))
                    elif solo_ed:
                        chapters[0] = (0.0, EPISODE)
                        chapters.append((marcas_tiempo[0], titulo_ed))
                        if not ajusta_final and len(marcas_tiempo) >= 2:
                            chapters.append((marcas_tiempo[1], POST_ED))
                    else:
                        chapters[0] = (0.0, PRE_OP)
                        chapters.append((marcas_tiempo[0], titulo_op))
                        chapters.append((marcas_tiempo[1], EPISODE))
                        if len(marcas_tiempo) == 4:
                            chapters.append((marcas_tiempo[2], titulo_ed))
                            if not ajusta_final:
                                chapters.append((marcas_tiempo[3], POST_ED))

            guardar_chapters(ruta_salida, chapters)
            self.progress.emit(100)
            self._log("• Chapters:")
            for _t, _nom in sorted(chapters, key=lambda x: x[0]):
                self._log(f"    [{_tiempo_sin_ms(_t)}]  {_nom}")
            self._log(f"✅ Listo: {ruta_salida}")
            self.terminado.emit(ruta_salida)

        except Exception as e:
            self.fallo.emit(str(e))

    def _buscar_con_ventana(
        self,
        y_zona:      np.ndarray,         # PCM completo de la zona (samples)
        feat_zona:   np.ndarray,         # features globales (n_feat, T_zona)
        hz:          int,
        wavs_temas:  List[TemaAudio],
        objetivo:    str,
        zona_offset: float,              # segundos absolutos del inicio de la zona
        zona_dur:    float,              # duración de la zona en segundos
        params,
        es_fallback: bool = False,
    ) -> Optional[ResultadoCoincidencia]:
        """
        Ventana deslizante sobre la zona indicada — sin llamadas a ffmpeg.
        El audio PCM y las features ya están en memoria; cada ventana es
        un slice de arrays, lo que elimina ~40 procesos ffmpeg por episodio.

        Conversiones de unidades:
          segundos → samples  : s * hz
          segundos → frames   : int(s * hz / _HOP_LENGTH)
          frames   → segundos : f * _HOP_LENGTH / hz
        """
        mejor: Optional[ResultadoCoincidencia] = None

        zona_label       = "zona inicial" if zona_offset == 0.0 else "zona final"
        sufijo_fallback  = " — respaldo por falta de tema" if es_fallback else ""
        self._log(f"  Buscando {objetivo} en el episodio ({zona_label}{sufijo_fallback}): {_tiempo_sin_ms(zona_offset)} ~ {_tiempo_sin_ms(zona_offset + zona_dur)}")

        win_samples  = int(_SLIDE_WIN_SEC * hz)
        step_samples = int(_SLIDE_STEP_SEC * hz)

        # Mínimo de frames para que DTW tenga contexto real: el mayor entre
        # 8s absolutos (cota dura) y 25% del tema más corto (cota relativa).
        # Evita rechazar ventanas al final de la zona donde el slice es más corto.
        min_frames_absoluto = int(8 * hz / _HOP_LENGTH)
        frames_tema_25pct = [
            int(0.25 * t.frames)
            for t in wavs_temas
            if t.nombre.upper().startswith(objetivo)
        ]
        min_frames_win = max(min_frames_absoluto, min(frames_tema_25pct, default=min_frames_absoluto))

        for paso, s_inicio_raw in enumerate(range(0, int(zona_dur * hz) - win_samples + 1, step_samples)):

            # ── Coherencia frame↔sample ───────────────────────────────────
            f_inicio = int(round(s_inicio_raw / _HOP_LENGTH))
            s_inicio = f_inicio * _HOP_LENGTH

            # ── Clamps ────────────────────────────────────────────────────
            f_fin = min(f_inicio + int(win_samples / _HOP_LENGTH), feat_zona.shape[1])
            s_fin = min(s_inicio + win_samples,                    len(y_zona))

            # ── Validar ventana no vacía ──────────────────────────────────
            # Puede ocurrir si f_inicio >= feat_zona.shape[1] tras el clamp.
            if f_fin <= f_inicio or s_fin <= s_inicio:
                continue

            # ── Filtro de ventana mínima ──────────────────────────────────
            if (f_fin - f_inicio) < min_frames_win:
                continue

            # ── Slices en memoria — sin ffmpeg ────────────────────────────
            y_win    = y_zona[s_inicio:s_fin]
            feat_win = feat_zona[:, f_inicio:f_fin]

            # Offset absoluto de esta ventana en el episodio
            ss_abs = zona_offset + s_inicio / hz

            res = self._coincidencia_con_features(
                y_win=y_win,
                feat_win=feat_win,
                hz=hz,
                wavs_temas=wavs_temas,
                objetivo=objetivo,
                params=params,
            )

            if res is not None:
                abs_res = ResultadoCoincidencia(
                    nombre_tema=res.nombre_tema,
                    inicio=res.inicio + ss_abs,
                    fin=res.fin   + ss_abs,
                    puntuacion=res.puntuacion,
                )
                logger.debug(
                    f"    ✓ win{paso} ({ss_abs:.0f}s): {abs_res.nombre_tema} "
                    f"{formatear_tiempo(abs_res.inicio)}→{formatear_tiempo(abs_res.fin)} "
                    f"score={abs_res.puntuacion:.3f}"
                )
                if mejor is None or abs_res.puntuacion > mejor.puntuacion:
                    mejor = abs_res

        if mejor:
            self._log(
                f"  ✓ [{mejor.nombre_tema}] {_tiempo_sin_ms(mejor.inicio)} ~ {_tiempo_sin_ms(mejor.fin)}"
                f"  (score {mejor.puntuacion:.3f})"
            )
        else:
            self._log(f"  ✗ [{objetivo}] No se encontró coincidencia en esta zona.")

        return mejor

    def _coincidencia_con_features(
        self,
        y_win:      np.ndarray,          # PCM de la ventana (para FFT)
        feat_win:   np.ndarray,          # features de la ventana (para DTW)
        hz:         int,
        wavs_temas: List[TemaAudio],
        objetivo:   str,
        params,
    ) -> Optional[ResultadoCoincidencia]:
        """
        Pipeline FFT→DTW para una sola ventana.
        Separa la lógica de matching del slicing para mantener claridad
        de unidades: y_win son samples, feat_win son frames.
        Los audios ya vienen precargados, resampleados y con frames_t calculado.
        """
        candidatos_fft: List[CandidatoFFT] = []

        for tema in wavs_temas:
            if not tema.nombre.upper().startswith(objetivo):
                continue

            res_fft = _fft_score(
                y_win, tema.audio, hz,
                submuestreo=params.submuestreo,
                porcion_theme=params.porcion_theme,
            )
            if res_fft is None:
                continue

            inicio_fft, fin_fft, fft_s = res_fft
            candidatos_fft.append(CandidatoFFT(
                tema=tema,
                inicio=inicio_fft,
                fin=fin_fft,
                score_fft=fft_s,
            ))

        if not candidatos_fft:
            return None

        candidatos_fft.sort(key=lambda c: c.score_fft, reverse=True)
        candidatos_top = candidatos_fft[:_TOP_K_FFT]

        # ── Early pruning con threshold spread-aware ─────────────────────
        # En lugar de solo el percentil 25, usamos p25 + 0.5 * spread (IQR).
        # Esto detecta si hay un candidato que destaca del resto:
        #   - Si todos los scores son similares y bajos → spread pequeño,
        #     threshold sube y filtra la ventana (todos malos por igual).
        #   - Si un candidato destaca → spread grande, threshold no lo bloquea.
        # Más robusto que percentil fijo ante audios con distintos niveles de señal.
        fft_scores   = [c.score_fft for c in candidatos_top]
        p25          = float(np.percentile(fft_scores, 25))
        p75          = float(np.percentile(fft_scores, 75))
        spread       = p75 - p25
        thr_dinamico = max(_FFT_PRUNING_MIN, p25 + 0.5 * spread)
        mejor_fft_s  = candidatos_top[0].score_fft

        logger.debug(
            f"FFT max={mejor_fft_s:.3f} p25={p25:.3f} p75={p75:.3f} "
            f"spread={spread:.3f} threshold={thr_dinamico:.3f}"
        )

        if mejor_fft_s < thr_dinamico:
            return None

        logger.debug(
            f"  → Top-{len(candidatos_top)} candidatos FFT: "
            + ", ".join(f"{c.tema.nombre}({c.score_fft:.3f})" for c in candidatos_top)
        )

        # DTW usa el slice de features ya calculado — sin re-extraer
        mejor: Optional[ResultadoCoincidencia] = None
        mejor_score = -1.0

        for cand in candidatos_top:
            try:
                dtw_costo   = _dtw_score(feat_win, cand.tema.features)
                dtw_s       = max(0.0, 1.0 - dtw_costo / 50.0)
                score_final = _W_DTW * dtw_s + _W_FFT * cand.score_fft

                logger.debug(
                    f"  - {cand.tema.nombre}: DTW={dtw_costo:.2f} dtw_s={dtw_s:.3f} "
                    f"fft_s={cand.score_fft:.3f} → score={score_final:.3f}"
                )

                if score_final < params.puntuacion_minima:
                    logger.debug(f"    ↳ descartado (score {score_final:.3f} < umbral {params.puntuacion_minima})")
                    continue

                if score_final > mejor_score:
                    mejor_score = score_final
                    mejor       = ResultadoCoincidencia(
                        nombre_tema=cand.tema.nombre,
                        inicio=cand.inicio,
                        fin=cand.fin,
                        puntuacion=score_final,
                    )

            except Exception as e:
                self._log(f"  - ⚠️ {cand.tema.nombre}: error en DTW ({e}), usando FFT como respaldo")
                if cand.score_fft >= params.puntuacion_minima and cand.score_fft > mejor_score:
                    mejor_score = cand.score_fft
                    mejor       = ResultadoCoincidencia(
                        nombre_tema=cand.tema.nombre,
                        inicio=cand.inicio,
                        fin=cand.fin,
                        puntuacion=cand.score_fft,
                    )

        if mejor:
            logger.debug(
                f"  ✓ Mejor: {mejor.nombre_tema} "
                f"{formatear_tiempo(mejor.inicio)}→{formatear_tiempo(mejor.fin)} "
                f"(score={mejor.puntuacion:.3f})"
            )

        return mejor
