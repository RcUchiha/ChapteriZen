"""Worker de QThread para la fase de resolucion de nombre/slug
(ResolverWorker). Dividido desde gui/workers.py (que combinaba
ResolverWorker y ChapterizerWorker) porque las dos clases no se llaman
entre si dentro del archivo -- solo se conectan desde __main__.py."""
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

from loguru import logger
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from ..modelos import (
    ParametrosTrabajo,
    PickRequest,
    AnimeDetectado,
)
from ..ffmpeg_utils import (
    log_clv,
    asegurar_ffmpeg,
    duracion_con_ffprobe,
    extraer_fotogramas_centrado,
)
from ..config import _es_error_transitorio
from ..trace_moe import _TRACE_UMBRAL_RAPIDO, identificar_anime_con_fotogramas
from ..anilist import (
    anilist_buscar_titulo,
    anilist_titulo_por_id,
    anilist_titulos_desde_item,
    anilist_resolver_temporada_por_sequel,
    anilist_navegar_por_episodio,
    _titulo_principal,
)
from ..animethemes import (
    buscar_anime_en_animethemes,
    animethemes_buscar_por_id_externo,
)
from ..parsing import (
    inferir_consulta_desde_nombre_archivo,
    quitar_sufijo_episodio,
    quitar_marcador_temporada,
    _titulo_es_usable,
    _titulo_tiene_artefacto_pegado,
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
    _comparar_titulos_para_verificacion,
    animethemes_coincidencia_exacta_por_titulo,
    filtrar_por_token_obligatorio,
)


_MSG_SELECCION_CANCELADA = "Selección cancelada."


def _ids_externos_de_picked_base(picked_base: Optional[dict]) -> List[Tuple[str, int]]:
    """IDs externos candidatos para el atajo de AnimeThemes por recurso
    (animethemes_buscar_por_id_externo), a partir del item ya resuelto
    por Jikan o AniList. Si el item es de AniList (shape con 'id'/'idMal',
    sin 'mal_id'), se intentan AMBOS sitios -- AnimeThemes puede tener
    enlazado el recurso de MyAnimeList pero no el de AniList, o
    viceversa; probar los dos maximiza la chance de evitar el camino de
    texto sin gastar mas que un par de requests cacheados."""
    if not picked_base:
        return []
    if "mal_id" in picked_base:
        mal_id = picked_base.get("mal_id")
        return [("MyAnimeList", mal_id)] if mal_id else []
    ids: List[Tuple[str, int]] = []
    if picked_base.get("id"):
        ids.append(("Anilist", picked_base["id"]))
    if picked_base.get("idMal"):
        ids.append(("MyAnimeList", picked_base["idMal"]))
    return ids


def _titulos_conocidos_de_picked_base(picked_base: Optional[dict]) -> List[str]:
    """Titulos que Jikan/AniList ya conocen para picked_base (incluye
    title_japanese/native) -- shape-aware, mismo criterio que ya usa
    _resolver_slug_con_picker para elegir entre jikan_titulos_desde_item
    y anilist_titulos_desde_item."""
    if not picked_base:
        return []
    if "mal_id" in picked_base:
        return jikan_titulos_desde_item(picked_base)
    return anilist_titulos_desde_item(picked_base)


def _token_ok_contra_titulos_conocidos(nombre_at: str, titulos_conocidos: List[str]) -> bool:
    """Validacion cruzada antes de aceptar el atajo por ID sin picker:
    ¿el nombre que devuelve AnimeThemes para ese ID no pierde tokens de
    AL MENOS uno de los titulos que Jikan/AniList ya conocen para el
    mismo item (incluyendo el titulo japones/native)? Comparar contra
    estos titulos en vez de contra el texto crudo del filename evita el
    falso rechazo por diferencia de idioma (filename en ingles vs nombre
    de AnimeThemes en romaji japones -- confirmado con datos reales:
    scripts/simular_atajo_por_id_animethemes.py, ver docs/TECH_DEBT.md).

    Importante -- lo que esta validacion NO cubre: si picked_base ya
    esta mal identificado desde el origen (ej. una identificacion de
    baja confianza via trace.moe que resolvio la serie equivocada), el
    nombre de AnimeThemes y los titulos conocidos heredan el mismo error
    y coinciden igual. Esta validacion protege contra un recurso externo
    mal enlazado DENTRO de AnimeThemes (dato inconsistente entre dos
    fuentes independientes), no contra una identificacion previa
    equivocada en una capa anterior del pipeline -- eso es un problema
    distinto (ver docs/TECH_DEBT.md)."""
    return any(
        _aceptar_canon_sin_perder_tokens(t, nombre_at)
        for t in titulos_conocidos
        if t
    )


def _variante_oficial_que_acepta(base: str, item: dict) -> Optional[Tuple[str, str]]:
    """Busca, entre las variantes OFICIALES de título de item (sin
    sinónimos) que no sean la principal/romaji, alguna que preserve los
    tokens significativos de `base`.

    Por qué existe: AnimeThemes/AniList/MAL tratan el título romaji como
    fuente de verdad para el slug/canon, pero el nombre de archivo puede
    usar una localización distinta (ej. el título occidentalizado en
    inglés: "Chained Soldier" para "Mato Seihei no Slave"). Comparar solo
    contra el título principal producía un falso rechazo en ese caso —
    esta función evita eso probando las demás variantes oficiales antes
    de rendirse. El título que el CALLER adopta si esta función encuentra
    una coincidencia sigue siendo siempre el principal/romaji, nunca la
    variante que hizo pasar el chequeo — eso es responsabilidad del
    caller, no de esta función.

    Devuelve (etiqueta_idioma, texto_variante) de la primera variante que
    pasa el chequeo, o None si ninguna lo hace."""
    if "mal_id" in item:
        variantes = [
            ("inglés",  item.get("title_english")),
            ("japonés", item.get("title_japanese")),
        ]
    else:
        title = item.get("title") or {}
        variantes = [
            ("inglés",    title.get("english")),
            ("nativo",    title.get("native")),
            ("preferido", title.get("userPreferred")),
        ]
    for etiqueta, texto in variantes:
        texto = (texto or "").strip()
        if texto and _aceptar_canon_sin_perder_tokens(base, texto):
            return etiqueta, texto
    return None


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
    cancelado  = pyqtSignal()

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
        """Único punto de entrada real a un picker (todas las llamadas a
        need_pick.emit pasan por aquí, directo o vía _pedir_pick_animethemes).
        El logging de apertura/selección/cancelación vive SOLO aquí,
        no en cada call site, para que los 3 tipos de picker (discrepancia,
        AnimeThemes, Jikan) queden registrados igual sin duplicar código --
        un 4to tipo de picker futuro lo hereda gratis si pasa por acá."""
        self._log(f"🖱️ Picker abierto: {req.titulo} ({len(req.filas)} opciones)")
        self._mx.lock()
        self._pick_index = None
        self._pick_ready = False
        self._mx.unlock()
        self.need_pick.emit(req)
        idx = self._wait_pick()
        if idx is None:
            self._log("🖱️ Picker cancelado por el usuario.")
        else:
            fila      = req.filas[idx] if 0 <= idx < len(req.filas) else None
            seleccion = " | ".join(str(c) for c in fila) if fila else f"opción #{idx}"
            self._log(f"🖱️ Selección: {seleccion}")
        return idx

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

                if (
                    not _titulo_es_usable(consulta_base)
                    or re.fullmatch(r'\d+', consulta_base.strip())
                    or _titulo_tiene_artefacto_pegado(consulta_base)
                ):
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
                        self._log("  - No se encontraron resultados.")

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
                    # picked_base puede venir de Jikan (shape con 'mal_id') o de
                    # AniList (fallback cuando Jikan agota reintentos, shape con
                    # 'id'/'idMal' y sin 'mal_id') -- cada shape usa su propio par
                    # resolver/extractor de título (jikan_resolver_temporada_por_sequel
                    # + título plano vs anilist_resolver_temporada_por_sequel +
                    # _titulo_principal), pero los logs de usuario son idénticos sin
                    # importar la fuente: la decisión de aceptar/rechazar el canon
                    # (_aceptar_canon_sin_perder_tokens) vive aquí, no en jikan.py ni
                    # en anilist.py, precisamente para que el mensaje sea el mismo.
                    if temporada >= 2 and picked_base and not temporada_fue_default:
                        # Camino A: filename declaró temporada explícita → navegar
                        # la cadena de secuelas hasta llegar a ese número de temporada.
                        try:
                            self._log("• Resolviendo temporada de secuela…")
                            if "mal_id" in picked_base:
                                picked_season = jikan_resolver_temporada_por_sequel(picked_base, temporada)
                                canon_season  = (picked_season.get("title") or "").strip()
                            else:
                                picked_season = anilist_resolver_temporada_por_sequel(picked_base, temporada)
                                canon_season  = _titulo_principal(picked_season, "")
                            canon_season = canon_season or titulo_resuelto or consulta_base
                            # titulo_confiable=True fijo (no el titulo_confiable real de
                            # la búsqueda base, que puede ser False si Jikan quedó
                            # ambiguo): la propia navegación de la cadena de secuelas ya
                            # es una señal de identidad independiente de esa ambigüedad.
                            consulta_base_antes = consulta_base
                            consulta_base = self._aplicar_canon_multivariante(
                                consulta_base, picked_season, canon_season, True, log_rechazo=True
                            )
                            # picked_base solo se actualiza a la temporada
                            # resuelta si el canon fue ACEPTADO (consulta_base
                            # cambió) -- _aplicar_canon_multivariante no
                            # devuelve un booleano de aceptar/rechazar, así que
                            # se infiere comparando antes/después. Si el canon
                            # se rechazó (picked_season no comparte tokens con
                            # el archivo -- ver
                            # test_canon_de_secuela_anilist_rechazado_por_recorte_log_identico_a_jikan),
                            # picked_base debe seguir siendo la temporada 1:
                            # usar el ID de picked_season ahí sería reutilizar,
                            # para el atajo de AnimeThemes, la misma entidad
                            # que el pipeline acaba de descartar por no
                            # relacionarse con el archivo. Antes de este fix,
                            # picked_base nunca se reasignaba en absoluto acá
                            # (a diferencia de Camino B, que sí lo hace) -- ver
                            # test_camino_a_actualiza_picked_base_a_la_temporada_resuelta.
                            if consulta_base != consulta_base_antes:
                                picked_base = picked_season
                        except Exception as e:
                            self._log(f"  - ⚠️ Secuela falló: {e}. Usando canon base si está disponible.")
                            consulta_base = self._aplicar_canon_multivariante(
                                consulta_base, picked_base, titulo_resuelto, titulo_confiable
                            )
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
                                if "mal_id" in picked_base:
                                    picked_base, episodio, temporada = jikan_navegar_por_episodio(
                                        picked_base, episodio
                                    )
                                    titulo_resuelto = (picked_base.get("title") or "").strip() or titulo_resuelto
                                else:
                                    picked_base, episodio, temporada = anilist_navegar_por_episodio(
                                        picked_base, episodio
                                    )
                                    titulo_resuelto = _titulo_principal(picked_base, "") or titulo_resuelto
                                self._log(
                                    f"  - Reasignado a temporada {temporada}, episodio {episodio} — {titulo_resuelto!r}"
                                )
                            except Exception as e:
                                self._log(
                                    f"  - ⚠️ Navegación por episodio fallida: {e}"
                                    " — usando episodio original."
                                )
                        consulta_base = self._aplicar_canon_multivariante(
                            consulta_base, picked_base, titulo_resuelto, titulo_confiable
                        )
                    else:
                        consulta_base = self._aplicar_canon_multivariante(
                            consulta_base, picked_base, titulo_resuelto, titulo_confiable
                        )

            # Pasar el item de Jikan para que _resolver_slug_con_picker pueda
            # buscar con títulos alternativos si la consulta base no da resultado.
            # Limitación conocida: cuando anilist_confirmado es True, picked_base es None
            # (Jikan fue omitido), así que jikan_item también es None. Si el título que
            # devuelve AniList no encuentra slug en AnimeThemes, se irá directo al picker
            # interactivo sin haber intentado los títulos alternativos que Jikan hubiera
            # aportado (jikan_titulos_desde_item). En el camino normal (sin AniList) ese
            # respaldo sí existe.
            jikan_item = picked_base if not override else None

            # IDs externos candidatos para el atajo de AnimeThemes por recurso
            # (ver _resolver_slug_con_picker) -- se salta por completo si hay
            # override (el usuario ya escribió una búsqueda manual, no
            # corresponde reemplazarla por un ID que no pidió). En el camino
            # de anilist_confirmado no hay picked_base (Jikan se omitió), pero
            # sí detectado_anilist_id -- se usa ese, y anilist_titulo_por_id
            # como único título conocido para la validación cruzada (no hay
            # un item completo con sinónimos disponible en esta rama).
            ids_externos: List[Tuple[str, int]] = []
            titulos_conocidos: List[str] = []
            if not override:
                if anilist_confirmado and detectado_anilist_id is not None:
                    ids_externos = [("Anilist", detectado_anilist_id)]
                    titulo_por_id = anilist_titulo_por_id(detectado_anilist_id)
                    titulos_conocidos = [titulo_por_id] if titulo_por_id else []
                else:
                    ids_externos      = _ids_externos_de_picked_base(picked_base)
                    titulos_conocidos = _titulos_conocidos_de_picked_base(picked_base)

            slug, titulo_usado = self._resolver_slug_con_picker(
                consulta_base, temporada, jikan_item=jikan_item,
                ids_externos=ids_externos, titulos_conocidos=titulos_conocidos,
            )
            self.progress.emit(30)
            p.slug         = slug
            p.titulo_usado = titulo_usado
            p.episodio     = episodio
            self.resolved.emit(p)

        except Exception as e:
            if str(e) == _MSG_SELECCION_CANCELADA:
                self.cancelado.emit()
            else:
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

    def _aplicar_canon_multivariante(
        self,
        consulta_base:    str,
        item:             Optional[dict],
        titulo_resuelto:  str,
        titulo_confiable: bool,
        log_rechazo:      bool = False,
    ) -> str:
        """Análogo a _aplicar_canon (jikan.py), pero si consulta_base no
        preserva tokens contra titulo_resuelto (el principal/romaji)
        directamente, intenta además las variantes oficiales de título de
        `item` (ver _variante_oficial_que_acepta) antes de rendirse.
        Adopta siempre titulo_resuelto si acepta — nunca la variante que
        hizo pasar el chequeo (AnimeThemes indexa por romaji, no por la
        variante). Si acepta por una variante distinta de la
        principal, deja rastro visible en el log.

        log_rechazo controla si el rechazo final (ninguna variante acepta)
        deja también rastro en el log con "Ignorando canon de temporada
        por recorte". Default False porque Camino B y el caso por defecto
        rechazan en silencio hoy (comportamiento preexistente, no tocado
        por este parámetro) -- solo Camino A (temporada explícita en el
        archivo) pasa True, replicando el warning que antes tenía inline."""
        if not (titulo_confiable and titulo_resuelto):
            return consulta_base
        if _aceptar_canon_sin_perder_tokens(consulta_base, titulo_resuelto):
            return titulo_resuelto
        if item:
            variante = _variante_oficial_que_acepta(consulta_base, item)
            if variante:
                etiqueta, texto_variante = variante
                self._log(
                    f"  - Título del archivo coincide por variante {etiqueta}: "
                    f"{texto_variante!r} → adoptando título romaji/principal: {titulo_resuelto!r}"
                )
                return titulo_resuelto
        if log_rechazo:
            self._log(
                f"  - ⚠️ Ignorando canon de temporada por recorte: "
                f"{consulta_base!r} → {titulo_resuelto!r}"
            )
        return consulta_base

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
            raise RuntimeError(_MSG_SELECCION_CANCELADA)
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
        # Synonym en ingles (si existe) para mostrar como subtitulo tenue
        # debajo del nombre principal -- viene de la misma respuesta de
        # busqueda (include[anime]=animesynonyms en buscar_anime_en_animethemes),
        # sin ninguna consulta adicional. None por candidato sin synonym
        # English -- DialogoSelectorTabla deja esa fila igual que hoy.
        subfilas = [
            next(
                (s.get("text") for s in (it.get("animesynonyms") or []) if s.get("type") == "English"),
                None,
            )
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
            subfilas=subfilas,
        )
        idx = self._pedir_pick(req)
        if idx is None:
            raise RuntimeError(_MSG_SELECCION_CANCELADA)
        return resultados[int(idx)]

    def _resolver_slug_con_picker(
        self,
        consulta:          str,
        temporada:         int,
        jikan_item:        Optional[dict] = None,
        ids_externos:      Optional[List[Tuple[str, int]]] = None,
        titulos_conocidos: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Resuelve el slug de AnimeThemes para una consulta dada.

        Camino principal (si ids_externos no está vacío): consulta directa
        por recurso externo (MyAnimeList/AniList) via
        animethemes_buscar_por_id_externo -- sin texto, sin ambigüedad, sin
        picker. Solo se acepta si hay EXACTAMENTE un resultado y su nombre
        pasa _token_ok_contra_titulos_conocidos contra titulos_conocidos
        (ver esa función para la limitación: protege contra un recurso
        externo mal enlazado en AnimeThemes, no contra picked_base ya mal
        identificado desde una capa anterior). Si el atajo no da un match
        válido para ningún ID candidato, cae exactamente al camino de texto
        de siempre, sin cambios.

        Camino de texto (respaldo, o si no hay ids_externos): si jikan_item
        está disponible, busca con todos sus títulos alternativos antes de
        abrir el selector interactivo. jikan_item puede venir de Jikan
        (shape con 'mal_id') o de AniList (fallback cuando Jikan agota
        reintentos, shape con 'id'/'idMal' y sin 'mal_id') -- cada shape usa
        su propia función de extracción de títulos porque tienen forma distinta
        (ver anilist_titulos_desde_item vs jikan_titulos_desde_item).
        """
        for site, ext_id in (ids_externos or []):
            resultados_id = animethemes_buscar_por_id_externo(site, ext_id)
            if len(resultados_id) != 1:
                continue
            it        = resultados_id[0]
            nombre_at = it.get("name") or ""
            slug      = (it.get("slug") or "").strip()
            if slug and _token_ok_contra_titulos_conocidos(nombre_at, titulos_conocidos or []):
                self._log(f"• AnimeThemes: match directo por {site} ID {ext_id}…")
                return slug, (it.get("name") or consulta).strip()

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
                raise RuntimeError(_MSG_SELECCION_CANCELADA)
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
