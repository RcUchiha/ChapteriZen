"""Worker de QThread para la generacion de chapters via matching de
audio (ChapterizerWorker). Dividido desde gui/workers.py (que combinaba
ResolverWorker y ChapterizerWorker) porque las dos clases no se llaman
entre si dentro del archivo -- solo se conectan desde __main__.py."""
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from loguru import logger
from PyQt6.QtCore import pyqtSignal

from ..modelos import (
    ParametrosTrabajo,
    TemaAudio,
    CandidatoFFT,
    ResultadoCoincidencia,
)
from ..ffmpeg_utils import (
    asegurar_ffmpeg,
    duracion_con_ffprobe,
    extraer_audio_wav_mono_16k,
    leer_pcm16_mono_wav,
)
from ..animethemes import (
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
from ..chapters_xml import guardar_chapters
from ..naming import construir_ruta_salida
from .resolver_worker import _BaseWorker


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
                    f"AnimeThemes no tiene ningún OP/ED catalogado para '{slug}'"
                    " — no se puede generar el XML sin matching exacto."
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
                raise RuntimeError(
                    "No se encontró coincidencia de audio suficiente para OP ni ED"
                    f" (umbral={p.puntuacion_minima}) — no se generará XML."
                )

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
