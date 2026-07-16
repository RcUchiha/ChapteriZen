"""
Tests de caracterizacion para parsear_nombre_archivo() y _titulo_es_usable().

Estos tests fijan el comportamiento ACTUAL (incluyendo casos donde el
resultado es sorprendente, como que _titulo_es_usable("2451") -> True)
para poder comparar contra el comportamiento tras la reestructuracion.
"""
from chapterizen import parsing as cz


class TestParsearNombreArchivo:
    def test_con_temporada_y_episodio_explicitos(self):
        r = cz.parsear_nombre_archivo(
            "HELL.MODE.The.Man.Who.Games.The.Dungeon.Core.And.Becomes.Invincible."
            "S01E05.1080p.WEB-DL.AAC2.0.H.264-Judas.mkv"
        )
        assert r.titulo == "HELL MODE The Man Who Games The Dungeon Core And Becomes Invincible"
        assert r.temporada == 1
        assert r.episodio == 5

    def test_sin_temporada_con_tag_pegado_hevc10bit_titulo_limpio_tras_fix_aniparse(self):
        """Antes del fix de _campos_desde_aniparse (ver
        TestCaracterizacionFixAniparseSchemaYDesempate), aniparse siempre
        devolvia titulo vacio y anitopy ganaba por default con
        "Some Anime Name HEVC10bit" -- "HEVC10bit" quedaba pegado al
        titulo porque anitopy no lo reconoce como tag de video separado.

        Tras el fix esto es una MEJORA real, no un cambio neutro: aniparse
        SI excluye "HEVC10bit" del titulo (lo clasifica como video_term),
        y gana el desempate por score real mas alto (2 vs 0 -- anitopy
        penaliza su propio titulo por dejar un token de ruido tecnico
        pegado), no por defecto ni por empate. El titulo queda mas limpio
        para la busqueda en Jikan/AniList."""
        r = cz.parsear_nombre_archivo("Some.Anime.Name.HEVC10bit.WEBRip - 05.mkv")
        assert r.titulo == "Some Anime Name"
        assert r.temporada is None
        assert r.episodio == 5

    def test_release_limpio_con_brackets(self):
        r = cz.parsear_nombre_archivo(
            "[SubsPlease] Frieren - Beyond Journeys End - 01 [1080p][HEVC10bit][AAC].mkv"
        )
        assert r.titulo == "Frieren - Beyond Journeys End"
        assert r.episodio == 1

    def test_puramente_numerico_devuelve_titulo_numerico(self):
        """Antes del fix de schema de aniparse, este test pasaba "por
        accidente": aniparse siempre devolvia vacio (bug de mapeo), asi
        que nunca llegaba a interpretar "12345" como numero de episodio.

        Tras arreglar el schema, aniparse SI interpreta el nombre de
        archivo completo como episodio=12345, con su propia confianza
        interna en 0.0 (aniparse.parse("12345") -> {'series': [{'episode':
        [{'number': 12345}]}], '_confidence': 0.0}) -- un valor claramente
        no confiable. Ahora pasa por una razon explicita, no por
        accidente: la validacion que descarta el episodio de aniparse
        cuando su titulo quedo vacio (ver docs/TECH_DEBT.md) lo filtra."""
        r = cz.parsear_nombre_archivo("12345.mkv")
        assert r.titulo == "12345"
        assert r.temporada is None
        assert r.episodio is None

    def test_puramente_numerico_corto(self):
        r = cz.parsear_nombre_archivo("01.mkv")
        assert r.titulo == "01"

    def test_vid_prefijo_mas_numero_no_es_puramente_numerico(self):
        """Caso VID_2451 vs 2451: el prefijo alfabetico hace que el parser
        produzca un titulo con letras, distinto del caso puramente numerico."""
        r = cz.parsear_nombre_archivo("VID_2451.mkv")
        assert r.titulo == "VID 2451"

    def test_fuente_reportada_es_aniparse_o_anitopy_o_combinada(self):
        r = cz.parsear_nombre_archivo("[SubsPlease] Frieren - Beyond Journeys End - 01.mkv")
        assert r.fuente in ("aniparse", "anitopy", "aniparse+anitopy", "fallback")


class TestTituloEsUsable:
    def test_titulo_limpio_es_usable(self):
        assert cz._titulo_es_usable("Frieren - Beyond Journeys End") is True

    def test_titulo_corto_valido_86_es_usable(self):
        assert cz._titulo_es_usable("86") is True

    def test_titulo_numerico_100_es_usable(self):
        assert cz._titulo_es_usable("Mob Psycho 100") is True

    def test_titulo_puramente_numerico_2451_ES_usable(self):
        """Comportamiento actual (posiblemente sorprendente): _titulo_es_usable
        por si sola NO rechaza numeros puros -- el rechazo de nombres
        puramente numericos ocurre en ResolverWorker via un chequeo
        adicional (re.fullmatch(r'\\d+', ...)), no dentro de esta funcion."""
        assert cz._titulo_es_usable("2451") is True

    def test_titulo_vid_2451_es_usable(self):
        assert cz._titulo_es_usable("VID 2451") is True

    def test_titulo_invalido_ova(self):
        assert cz._titulo_es_usable("OVA") is False

    def test_titulo_invalido_final(self):
        assert cz._titulo_es_usable("Final") is False

    def test_titulo_valido_con_final_en_nombre_largo(self):
        assert cz._titulo_es_usable("Golden Kamuy Final Season") is True

    def test_titulo_ruido_tecnico_puro_no_usable(self):
        assert cz._titulo_es_usable("1080p AAC x264") is False

    def test_titulo_hash_hex_no_usable(self):
        assert cz._titulo_es_usable("F4FB217B") is False

    def test_titulo_vacio_no_usable(self):
        assert cz._titulo_es_usable("") is False

    def test_titulo_un_caracter_no_usable(self):
        assert cz._titulo_es_usable("K") is False


class TestEsTokenRuidoAmpliacion:
    """Regresion para la ampliacion de _RUIDO_TOKENS/_RE_RUIDO_TITULO/
    _RE_RUIDO_TOKEN_INICIO investigada contra tags reales de Nyaa.si --
    un test por token/patron, con el caso real que lo motivo."""

    def test_avc_es_ruido(self):
        assert cz._es_token_ruido("AVC") is True

    def test_multisubs_pegado_es_ruido(self):
        assert cz._es_token_ruido("MultiSubs") is True

    def test_multi_guion_bajo_subs_es_ruido(self):
        assert cz._es_token_ruido("Multi_Subs") is True

    def test_multiple_subtitle_es_ruido(self):
        """multiple[-_ ]?subtitles? matchea tanto con espacio real (el
        tag [Multiple Subtitle] tal cual aparece en releases) como con
        guion/guion bajo si viniera pegado en un solo token."""
        assert cz._es_token_ruido("Multiple Subtitle") is True
        assert cz._es_token_ruido("Multiple Subtitles") is True
        assert cz._es_token_ruido("Multiple-Subtitle") is True

    def test_pt_br_es_ruido(self):
        assert cz._es_token_ruido("PT-BR") is True

    def test_srtx2_es_ruido(self):
        assert cz._es_token_ruido("SRTx2") is True

    def test_bd_es_ruido(self):
        assert cz._es_token_ruido("BD") is True

    def test_plataformas_streaming_bili_tver_ytb_son_ruido(self):
        assert cz._es_token_ruido("BILI") is True
        assert cz._es_token_ruido("TVER") is True
        assert cz._es_token_ruido("YTB") is True

    def test_vostfr_es_ruido(self):
        assert cz._es_token_ruido("VOSTFR") is True

    def test_web_standalone_excluido_a_proposito_no_es_ruido(self):
        """Excluido deliberadamente: hay titulos reales de AniList donde
        'WEB' es parte legitima del titulo (ej. "Azumanga WEB Daiou",
        confirmado con datos reales) -- agregarlo como token suelto
        corromperia esos titulos."""
        assert cz._es_token_ruido("WEB") is False


class TestCaracterizacionFixAniparseSchemaYDesempate:
    """Tests de caracterizacion para el fix YA IMPLEMENTADO de
    _campos_desde_aniparse/_campos_desde_anitopy (bug de mapeo de schema
    de aniparse, ver docs/TECH_DEBT.md), combinado con tres ajustes de
    logica implementados junto con el: (1) Opcion 2 -- desconfiar de la
    temporada de aniparse si coincide con el episodio que leyo anitopy
    (senal de que aniparse confundio un digito de episodio con uno de
    temporada); (2) desempate de titulo invertido -- en empate de
    _score_titulo, gana anitopy en vez de aniparse; y (3) no confiar en
    el episodio de aniparse si su titulo quedo vacio (ver
    TestParsearNombreArchivo.test_puramente_numerico_devuelve_titulo_numerico
    mas arriba, caso "12345.mkv").

    Tres simulaciones aisladas sucesivas (no en este repo --
    scripts/*.csv gitignoreados, contienen nombres de archivo reales de
    una libreria personal) corrieron cada ajuste por separado y despues
    los tres juntos sobre los 204 archivos reales disponibles, y
    confirmaron que solo 1 (Tojima, ver abajo) cambia de resultado --
    los otros 203, incluyendo Golden Kamuy y los 10 archivos de la serie
    "Android", dan exactamente el mismo resultado antes y despues del
    fix. Ver docs/TECH_DEBT.md para el detalle completo.
    """

    def test_golden_kamuy_no_debe_cambiar_tras_el_fix(self):
        """HOY (bug de mapeo presente) ya da el resultado CORRECTO, pero
        por la razon equivocada: aniparse no aporta nada (temporada/
        episodio/titulo vacios por el bug), asi que anitopy gana por
        default con temporada=None, episodio=7 -- que es lo correcto,
        porque "Final Season" no tiene un numero de temporada propio.

        Tras el fix, aniparse SI aportara una temporada (=7) -- pero
        esa temporada es INCORRECTA (aniparse lee el "07" del episodio
        como si fuera temporada). La validacion de la Opcion 2 debe
        rechazarla porque coincide con el episodio que leyo anitopy (=7),
        y el resultado final debe quedar IDENTICO al de hoy. Este test
        no debe cambiar sus assertions despues del fix -- es la prueba
        de que la Opcion 2 previene la regresion que motivo toda esta
        investigacion (ver docs/TECH_DEBT.md, caso Golden Kamuy)."""
        r = cz.parsear_nombre_archivo(
            "[Erai-raws] Golden Kamuy Final Season - 07 "
            "[1080p CR WEBRip HEVC AAC][MultiSub][08E6CA73].mkv"
        )
        assert r.titulo == "Golden Kamuy Final Season"
        assert r.temporada is None
        assert r.episodio == 7

    def test_android_no_debe_truncar_titulo_tras_el_fix(self):
        """HOY (bug de mapeo presente) ya da el titulo COMPLETO, pero por
        la razon equivocada: aniparse no aporta titulo (vacio por el
        bug), asi que anitopy gana por default con el titulo completo.

        Tras el fix, aniparse SI aportara un titulo -- pero le falta la
        palabra "Android" (bug propio de aniparse, confirmado en las 10
        releases reales de esta serie en el corpus). Ese titulo empata
        en score con el de anitopy; el desempate invertido (gana anitopy
        en empate) debe mantener el titulo completo. Estos 2 casos
        (2 releases distintas, misma serie) representan el mecanismo --
        no hace falta un test por cada uno de los 10 archivos reales
        afectados, todos comparten la misma causa raiz."""
        r1 = cz.parsear_nombre_archivo(
            "[Mamele] Does it Count if You Lose Your Innocence to an "
            "Android - S01E06 (WEB 1080p).mkv"
        )
        assert r1.titulo == "Does it Count if You Lose Your Innocence to an Android"
        assert r1.temporada == 1
        assert r1.episodio == 6

        r2 = cz.parsear_nombre_archivo(
            "[sgt] Does It Count If You Lose Your Innocence to an "
            "Android - S00E01 (WEB 1080p HEVC).mkv"
        )
        assert r2.titulo == "Does It Count If You Lose Your Innocence to an Android"
        assert r2.temporada == 0
        assert r2.episodio == 1

    def test_tojima_kamen_rider_mejora_temporada_y_episodio_tras_el_fix(self):
        """Unico archivo de los 204 reales que efectivamente CAMBIA de
        resultado con el fix (confirmado por simulacion, no solo
        esperado). Antes del fix, anitopy fallaba por completo con esta
        release (titulo tipo oracion larga con puntos como separador, ver
        docs/KNOWN_LIMITATIONS.md) y aniparse estaba roto por el bug de
        mapeo -- ninguno de los dos aportaba temporada/episodio, asi que
        el resultado era temporada=None, episodio=None (incorrecto).

        Tras el fix, aniparse SI extrae bien estos dos campos (schema
        correcto), y la Opcion 2 no interfiere porque su temporada no
        coincide con ningun episodio de anitopy (anitopy no aporto
        ninguno). El titulo NO cambia respecto de antes del fix: sigue
        ganando anitopy por el desempate invertido -- aniparse tambien
        acierta el titulo aqui, asi que no hay necesidad de preferirlo
        por sobre anitopy."""
        r = cz.parsear_nombre_archivo(
            "Tojima.Wants.to.Be.a.Kamen.Rider.S01E19.I.Have.No.Regrets."
            "Dying.as.a.Kamen.Rider.1080p.BILI.WEB-DL.AAC2.0.H.264-VARYG.mkv"
        )
        assert r.titulo == (
            "Tojima Wants to Be a Kamen Rider S01E19.I Have No Regrets "
            "Dying as a Kamen Rider"
        )
        assert r.temporada == 1
        assert r.episodio == 19

    def test_frieren_representativo_de_los_203_archivos_sin_cambios(self):
        """Representante de los 203/204 archivos reales que la
        simulacion confirmo que NO cambian con el fix combinado -- caso
        simple, SxxExx estandar, sin ninguna de las 4 causas raiz
        (confusion temporada/episodio, "Nth Season" textual, truncado de
        titulo, o fallo completo de un parser) involucrada."""
        r = cz.parsear_nombre_archivo(
            "[One Fansub] Sousou no Frieren - S02E05 [WebRip 1080p HEVC-10bits AAC].mkv"
        )
        assert r.titulo == "Sousou no Frieren"
        assert r.temporada == 2
        assert r.episodio == 5


class TestNoRegresionTitulosCortosConNumeros:
    """La ampliacion de _RUIDO_TOKENS (AVC, BD, SRT, PT-BR, etc.) no debe
    afectar la proteccion existente de titulos cortos/numericos
    legitimos que _titulo_es_usable ya protegia antes de este cambio."""

    def test_86_sigue_siendo_usable(self):
        assert cz._titulo_es_usable("86") is True

    def test_titulo_con_numeros_mob_psycho_100_sigue_usable(self):
        assert cz._titulo_es_usable("Mob Psycho 100") is True
