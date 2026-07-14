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

    def test_sin_temporada_con_tag_pegado_hevc10bit(self):
        r = cz.parsear_nombre_archivo("Some.Anime.Name.HEVC10bit.WEBRip - 05.mkv")
        assert r.titulo == "Some Anime Name HEVC10bit"
        assert r.temporada is None
        assert r.episodio == 5

    def test_release_limpio_con_brackets(self):
        r = cz.parsear_nombre_archivo(
            "[SubsPlease] Frieren - Beyond Journeys End - 01 [1080p][HEVC10bit][AAC].mkv"
        )
        assert r.titulo == "Frieren - Beyond Journeys End"
        assert r.episodio == 1

    def test_puramente_numerico_devuelve_titulo_numerico(self):
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

    def test_viki_excluido_a_proposito_no_es_ruido(self):
        """Excluido deliberadamente al decidir la lista: solo 1 aparicion
        en ~300 publicaciones reales revisadas -- evidencia insuficiente."""
        assert cz._es_token_ruido("VIKI") is False

    def test_vostfr_es_ruido(self):
        assert cz._es_token_ruido("VOSTFR") is True

    def test_web_standalone_excluido_a_proposito_no_es_ruido(self):
        """Excluido deliberadamente: hay titulos reales de AniList donde
        'WEB' es parte legitima del titulo (ej. "Azumanga WEB Daiou",
        confirmado con datos reales) -- agregarlo como token suelto
        corromperia esos titulos."""
        assert cz._es_token_ruido("WEB") is False


class TestNoRegresionTitulosCortosConNumeros:
    """La ampliacion de _RUIDO_TOKENS (AVC, BD, SRT, PT-BR, etc.) no debe
    afectar la proteccion existente de titulos cortos/numericos
    legitimos que _titulo_es_usable ya protegia antes de este cambio."""

    def test_86_sigue_siendo_usable(self):
        assert cz._titulo_es_usable("86") is True

    def test_titulo_con_numeros_mob_psycho_100_sigue_usable(self):
        assert cz._titulo_es_usable("Mob Psycho 100") is True
