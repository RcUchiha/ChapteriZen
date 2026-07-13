"""
Tests para _comparar_titulos_para_verificacion() y
_aceptar_canon_sin_perder_tokens() -- comparacion de titulos entre
Jikan/AniList y aceptacion de canon.
"""
import chapterizen as cz


class TestCompararTitulosParaVerificacion:
    def test_igualdad_exacta(self):
        resultado, motivo = cz._comparar_titulos_para_verificacion("Frieren", "Frieren")
        assert resultado is True
        assert motivo == "igualdad_exacta"

    def test_igualdad_exacta_tras_normalizacion(self):
        resultado, motivo = cz._comparar_titulos_para_verificacion(
            "Sousou no Frieren", "sousou no frieren!"
        )
        assert resultado is True
        assert motivo == "igualdad_exacta"

    def test_relacion_de_prefijo_se_rechaza(self):
        resultado, motivo = cz._comparar_titulos_para_verificacion(
            "Attack on Titan", "Attack on Titan Season 2"
        )
        assert resultado is False
        assert motivo == "prefijo"

    def test_fuzzy_alto_se_acepta(self):
        resultado, motivo = cz._comparar_titulos_para_verificacion(
            "Shingeki no Kyojin", "Shingeki no Kyoujin"
        )
        assert resultado is True
        assert motivo == "similitud_alta"

    def test_titulos_sin_relacion_devuelve_ninguno(self):
        resultado, motivo = cz._comparar_titulos_para_verificacion(
            "Frieren", "One Piece"
        )
        assert resultado is None
        assert motivo == "ninguno"

    def test_titulo_vacio_devuelve_ninguno(self):
        resultado, motivo = cz._comparar_titulos_para_verificacion("", "Frieren")
        assert resultado is None
        assert motivo == "ninguno"


class TestAceptarCanonSinPerderTokens:
    def test_canon_agrega_informacion_se_acepta(self):
        assert cz._aceptar_canon_sin_perder_tokens("Bleach", "Bleach (2004)") is True

    def test_canon_recorta_titulo_se_rechaza(self):
        assert cz._aceptar_canon_sin_perder_tokens(
            "Sword Art Online", "SAO Alicization"
        ) is False

    def test_canon_identico_se_acepta(self):
        assert cz._aceptar_canon_sin_perder_tokens("Frieren", "Frieren") is True

    def test_base_vacia_se_acepta(self):
        assert cz._aceptar_canon_sin_perder_tokens("", "Cualquier Cosa") is True
