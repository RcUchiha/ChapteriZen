"""
Cobertura de _subtitulo_alt_para_picker (gui/resolver_worker.py): orden
de prioridad para el subtitulo tenue del picker de AnimeThemes --
1) synonym type="English" si existe, 2) si no, el type="Other" MAS
LARGO entre los disponibles (mas descriptivo que una sigla corta),
3) si no hay ninguno, None (sin subtitulo).

Decision basada en muestreo real de 423 valores unicos de type="Other"
sobre el corpus de 204 archivos: ~75-80% traduccion util, ~15-20%
variantes de romanizacion (no confunden), y 1/423 en otro idioma -- de
ahi que "Other" se acepte como respaldo en vez de descartarlo por
completo.
"""
from chapterizen.gui.resolver_worker import _subtitulo_alt_para_picker


def test_prioriza_english_aunque_haya_other_tambien():
    item = {
        "name": "Ejemplo",
        "animesynonyms": [
            {"type": "Other", "text": "Un synonym Other cualquiera"},
            {"type": "English", "text": "The English Title"},
            {"type": "Native", "text": "何かの日本語"},
        ],
    }
    assert _subtitulo_alt_para_picker(item) == "The English Title"


def test_sin_english_usa_el_other_mas_largo_entre_varios():
    item = {
        "name": "Ejemplo",
        "animesynonyms": [
            {"type": "Other", "text": "GSZS"},
            {"type": "Other", "text": "Goodbye Mr. Despair OAD"},
            {"type": "Other", "text": "Sayonara Zetsubou Sensei OAD"},
            {"type": "Native", "text": "何かの日本語"},
        ],
    }
    # "Sayonara Zetsubou Sensei OAD" (28 caracteres) es mas largo que
    # "Goodbye Mr. Despair OAD" (23) y que "GSZS" (4).
    assert _subtitulo_alt_para_picker(item) == "Sayonara Zetsubou Sensei OAD"


def test_sin_english_ni_other_devuelve_none():
    item = {
        "name": "Ejemplo",
        "animesynonyms": [
            {"type": "Native", "text": "何かの日本語"},
        ],
    }
    assert _subtitulo_alt_para_picker(item) is None


def test_sin_animesynonyms_en_absoluto_devuelve_none():
    item = {"name": "Ejemplo"}
    assert _subtitulo_alt_para_picker(item) is None


def test_other_con_texto_vacio_no_se_elige_y_cae_a_none():
    item = {
        "name": "Ejemplo",
        "animesynonyms": [
            {"type": "Other", "text": ""},
            {"type": "Other", "text": None},
        ],
    }
    assert _subtitulo_alt_para_picker(item) is None
