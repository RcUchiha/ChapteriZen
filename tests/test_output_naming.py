"""
Tests para construir_ruta_salida() -- construccion de la ruta del XML de
chapters de salida.
"""
from pathlib import Path

from chapterizen import naming as cz


class TestConstruirRutaSalida:
    def test_sin_carpeta_salida_usa_carpeta_del_video(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.write_bytes(b"")
        ruta = cz.construir_ruta_salida(
            video_path=str(video),
            carpeta_salida="",
            crear_subcarpeta=False,
            titulo_anime="Frieren",
            episodio=1,
        )
        assert Path(ruta).parent == tmp_path
        assert Path(ruta).name == "Frieren - 01 [Chapters].xml"

    def test_con_carpeta_salida_explicita(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.write_bytes(b"")
        salida_dir = tmp_path / "salida"
        ruta = cz.construir_ruta_salida(
            video_path=str(video),
            carpeta_salida=str(salida_dir),
            crear_subcarpeta=False,
            titulo_anime="Frieren",
            episodio=5,
        )
        assert Path(ruta).parent == salida_dir
        assert salida_dir.exists()
        assert Path(ruta).name == "Frieren - 05 [Chapters].xml"

    def test_crear_subcarpeta_chapters(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.write_bytes(b"")
        ruta = cz.construir_ruta_salida(
            video_path=str(video),
            carpeta_salida="",
            crear_subcarpeta=True,
            titulo_anime="Frieren",
            episodio=1,
        )
        assert Path(ruta).parent == tmp_path / "Chapters"
        assert (tmp_path / "Chapters").exists()

    def test_titulo_con_caracteres_invalidos_se_sanitiza(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.write_bytes(b"")
        ruta = cz.construir_ruta_salida(
            video_path=str(video),
            carpeta_salida="",
            crear_subcarpeta=False,
            titulo_anime="Anime: The Question?",
            episodio=2,
        )
        nombre = Path(ruta).name
        assert ":" not in nombre.replace("[Chapters]", "")
        assert "?" not in nombre.replace("[Chapters]", "")

    def test_episodio_none_se_trata_como_cero(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.write_bytes(b"")
        ruta = cz.construir_ruta_salida(
            video_path=str(video),
            carpeta_salida="",
            crear_subcarpeta=False,
            titulo_anime="Anime",
            episodio=None,
        )
        assert Path(ruta).name == "Anime - 00 [Chapters].xml"

    def test_titulo_vacio_usa_anime_por_defecto(self, tmp_path):
        video = tmp_path / "video.mkv"
        video.write_bytes(b"")
        ruta = cz.construir_ruta_salida(
            video_path=str(video),
            carpeta_salida="",
            crear_subcarpeta=False,
            titulo_anime="",
            episodio=1,
        )
        assert Path(ruta).name == "Anime - 01 [Chapters].xml"
