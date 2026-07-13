"""
Tests para formatear_tiempo(), _tiempo_sin_ms() y tiempo_mkv() -- funciones
puras de formateo de tiempo usadas en logs y en el XML de chapters.
"""
import chapterizen as cz


class TestFormatearTiempo:
    def test_cero(self):
        assert cz.formatear_tiempo(0) == "00:00:00.000"

    def test_con_milisegundos(self):
        assert cz.formatear_tiempo(65.5) == "00:01:05.500"

    def test_con_horas(self):
        assert cz.formatear_tiempo(3661.234) == "01:01:01.234"


class TestTiempoSinMs:
    def test_cero(self):
        assert cz._tiempo_sin_ms(0) == "00:00"

    def test_sin_horas(self):
        assert cz._tiempo_sin_ms(65) == "01:05"

    def test_con_horas(self):
        assert cz._tiempo_sin_ms(3665) == "01:01:05"


class TestTiempoMkv:
    def test_cero(self):
        assert cz.tiempo_mkv(0) == "00:00:00.000000000"

    def test_con_segundos_y_medio(self):
        assert cz.tiempo_mkv(1.5) == "00:00:01.500000000"

    def test_con_horas_exactas(self):
        assert cz.tiempo_mkv(3661.0) == "01:01:01.000000000"
