"""
Tests para _es_error_transitorio() -- clasificacion pura de que errores
HTTP ameritan reintento segun _reintento_http (tenacity).
"""
import httpx
import pytest

import chapterizen as cz


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


class TestEsErrorTransitorio:
    @pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504, 522])
    def test_status_codes_que_deben_reintentar(self, status_code):
        assert cz._es_error_transitorio(_http_status_error(status_code)) is True

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 405, 410, 418, 422, 451])
    def test_status_codes_que_no_deben_reintentar(self, status_code):
        assert cz._es_error_transitorio(_http_status_error(status_code)) is False

    def test_timeout_exception_debe_reintentar(self):
        exc = httpx.ConnectTimeout("timed out")
        assert cz._es_error_transitorio(exc) is True

    def test_read_timeout_debe_reintentar(self):
        exc = httpx.ReadTimeout("timed out")
        assert cz._es_error_transitorio(exc) is True

    def test_excepcion_generica_no_reintenta(self):
        assert cz._es_error_transitorio(ValueError("algo distinto")) is False

    def test_excepcion_de_conexion_no_timeout_no_reintenta(self):
        # httpx.ConnectError no es un HTTPStatusError ni un TimeoutException
        exc = httpx.ConnectError("conexion rechazada")
        assert cz._es_error_transitorio(exc) is False
