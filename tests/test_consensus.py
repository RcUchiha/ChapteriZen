"""
Tests para _consenso_trace() y _consenso_episodio() -- funciones puras,
sin red, que implementan el voto por mayoria del pipeline de trace.moe.
"""
import chapterizen as cz


class TestConsensoTrace:
    def test_consenso_claro_gana_mayor_similitud_dentro_del_ganador(self):
        tops = [
            {"anilist": 100, "similarity": 0.90, "episode": 5},
            {"anilist": 100, "similarity": 0.95, "episode": 5},
            {"anilist": 200, "similarity": 0.99, "episode": 5},
        ]
        mejor, sim = cz._consenso_trace(tops, require_mayoria=True)
        assert mejor["anilist"] == 100
        assert sim == 0.95

    def test_empate_1_1_1_sin_mayoria_devuelve_none(self):
        tops = [
            {"anilist": 100, "similarity": 0.90, "episode": 5},
            {"anilist": 200, "similarity": 0.95, "episode": 5},
            {"anilist": 300, "similarity": 0.99, "episode": 5},
        ]
        mejor, sim = cz._consenso_trace(tops, require_mayoria=True)
        assert mejor is None
        assert sim == -1.0

    def test_empate_sin_require_mayoria_elige_un_candidato(self):
        tops = [
            {"anilist": 100, "similarity": 0.90, "episode": 5},
            {"anilist": 200, "similarity": 0.95, "episode": 5},
            {"anilist": 300, "similarity": 0.99, "episode": 5},
        ]
        mejor, sim = cz._consenso_trace(tops, require_mayoria=False)
        assert mejor is not None
        assert mejor["anilist"] in (100, 200, 300)

    def test_sin_datos_de_anilist_elige_por_similitud_maxima(self):
        tops = [
            {"anilist": None, "similarity": 0.70, "episode": 5},
            {"anilist": None, "similarity": 0.99, "episode": 5},
        ]
        mejor, sim = cz._consenso_trace(tops, require_mayoria=True)
        assert sim == 0.99

    def test_lista_vacia(self):
        mejor, sim = cz._consenso_trace([])
        assert mejor is None
        assert sim == -1.0


class TestConsensoEpisodio:
    def test_mayoria_clara(self):
        candidatos = [{"episode": 5}, {"episode": 5}, {"episode": 5}, {"episode": 6}]
        assert cz._consenso_episodio(candidatos) == 5

    def test_empate_2_vs_2_no_es_mayoria_estricta(self):
        candidatos = [{"episode": 5}, {"episode": 5}, {"episode": 6}, {"episode": 6}]
        assert cz._consenso_episodio(candidatos) is None

    def test_sin_datos_de_episodio_devuelve_none_sin_logs(self):
        logs = []
        candidatos = [{"episode": None}, {"episode": None}, {"anilist": 1}]
        resultado = cz._consenso_episodio(candidatos, log_fn=logs.append)
        assert resultado is None
        assert logs == []

    def test_mayoria_con_un_solo_voto_valido(self):
        candidatos = [{"episode": 7}]
        assert cz._consenso_episodio(candidatos) == 7

    def test_episode_no_numerico_se_ignora(self):
        candidatos = [{"episode": "no-numero"}, {"episode": 5}, {"episode": 5}]
        assert cz._consenso_episodio(candidatos) == 5

    def test_dispersion_logea_advertencia(self):
        logs = []
        candidatos = [{"episode": 5}, {"episode": 6}]
        cz._consenso_episodio(candidatos, log_fn=logs.append)
        assert any("Dispersión" in s for s in logs)
        assert any("Sin coincidencia clara" in s for s in logs)
