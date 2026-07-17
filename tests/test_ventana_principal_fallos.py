"""
Cobertura del rediseño de la ventana de fallo en VentanaPrincipal (parte
del mismo cambio que elimino el checkbox "OP/ED exactos"): un fallo real
(Camino C, Camino D, o cualquier RuntimeError del pipeline) debe mostrar
un QMessageBox.warning con el mensaje puntual -- pero la cancelacion de
un picker por el usuario (senal cancelado(), separada de failed() desde
este mismo cambio) es una accion intencional, no un error, y no debe
mostrar ningun popup.

VentanaPrincipal.__init__ solo construye widgets (sin red, sin bloquear),
asi que alcanza con una QApplication headless (offscreen) para instanciarla
sin mostrar ninguna ventana real. Se monkeypatchean QMessageBox.warning y
QMessageBox.critical a nivel de clase para no abrir un dialogo modal real
durante el test.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from chapterizen.__main__ import VentanaPrincipal


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def ventana(qapp, monkeypatch):
    popups = []
    monkeypatch.setattr(QMessageBox, "warning",  lambda *a, **k: popups.append(("warning",  a, k)))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: popups.append(("critical", a, k)))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: popups.append(("information", a, k)))
    w = VentanaPrincipal()
    w.popups = popups
    return w


def test_on_fail_muestra_warning_no_critical(ventana):
    ventana._on_fail("AnimeThemes no tiene ningún OP/ED catalogado para 'test-anime'…")

    assert len(ventana.popups) == 1
    tipo, args, _ = ventana.popups[0]
    assert tipo == "warning"
    assert "AnimeThemes no tiene ningún OP/ED catalogado" in args[2]


def test_on_cancelado_no_muestra_ningun_popup(ventana):
    ventana._on_cancelado()

    assert ventana.popups == []


def test_on_fail_y_on_cancelado_limpian_estado_igual(ventana):
    """Ambos deben rehabilitar los controles y resetear el progreso -- la
    unica diferencia entre los dos es el popup, no el resto de la
    limpieza (_limpiar_estado_tras_terminar es compartido)."""
    ventana.habilitar_controles(False)
    ventana.progress.setValue(55)

    ventana._on_cancelado()
    assert ventana.progress.value() == 0
    assert ventana.btn_run.isEnabled()

    ventana.habilitar_controles(False)
    ventana.progress.setValue(55)

    ventana._on_fail("algún error")
    assert ventana.progress.value() == 0
    assert ventana.btn_run.isEnabled()
