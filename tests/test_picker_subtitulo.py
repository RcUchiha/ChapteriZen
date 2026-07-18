"""
Cobertura del subtitulo opcional (ej. synonym en ingles de AnimeThemes)
en DialogoSelectorTabla, columna 0. Foco principal: no-regresion cuando
subfilas es None (los pickers de "discrepancia" y "jikan" nunca lo
pasan) -- el comportamiento debe ser identico al que ya existia antes
de agregar esta funcionalidad.

DialogoSelectorTabla solo construye widgets (sin red, sin bloquear),
asi que alcanza con una QApplication headless (offscreen) -- mismo
patron que test_ventana_principal_fallos.py.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from chapterizen.gui.pickers import DialogoSelectorTabla

COLUMNAS = [("Nombre", 200), ("Año", 60)]
FILAS = [
    ["Serie A", "2020"],
    ["Serie B", "2021"],
]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_subfilas_none_no_agrega_ningun_widget_de_celda(qapp):
    """No-regresion: sin subfilas (o con el default None), cada celda
    sigue siendo un QTableWidgetItem comun -- ningun cellWidget() en la
    columna 0, exactamente como antes de esta funcionalidad."""
    dlg = DialogoSelectorTabla(None, "titulo", "subtitulo", COLUMNAS, FILAS)

    for i in range(len(FILAS)):
        assert dlg.table.cellWidget(i, 0) is None
        assert dlg.table.item(i, 0) is not None
        assert dlg.table.item(i, 0).text() == FILAS[i][0]
        assert dlg.table.item(i, 1) is not None


def test_subfilas_con_none_explicito_por_fila_tampoco_agrega_widget(qapp):
    """Pasar subfilas=[None, None] (todas las filas sin synonym) debe
    comportarse igual que no pasar subfilas en absoluto."""
    dlg = DialogoSelectorTabla(None, "titulo", "subtitulo", COLUMNAS, FILAS, [None, None])

    for i in range(len(FILAS)):
        assert dlg.table.cellWidget(i, 0) is None
        assert dlg.table.item(i, 0) is not None


def test_subfila_con_texto_agrega_widget_solo_en_esa_fila_columna_0(qapp):
    """Cuando una fila SI tiene subtitulo, solo esa fila (y solo la
    columna 0) usa cellWidget() -- el resto de columnas de esa misma
    fila, y las demas filas sin subtitulo, siguen usando QTableWidgetItem."""
    dlg = DialogoSelectorTabla(
        None, "titulo", "subtitulo", COLUMNAS, FILAS,
        [None, "Alternate English Title"],
    )

    # Fila 0: sin subtitulo -- igual que siempre.
    assert dlg.table.cellWidget(0, 0) is None
    assert dlg.table.item(0, 0) is not None

    # Fila 1: con subtitulo -- widget compuesto en columna 0, pero la
    # columna 1 (Año) sigue siendo un QTableWidgetItem normal.
    assert dlg.table.cellWidget(1, 0) is not None
    assert dlg.table.item(1, 0) is None
    assert dlg.table.item(1, 1) is not None
    assert dlg.table.item(1, 1).text() == "2021"
