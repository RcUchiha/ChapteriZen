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

from chapterizen.gui.pickers import (
    DialogoSelectorTabla,
    _ESTILO_TITULO_NORMAL,
    _ESTILO_TITULO_SELECCIONADO,
    _ESTILO_SUBTITULO_NORMAL,
    _ESTILO_SUBTITULO_SELECCIONADO,
)

COLUMNAS = [("Nombre", 200), ("Año", 60)]
FILAS = [
    ["Serie A", "2020"],
    ["Serie B", "2021"],
]
SUBFILAS_AMBAS = ["Alt A", "Alt B"]


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


def test_fila_seleccionada_por_defecto_usa_colores_de_seleccion(qapp):
    """DialogoSelectorTabla selecciona la fila 0 al construirse
    (selectRow(0)) -- si esa fila tiene subtitulo, sus labels deben
    arrancar ya con el estilo de seleccionado (oscuro sobre el naranja
    de QTableWidget::item:selected), no con el normal. La fila 1 (no
    seleccionada) debe quedar en el estilo normal."""
    dlg = DialogoSelectorTabla(None, "titulo", "subtitulo", COLUMNAS, FILAS, SUBFILAS_AMBAS)

    lbl_principal_0, lbl_secundario_0 = dlg._celdas_compuestas[0]
    assert lbl_principal_0.styleSheet() == _ESTILO_TITULO_SELECCIONADO
    assert lbl_secundario_0.styleSheet() == _ESTILO_SUBTITULO_SELECCIONADO

    lbl_principal_1, lbl_secundario_1 = dlg._celdas_compuestas[1]
    assert lbl_principal_1.styleSheet() == _ESTILO_TITULO_NORMAL
    assert lbl_secundario_1.styleSheet() == _ESTILO_SUBTITULO_NORMAL


def test_cambiar_seleccion_alterna_los_colores_entre_filas(qapp):
    """Al seleccionar la fila 1, sus labels pasan a seleccionado y los
    de la fila 0 (antes seleccionada) vuelven a normal -- confirma que
    el toggle reacciona a cambios de seleccion, no solo al estado inicial."""
    dlg = DialogoSelectorTabla(None, "titulo", "subtitulo", COLUMNAS, FILAS, SUBFILAS_AMBAS)

    dlg.table.selectRow(1)

    lbl_principal_0, lbl_secundario_0 = dlg._celdas_compuestas[0]
    assert lbl_principal_0.styleSheet() == _ESTILO_TITULO_NORMAL
    assert lbl_secundario_0.styleSheet() == _ESTILO_SUBTITULO_NORMAL

    lbl_principal_1, lbl_secundario_1 = dlg._celdas_compuestas[1]
    assert lbl_principal_1.styleSheet() == _ESTILO_TITULO_SELECCIONADO
    assert lbl_secundario_1.styleSheet() == _ESTILO_SUBTITULO_SELECCIONADO
