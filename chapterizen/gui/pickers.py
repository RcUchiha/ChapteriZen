"""Dialogo generico de seleccion en tabla (usado para desambiguar
resultados de Jikan/AnimeThemes/discrepancias). Movido sin cambios
desde chapterizen.py (monolito original, v0.0.7)."""
from typing import Optional, Tuple, List, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
)

_ALTO_FILA_CON_SUBTITULO = 44  # 2 lineas (13px + 11px) necesitan mas alto que la fila default

# Colores del widget compuesto (columna Nombre con subtitulo), normal vs.
# fila seleccionada. "Normal" reusa lo que ya habia (titulo heredado del
# stylesheet de la app -- setStyleSheet("") revierte a eso -- subtitulo
# #888888, mismo valor que QLabel#section). "Seleccionado" replica el
# mismo par oscuro-sobre-naranja que ya usa QPushButton#run (texto #1e1e1e
# sobre fondo #de765d) -- ver QTableWidget::item:selected en __main__.py,
# que pinta ese mismo fondo detras de esta celda. El subtitulo va un poco
# mas claro que el titulo (#3a3a3a) para conservar la misma jerarquia
# "principal vs. tenue" que tiene en el estado normal, pero legible sobre
# el naranja (no #888888, que ahi se leeria mal por poco contraste).
_ESTILO_SUBTITULO_NORMAL      = "font-size: 11px; color: #888888;"
_ESTILO_SUBTITULO_SELECCIONADO = "font-size: 11px; color: #3a3a3a;"
_ESTILO_TITULO_NORMAL         = ""
_ESTILO_TITULO_SELECCIONADO  = "color: #1e1e1e;"


def _celda_con_subtitulo(titulo: str, subtitulo: str) -> Tuple[QWidget, QLabel, QLabel]:
    """Widget compuesto para la columna 0 cuando hay un titulo alternativo
    (ej. synonym en ingles de AnimeThemes) -- 2 QLabel apilados: principal
    con el estilo por defecto, secundario chico y tenue (11px, #888888 --
    mismos valores que QLabel#section en el stylesheet de __main__.py, no
    un gris inventado). Devuelve (contenedor, lbl_principal, lbl_secundario)
    para que el caller pueda alternar sus colores segun el estado de
    seleccion de la fila (ver DialogoSelectorTabla._actualizar_estilo_seleccion).

    WA_TransparentForMouseEvents se pone en el contenedor Y en cada QLabel
    a proposito: en Qt, un click sobre un widget hijo (los QLabel) se
    resuelve contra ESE hijo primero, no contra el contenedor -- si solo
    el contenedor fuera transparente a mouse, un click justo sobre el
    texto de un QLabel igual quedaria absorbido ahi y nunca llegaria a la
    tabla. Marcando los 3 (contenedor + 2 labels), cualquier click dentro
    de la celda cae directo al viewport de la QTableWidget de atras --
    preserva selección de fila y cellDoubleClicked (aceptar con doble
    click) exactamente igual que con un QTableWidgetItem comun."""
    contenedor = QWidget()
    contenedor.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    contenedor.setStyleSheet("background: transparent;")

    lay = QVBoxLayout(contenedor)
    lay.setContentsMargins(4, 2, 4, 2)
    lay.setSpacing(0)

    lbl_principal = QLabel(titulo)
    lbl_principal.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    lbl_secundario = QLabel(subtitulo)
    lbl_secundario.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    lbl_secundario.setStyleSheet(_ESTILO_SUBTITULO_NORMAL)

    lay.addWidget(lbl_principal)
    lay.addWidget(lbl_secundario)
    return contenedor, lbl_principal, lbl_secundario


class DialogoSelectorTabla(QDialog):
    def __init__(
        self,
        ventana_padre,
        titulo:    str,
        subtitulo: str,
        columnas:  List[Tuple[str, int]],
        filas:     List[List[str]],
        subfilas:  Optional[List[Optional[str]]] = None,
    ):
        super().__init__(ventana_padre)
        self.setWindowTitle(titulo)
        self.setModal(True)
        self.resize(980, 420)

        lay = QVBoxLayout()
        lbl = QLabel(subtitulo)
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(len(columnas))
        self.table.setRowCount(len(filas))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        for j, (name, w) in enumerate(columnas):
            self.table.setHorizontalHeaderItem(j, QTableWidgetItem(name))
            self.table.setColumnWidth(j, w)

        # Filas con celda compuesta (columna 0 con subtitulo) -- se
        # trackean para poder alternar sus colores segun la fila
        # seleccionada (ver _actualizar_estilo_seleccion). Las celdas
        # normales (QTableWidgetItem) no necesitan esto: su color de
        # seleccion ya lo resuelve QTableWidget::item:selected (stylesheet
        # de __main__.py) automaticamente, sin ningun codigo de por medio.
        self._celdas_compuestas: Dict[int, Tuple[QLabel, QLabel]] = {}

        for i, fila in enumerate(filas):
            subtitulo_fila = subfilas[i] if subfilas and i < len(subfilas) else None
            for j, val in enumerate(fila):
                if j == 0 and subtitulo_fila:
                    contenedor, lbl_principal, lbl_secundario = _celda_con_subtitulo(val, subtitulo_fila)
                    self.table.setCellWidget(i, j, contenedor)
                    self._celdas_compuestas[i] = (lbl_principal, lbl_secundario)
                else:
                    self.table.setItem(i, j, QTableWidgetItem(val))
            if subtitulo_fila:
                self.table.setRowHeight(i, _ALTO_FILA_CON_SUBTITULO)

        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        lay.addWidget(self.table)

        btnrow     = QHBoxLayout()
        btn_ok     = QPushButton("Usar seleccionado")
        btn_cancel = QPushButton("Cancelar")
        btnrow.addWidget(btn_ok)
        btnrow.addStretch(1)
        btnrow.addWidget(btn_cancel)
        lay.addLayout(btnrow)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        self.table.cellDoubleClicked.connect(lambda *_: self.accept())
        self.table.itemSelectionChanged.connect(self._actualizar_estilo_seleccion)

        self.setLayout(lay)
        if filas:
            self.table.selectRow(0)
        self._actualizar_estilo_seleccion()

    def _actualizar_estilo_seleccion(self):
        """Alterna los colores del widget compuesto (columna 0 con
        subtitulo) entre normal y seleccionado -- las celdas
        QTableWidgetItem normales no necesitan esto, lo resuelve solo
        QTableWidget::item:selected."""
        filas_seleccionadas = {idx.row() for idx in self.table.selectionModel().selectedRows()}
        for fila, (lbl_principal, lbl_secundario) in self._celdas_compuestas.items():
            if fila in filas_seleccionadas:
                lbl_principal.setStyleSheet(_ESTILO_TITULO_SELECCIONADO)
                lbl_secundario.setStyleSheet(_ESTILO_SUBTITULO_SELECCIONADO)
            else:
                lbl_principal.setStyleSheet(_ESTILO_TITULO_NORMAL)
                lbl_secundario.setStyleSheet(_ESTILO_SUBTITULO_NORMAL)

    def indice_seleccionado(self) -> Optional[int]:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        return int(sel[0].row())
