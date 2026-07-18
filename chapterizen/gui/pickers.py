"""Dialogo generico de seleccion en tabla (usado para desambiguar
resultados de Jikan/AnimeThemes/discrepancias). Movido sin cambios
desde chapterizen.py (monolito original, v0.0.7)."""
from typing import Optional, Tuple, List

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


def _celda_con_subtitulo(titulo: str, subtitulo: str) -> QWidget:
    """Widget compuesto para la columna 0 cuando hay un titulo alternativo
    (ej. synonym en ingles de AnimeThemes) -- 2 QLabel apilados: principal
    con el estilo por defecto, secundario chico y tenue (11px, #888888 --
    mismos valores que QLabel#section en el stylesheet de __main__.py, no
    un gris inventado).

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
    lbl_secundario.setStyleSheet("font-size: 11px; color: #888888;")

    lay.addWidget(lbl_principal)
    lay.addWidget(lbl_secundario)
    return contenedor


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

        for i, fila in enumerate(filas):
            subtitulo_fila = subfilas[i] if subfilas and i < len(subfilas) else None
            for j, val in enumerate(fila):
                if j == 0 and subtitulo_fila:
                    self.table.setCellWidget(i, j, _celda_con_subtitulo(val, subtitulo_fila))
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

        self.setLayout(lay)
        if filas:
            self.table.selectRow(0)

    def indice_seleccionado(self) -> Optional[int]:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return None
        return int(sel[0].row())
