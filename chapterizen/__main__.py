"""Punto de entrada de la aplicacion: estilo Qt, ventana principal y
arranque. Movido sin cambios desde chapterizen.py (monolito original,
v0.0.7)."""
import sys
from pathlib import Path
from typing import Optional

import qtawesome as qta
from PyQt6.QtCore import QObject, QEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
    QGroupBox,
    QFrame,
    QProgressBar,
)

from .config import VIDEO_EXTS
from .modelos import ParametrosTrabajo, PickRequest
from .gui.pickers import DialogoSelectorTabla
from .gui.workers import ResolverWorker, ChapterizerWorker


STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
QLabel#title {
    font-size: 20px;
    font-weight: bold;
    color: #de765d;
    padding: 8px 0px;
}
QLabel#section {
    font-size: 11px;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QLineEdit {
    background-color: #313131;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #d4d4d4;
}
QLineEdit:focus:!read-only { border: 1px solid #de765d; }
QLineEdit:read-only { background-color: #282828; color: #d4d4d4; }
QPushButton#browse {
    background-color: #313131;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 7px 9px 8px 9px;
    color: #d4d4d4;
    min-width: 36px;
}
QPushButton#browse:hover    { background-color: #3a3a3a; border-color: #de765d; }
QPushButton#browse:disabled { color: #555555; border-color: #2a2a2a; }
QPushButton#run {
    background-color: #de765d;
    border: none;
    border-radius: 8px;
    padding: 10px 30px;
    color: #1e1e1e;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#run:hover    { background-color: #e88b74; }
QPushButton#run:disabled { background-color: #333333; color: #555555; }
QProgressBar {
    background-color: #313131;
    border: none;
    border-radius: 5px;
    height: 10px;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #de765d, stop:1 #c4923a);
    border-radius: 5px;
}
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    color: #888888;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #888888;
}
QCheckBox { color: #aaaaaa; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #444444;
    border-radius: 3px;
    background-color: #1e1e1e;
}
QCheckBox::indicator:checked { background-color: #de765d; border-color: #de765d; }
QPlainTextEdit {
    background-color: #181818;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 8px;
    color: #a6e3a1;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
QFrame#separator { background-color: #2a2a2a; max-height: 1px; }
QTableWidget {
    background-color: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #d4d4d4;
    gridline-color: #2a2a2a;
}
QHeaderView::section {
    background-color: #1a1a1a;
    color: #888888;
    border: none;
    padding: 4px 8px;
    font-size: 11px;
    text-transform: uppercase;
}
"""


class FieldRow(QWidget):
    def __init__(
        self,
        label:       str,
        btn_text:    str  = "Buscar",
        read_only:   bool = False,
        placeholder: str  = "",
    ):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label.upper())
        lbl.setObjectName("section")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.entry = QLineEdit()
        self.entry.setReadOnly(read_only)
        if placeholder:
            self.entry.setPlaceholderText(placeholder)
        row.addWidget(self.entry)

        self.btn = QPushButton(btn_text)
        self.btn.setObjectName("browse")
        row.addWidget(self.btn)

        layout.addLayout(row)

    def get(self) -> str:
        return self.entry.text().strip()

    def set(self, val: str):
        self.entry.setText(val)


class _HoverIcon(QObject):
    """Event filter que cambia el ícono de un botón al entrar/salir el mouse.
    Sincroniza el color del ícono con el :hover de QSS, que dispara en Enter/Leave
    — distinto de QIcon::Active, que dispara solo al presionar el botón."""
    def __init__(self, btn, icon_normal, icon_hover):
        super().__init__(btn)           # parent=btn mantiene el objeto vivo en Qt
        self._btn        = btn
        self._icon_normal = icon_normal
        self._icon_hover  = icon_hover
        btn.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._btn:
            t = event.type()
            if t == QEvent.Type.Enter:
                self._btn.setIcon(self._icon_hover)
            elif t == QEvent.Type.Leave:
                self._btn.setIcon(self._icon_normal)
        return False


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChapteriZen")
        self.setMinimumWidth(900)
        self._worker:   Optional[ChapterizerWorker] = None
        self._resolver: Optional[ResolverWorker]    = None
        self._construir_interfaz()
        self.setStyleSheet(STYLE)

    def _construir_interfaz(self):
        from PyQt6.QtCore import Qt, QSize

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(14)

        title = QLabel("🎞️ ChapteriZen")
        title.setObjectName("title")
        root.addWidget(title)

        sep = QFrame()
        sep.setObjectName("separator")
        root.addWidget(sep)

        self.row_video = FieldRow(
            "Video", btn_text="Buscar", read_only=True,
            placeholder="Selecciona el archivo de video…",
        )
        self.row_video.btn.clicked.connect(self.elegir_video)
        _icono_video_n = qta.icon('fa5s.file-video', color='#d4d4d4', color_disabled='#555555')
        _icono_video_h = qta.icon('fa5s.file-video', color='#de765d', color_disabled='#555555')
        self.row_video.btn.setIcon(_icono_video_n)
        self.row_video.btn.setIconSize(QSize(16, 16))
        self.row_video.btn.setText("")
        self.row_video.btn.setToolTip("Seleccionar archivo de video")
        self._hover_video = _HoverIcon(self.row_video.btn, _icono_video_n, _icono_video_h)
        root.addWidget(self.row_video)

        self.row_outdir = FieldRow(
            "Carpeta de salida", btn_text="Elegir",
            placeholder="Si no se elige ruta, se guardará junto al video",
        )
        self.row_outdir.btn.clicked.connect(self.elegir_carpeta_salida)
        _icono_dir_n = qta.icon('fa5s.folder-open', color='#d4d4d4', color_disabled='#555555')
        _icono_dir_h = qta.icon('fa5s.folder-open', color='#de765d', color_disabled='#555555')
        self.row_outdir.btn.setIcon(_icono_dir_n)
        self.row_outdir.btn.setIconSize(QSize(16, 16))
        self.row_outdir.btn.setText("")
        self.row_outdir.btn.setToolTip("Elegir carpeta de salida")
        self._hover_dir = _HoverIcon(self.row_outdir.btn, _icono_dir_n, _icono_dir_h)
        root.addWidget(self.row_outdir)

        self.chk_subcarpeta = QCheckBox('Guardar en carpeta "Chapters"')
        root.addWidget(self.chk_subcarpeta)

        self.chk_exacto = QCheckBox("OP/ED exactos (AnimeThemes + coincidencia de audio)")
        self.chk_exacto.setChecked(True)
        root.addWidget(self.chk_exacto)

        self.row_search = FieldRow(
            "Búsqueda en AnimeThemes (opcional)",
            placeholder="Dejar vacío para detectar automáticamente",
        )
        self.row_search.btn.hide()
        root.addWidget(self.row_search)

        box    = QGroupBox("Parámetros de coincidencia")
        boxlay = QHBoxLayout()
        boxlay.setSpacing(16)

        for attr, label, default, width in [
            ("ed_sub",     "Submuestreo",                     "32",   70),
            ("ed_portion", "Porción del theme (0.5~1.0)",     "0.90", 80),
            ("ed_min",     "Umbral de puntuación (0.10~0.40)","0.25", 80),
        ]:
            col = QVBoxLayout()
            col.setSpacing(4)
            col.addWidget(QLabel(label.upper()))
            field = QLineEdit(default)
            field.setFixedWidth(width)
            col.addWidget(field)
            boxlay.addLayout(col)
            setattr(self, attr, field)

        boxlay.addStretch(1)
        box.setLayout(boxlay)
        root.addWidget(box)

        sep2 = QFrame()
        sep2.setObjectName("separator")
        root.addWidget(sep2)

        self.btn_run = QPushButton("Generar XML")
        self.btn_run.setObjectName("run")
        self.btn_run.clicked.connect(self.iniciar)
        root.addWidget(self.btn_run, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        log_lbl = QLabel("LOG DE PROCESO")
        log_lbl.setObjectName("section")
        root.addWidget(log_lbl)

        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(2000)
        self.log_widget.setMinimumHeight(160)
        root.addWidget(self.log_widget, 1)

    def _agregar_log(self, s: str):
        self.log_widget.appendPlainText(s)

    def _todos_controles(self):
        return [
            self.row_video.btn, self.row_outdir.btn,
            self.row_outdir.entry,
            self.chk_subcarpeta, self.chk_exacto,
            self.row_search.entry,
            self.ed_sub, self.ed_portion, self.ed_min,
            self.btn_run,
        ]

    def habilitar_controles(self, enabled: bool):
        for w in self._todos_controles():
            w.setEnabled(enabled)

    def elegir_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Selecciona un video", "",
            "Videos (*.mkv *.mp4 *.avi *.webm *.mov *.m2ts *.ts *.wmv *.vob);;Todos (*.*)",
        )
        if fp:
            self.row_video.set(fp)

    def elegir_carpeta_salida(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Selecciona una carpeta de salida")
        if carpeta:
            self.row_outdir.set(carpeta)

    def iniciar(self):
        video = self.row_video.get()
        if not video or not Path(video).exists() or not video.lower().endswith(VIDEO_EXTS):
            QMessageBox.critical(self, "Error", "Selecciona un video válido.")
            return

        try:
            params = ParametrosTrabajo(
                video=video,
                carpeta_salida=self.row_outdir.get(),
                crear_subcarpeta=self.chk_subcarpeta.isChecked(),
                usar_exacto=self.chk_exacto.isChecked(),
                submuestreo=int(self.ed_sub.text().strip() or "32"),
                porcion_theme=float(self.ed_portion.text().strip() or "0.90"),
                puntuacion_minima=float(self.ed_min.text().strip() or "0.25"),
                search_override=self.row_search.get(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Parámetros inválidos:\n{e}")
            return

        self.log_widget.clear()
        self.progress.setValue(0)
        self.habilitar_controles(False)

        self._resolver = ResolverWorker(self, params, interactivo=True)
        self._resolver.log.connect(self._agregar_log)
        self._resolver.progress.connect(self.progress.setValue)
        self._resolver.need_pick.connect(self._on_need_pick)
        self._resolver.resolved.connect(self._on_resolved_params)
        self._resolver.failed.connect(self._on_fail)
        self._resolver.start()

    def _on_need_pick(self, req: PickRequest):
        dlg = DialogoSelectorTabla(
            self, req.titulo, req.subtitulo, req.columnas, req.filas
        )
        idx = dlg.indice_seleccionado() if dlg.exec() == QDialog.DialogCode.Accepted else None
        if self._resolver:
            self._resolver.entregar_pick(idx)

    def _on_resolved_params(self, params: ParametrosTrabajo):
        ep_str   = f" — Ep. {params.episodio:02d}" if params.episodio else ""
        slug_str = f"  [{params.slug}]"             if params.slug     else ""
        self._agregar_log(f"• {params.titulo_usado or 'Anime'}{ep_str}{slug_str}")
        self._agregar_log("─" * 48)
        self._worker = ChapterizerWorker(self, params)
        self._worker.log.connect(self._agregar_log)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.terminado.connect(self._on_done)
        self._worker.fallo.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, ruta_salida: str):
        self.habilitar_controles(True)
        self._resolver = None
        self._worker   = None
        QMessageBox.information(self, "OK", f"Chapters generados:\n{ruta_salida}")

    def _on_fail(self, msg: str):
        self.habilitar_controles(True)
        self.progress.setValue(0)
        if self._resolver and self._resolver.isRunning():
            self._resolver.cancelar()
            self._resolver.wait(2000)
        self._resolver = None
        self._worker   = None
        self._agregar_log(f"❌ Error: {msg}")
        QMessageBox.critical(self, "Error", msg)


def main():
    import sys
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = VentanaPrincipal()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
