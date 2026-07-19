# -*- mode: python ; coding: utf-8 -*-
# Build: pyinstaller ChapteriZen.spec
#
# Requiere un venv NUEVO con SOLO requirements.txt + pyinstaller instalados.
# Un Python "sucio" con paquetes de otros proyectos (torch, pandas, sklearn,
# sqlalchemy, matplotlib, etc.) infla el .exe de ~139MB a ~295MB: el hook de
# librosa (pyinstaller-hooks-contrib) fuerza el analisis de TODOS sus
# submodulos aunque la app no los use, y algunos tienen imports condicionales
# opcionales (ej. soporte de tensores via `try: import torch`) que arrastran
# lo que este instalado, sin que sea una dependencia real de la app. Detalle
# completo en docs/TECH_DEBT.md.

# Excluida tras confirmar por grep que audio_matching.py y las funciones de
# librosa que llamamos (mfcc, chroma_stft, sequence.dtw, util.normalize) no
# la usan -- es arrastrada solo porque el hook de librosa fuerza el analisis
# de todo su arbol de submodulos (ver docs/TECH_DEBT.md). Validada con la
# misma prueba de flujo completo (video real -> AnimeThemes -> FFT/DTW ->
# XML) antes de darla por segura.
#
# NO agregar 'scipy.optimize._highspy' aca -- probado y revertido: rompe
# librosa.sequence.dtw() (scipy/optimize/__init__.py importa su arbol
# completo al inicializarse, asi que excluir cualquier submodulo individual
# de scipy.optimize tira abajo el subpaquete entero, no solo esa pieza).
# Ver docs/TECH_DEBT.md.
_EXCLUDES = [
    'sklearn',  # dependencia transitiva real de librosa, ~12MB, no usada por nuestro codigo
]

a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ChapteriZen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
