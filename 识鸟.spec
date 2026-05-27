# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['e:/git home/Python-OpenCV-Image-Recognition-System-for-Avian-Monitoring/main_ui.py'],
    pathex=[],
    binaries=[('C:\\Windows\\System32\\msvcp140.dll', '.'), ('C:\\Windows\\System32\\vcruntime140.dll', '.'), ('C:\\Windows\\System32\\vcruntime140_1.dll', '.')],
    datas=[],
    hiddenimports=['numpy.core.multiarray', 'numpy.core.overrides', 'skimage'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [('O', None, 'OPTION'), ('O', None, 'OPTION')],
    exclude_binaries=True,
    name='识鸟',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:\\pictures\\下载\\识鸟.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='识鸟',
)
