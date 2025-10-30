# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('.env', '.'), ('api_dic.json', '.'), ('AM_APISPECS_faiss_index', 'AM_APISPECS_faiss_index'), ('AM_APISPECS_key_faiss_index', 'AM_APISPECS_key_faiss_index'), ('CN_APISPECS_faiss_index', 'CN_APISPECS_faiss_index'), ('CN_APISPECS_key_faiss_index', 'CN_APISPECS_key_faiss_index'), ('faiss_index', 'faiss_index'), ('HK_APISPECS_faiss_index', 'HK_APISPECS_faiss_index'), ('HK_APISPECS_key_faiss_index', 'HK_APISPECS_key_faiss_index'), ('key_faiss_index', 'key_faiss_index'), ('OT_APISPECS_faiss_index', 'OT_APISPECS_faiss_index'), ('OT_APISPECS_key_faiss_index', 'OT_APISPECS_key_faiss_index'), ('D:\\anaconda3\\envs\\lc\\Lib\\site-packages\\gradio_client\\types.json', 'gradio_client'), ('D:\\anaconda3\\envs\\lc\\Lib\\site-packages\\safehttpx\\version.txt', 'safehttpx'), ('D:\\anaconda3\\envs\\lc\\Lib\\site-packages\\groovy\\version.txt', 'groovy'), ('D:\\anaconda3\\envs\\lc\\Lib\\site-packages\\akshare\\file_fold\\calendar.json', 'akshare\\file_fold')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('gradio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main_with_gradio.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='RAG',
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
