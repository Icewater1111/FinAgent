# PyInstaller runtime hook to fix jaraco import issues
# 这个 hook 必须在 pyi_rth_pkgres.py 之前运行（按字母顺序）
import sys
import os

# 在 pkg_resources 导入之前，先导入 jaraco 模块
# 这样可以避免 pkg_resources 在初始化时找不到 jaraco

# 尝试导入所有 jaraco 相关模块
_jaraco_modules = [
    'jaraco',
    'jaraco.text',
    'jaraco.context',
    'jaraco.functools',
    'jaraco.collections',
    'jaraco.classes',
    'jaraco.itertools',
    'jaraco.packaging',
    'jaraco.versioning',
    'jaraco.stream',
    'jaraco.logging',
]

# 首先尝试从 setuptools._vendor 导入（这是 setuptools 内置的版本）
for module_name in _jaraco_modules:
    try:
        # 尝试从 setuptools._vendor 导入
        vendor_name = module_name.replace('jaraco', 'setuptools._vendor.jaraco')
        mod = __import__(vendor_name, fromlist=[''])
        # 如果成功，将其注册到 sys.modules 中，使用原始名称
        sys.modules[module_name] = mod
    except (ImportError, ModuleNotFoundError):
        # 如果从 setuptools._vendor 导入失败，尝试直接导入
        try:
            __import__(module_name)
        except (ImportError, ModuleNotFoundError):
            pass

