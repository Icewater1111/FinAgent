# 自定义 pkg_resources hook，在导入前先确保 jaraco 可用
# 这个文件会替换 PyInstaller 默认的 pyi_rth_pkgres.py
import sys

# 首先尝试导入 setuptools，然后从 _vendor 中获取 jaraco
try:
    import setuptools
    # 尝试从 setuptools._vendor 导入 jaraco 模块
    if hasattr(setuptools, '_vendor'):
        try:
            # 创建 jaraco 命名空间
            jaraco_module = type(sys)('jaraco')
            sys.modules['jaraco'] = jaraco_module
            
            # 导入并注册 jaraco.text
            try:
                from setuptools._vendor import jaraco
                jaraco_text = jaraco.text
                sys.modules['jaraco.text'] = jaraco_text
            except (ImportError, AttributeError):
                pass
            
            # 导入并注册 jaraco.context
            try:
                jaraco_context = jaraco.context
                sys.modules['jaraco.context'] = jaraco_context
            except (ImportError, AttributeError):
                pass
            
            # 导入并注册 jaraco.functools
            try:
                jaraco_functools = jaraco.functools
                sys.modules['jaraco.functools'] = jaraco_functools
            except (ImportError, AttributeError):
                pass
        except (ImportError, AttributeError):
            pass
except ImportError:
    pass

# 如果上面的方法失败，尝试直接导入 jaraco
if 'jaraco' not in sys.modules:
    try:
        import jaraco
        import jaraco.text
        import jaraco.context
        import jaraco.functools
    except ImportError:
        pass

# 现在可以安全地导入 pkg_resources
try:
    import pkg_resources
except ImportError:
    pass

