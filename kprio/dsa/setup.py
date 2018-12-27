from distutils.core import setup, Extension

extension_mod = Extension("_dsa", ["_dsa_module.cc", "dsa.cpp"],
        extra_compile_args=["-Wno-deprecated","-O3", "-std=c++11"])

setup(name = "dsa", ext_modules=[extension_mod])
