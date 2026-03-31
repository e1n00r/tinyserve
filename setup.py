from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "tinyserve._fast_cache",
        ["tinyserve/_fast_cache.pyx"],
    ),
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
    }),
)
