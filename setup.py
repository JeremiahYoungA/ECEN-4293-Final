from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension modules for the Hex Engine
extensions = [
    # The Board implementation
    Extension(
        "hex_engine.board",
        sources=["src/hex_engine/board/board.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        include_dirs=[numpy.get_include()]
    ),
    # Optimized Coordinate Utilities
    Extension(
        "hex_engine.utils.coordinates",
        sources=["src/hex_engine/utils/coordinates.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="hex_engine",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,
        }
    ),
    zip_safe=False,
)