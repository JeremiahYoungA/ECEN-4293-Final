from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension modules for the Hex Engine
extensions = [
    # The Board implementation
    Extension(
        name="src.hex_engine.board.board_cython",
        sources=["src/hex_engine/board/board_cython.pyx"],
        language="c++",
        extra_compile_args=["-std=c++14"],
        include_dirs=[numpy.get_include()]
    ),
    # Optimized Coordinate Utilities
    Extension(
        name="src.hex_engine.utils.coordinates_cython",
        sources=["src/hex_engine/utils/coordinates_cython.pyx"],
        language="c++",
        extra_compile_args=["-std=c++14"],
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