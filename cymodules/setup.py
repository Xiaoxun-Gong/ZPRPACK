# python setup.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("zprworker", ["zprworker.pyx"],
              include_dirs=[numpy.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),],
              ),
]
setup(
    name="cymodules",
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={'language_level' : "3"}),
)