import subprocess
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

link_args = subprocess.check_output("pkg-config --libs opencv".split()).decode('ascii')
compile_args = subprocess.check_output("pkg-config --cflags opencv".split()).decode('ascii')

extensions = [
        Extension("vc_me.util", sources=["vc_me/util.pyx"],
                  include_dirs=[
                      "src/nn/",
                  ],
                  extra_compile_args=["-std=c++11"] + compile_args.split(),
                  extra_link_args=["-std=c++11"] + link_args.split(),
                  language='c++'),
        Extension("vc_me.test", sources=["vc_me/test.pyx"],
                  include_dirs=[
                      "src/nn/",
                  ],
                  extra_compile_args=["-std=c++11"] + compile_args.split(),
                  extra_link_args=["-std=c++11"] + link_args.split(),
                  language='c++'),
]

setup(
    name='vc_me',
    ext_modules=cythonize(extensions, nthreads=4,
                         include_path=["vc_me/", np.get_include()],
                          compiler_directives={'language_level' : "3"},
                         gdb_debug=True)
)
