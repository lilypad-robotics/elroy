import subprocess
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

CPP_SRC_DIR = "src"
EXTRA_LINK_ARGS = [
    "-lnvinfer", "-lnvparsers", "-lnvonnxparser",
    "-lnvonnxparser_runtime", "-lglog", "-lyuv"
]
EXTRA_INCLUDE= ["/usr/local/cuda/include"]

link_args = subprocess.check_output("pkg-config --libs opencv".split()).decode('ascii') \
                                        + " ".join(EXTRA_LINK_ARGS)
compile_args = subprocess.check_output("pkg-config --cflags opencv".split()).decode('ascii') + " ".join(["-I"+a for a in EXTRA_INCLUDE])

extensions = [
        Extension("elroy.tracking", sources=["elroy/tracking.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
                  ],
                  extra_compile_args=["-std=c++17"] + compile_args.split(),
                  extra_link_args=["-std=c++17"] + link_args.split(),
                  language='c++'),
        Extension("elroy.util", sources=["elroy/util.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
                  ],
                  extra_compile_args=["-std=c++17"] + compile_args.split(),
                  extra_link_args=["-std=c++17"] + link_args.split(),
                  language='c++'),
        Extension("elroy.test", sources=["elroy/test.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
                  ],
                  extra_compile_args=["-std=c++17"] + compile_args.split(),
                  extra_link_args=["-std=c++17"] + link_args.split(),
                  language='c++'),
        Extension("elroy.nn", sources=["elroy/nn.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
                  ],
                  extra_compile_args=["-std=c++17"] + compile_args.split(),
                  extra_link_args=["-std=c++17"] + link_args.split(),
                  language='c++'),
]

setup(
    name='elroy',
    ext_modules=cythonize(extensions, nthreads=5,
                          include_path=["elroy/", np.get_include()],
                          compiler_directives={'language_level' : "3"},
                          gdb_debug=False)
)
