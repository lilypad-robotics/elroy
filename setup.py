import subprocess
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

CPP_SRC_DIR = "src"
EXTRA_LINK_ARGS = ["-lnvinfer", "-lnvparsers", "-lnvonnxparser",
                   "-lnvonnxparser_runtime", "-ljetson-utils",
                  "-lglog"]
EXTRA_INCLUDE= ["/usr/local/cuda/include", "/usr/local/include/jetson-utils"]

link_args = subprocess.check_output("pkg-config --libs opencv".split()).decode('ascii') \
                                        + " ".join(EXTRA_LINK_ARGS)
compile_args = subprocess.check_output("pkg-config --cflags opencv".split()).decode('ascii') + " ".join(["-I"+a for a in EXTRA_INCLUDE])

extensions = [
        Extension("vc_me.util", sources=["vc_me/util.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
                  ],
                  extra_compile_args=["-std=c++11"] + compile_args.split(),
                  extra_link_args=["-std=c++11"] + link_args.split(),
                  language='c++'),
        Extension("vc_me.test", sources=["vc_me/test.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
                  ],
                  extra_compile_args=["-std=c++11"] + compile_args.split(),
                  extra_link_args=["-std=c++11"] + link_args.split(),
                  language='c++'),
        Extension("vc_me.nn", sources=["vc_me/nn.pyx"],
                  include_dirs=[
                      CPP_SRC_DIR
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
