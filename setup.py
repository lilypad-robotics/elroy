import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
        Extension("nn", sources=["vc_me/rect.pyx", "src/nn/common.cpp",
                                 "src/nn/camera.cpp", "src/nn/image_queue.cpp"],
                  include_dirs=[
                      "src/nn/",
                      "/usr/include/aarch64-linux-gnu/"
                  ],
                  extra_compile_args=["-std=c++11"],
                  extra_link_args=["-std=c++11"],
                  language='c++'),
]

setup(
    name='vc_me',
    ext_modules=cythonize(extensions)
)
