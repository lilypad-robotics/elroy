# distutils: sources = src/nn/camera.cpp

from cpython.ref cimport PyObject
cimport numpy as np
from camera cimport Camera
from vc_me.util cimport get_array

cdef Camera* camera = new Camera(1)

def main():
    camera.start()
    for _ in range(100):
        print(get_array(camera.read()))
    camera.stop()
