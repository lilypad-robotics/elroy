# distutils: sources = src/nn/camera.cpp

from cpython.ref cimport PyObject
cimport numpy as np
from camera cimport Camera
from vc_me.util cimport get_array

cdef Camera* camera = new Camera(1)
cdef np.ndarray process():
    return get_array(camera.get_frame())

def main():
    return process()
