from libcpp cimport bool
from cpython.ref cimport PyObject
cimport numpy as np

from opencv cimport Mat

cdef extern from "cv/ndarray_converter.h":
    cdef cppclass NDArrayConverter:
        Mat to_mat(PyObject* o)
        np.ndarray to_array(Mat mat)

cdef NDArrayConverter* array_converter
cdef Mat to_mat(np.ndarray o)
cdef np.ndarray to_array(Mat m)
