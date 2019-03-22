from libcpp cimport bool
from cpython.ref cimport PyObject
cimport numpy as np

from opencv cimport Mat

cdef extern from "ndarray_converter.h":
    cdef cppclass NDArrayConverter:
        bool toMat(PyObject* o, Mat &m)
        np.ndarray toNDArray(Mat& mat)

cdef NDArrayConverter* array_converter
cdef np.ndarray get_array(Mat frame)
