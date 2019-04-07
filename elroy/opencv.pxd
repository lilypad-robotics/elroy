from libcpp.vector cimport vector

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        Mat(vector[float] input) except +
        void create(int, int, int)
        void* data
        int rows
        int cols
        int channels()
