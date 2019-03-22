cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat() except +
        void create(int, int, int)
        void* data
        int rows
        int cols
        int channels()
