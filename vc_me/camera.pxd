from opencv cimport Mat

cdef extern from "camera.h":
    cdef cppclass Camera:
        Camera(int device)
        Mat get_frame()
