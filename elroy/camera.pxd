from opencv cimport Mat

cdef extern from "cv/camera.h":
    cdef cppclass Camera:
        Camera(int device)
        Camera(int device, unsigned int width, unsigned int height)

        void start()
        void stop()
        Mat read()
