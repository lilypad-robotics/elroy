from opencv cimport Mat

cdef extern from "cv/camera.h":
    cdef cppclass Camera:
        Camera(int device)
        Camera(int device, int max_queue_size)
        Camera(int device, unsigned int max_queue_size, unsigned int width, unsigned int height)

        void start()
        void stop()
        Mat read()
