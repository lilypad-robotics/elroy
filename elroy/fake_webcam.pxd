from elroy.camera cimport Camera
from libcpp.string cimport string

cdef extern from "cv/fake_webcam.h":
    
    cdef cppclass FakeWebcam:
        FakeWebcam(Camera* input_device, string output_device)
        void start()
        void stop()
