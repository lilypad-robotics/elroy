# distutils: language = c++

cdef extern from "camera.h":
    void test()

def main():
    test()

main()
