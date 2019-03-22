# distutils: sources = src/nn/ndarray_converter.cpp

array_converter = new NDArrayConverter()

cdef np.ndarray get_array(Mat frame):
    return array_converter.toNDArray(frame)

def poop():
    print("poop")
