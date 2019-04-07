# distutils: sources = src/cv/ndarray_converter.cpp

array_converter = new NDArrayConverter()

cdef np.ndarray to_array(Mat frame):
    return array_converter.to_array(frame)

cdef Mat to_mat(np.ndarray o):
    return array_converter.to_mat(<PyObject*>o)
