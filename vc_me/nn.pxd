from opencv cimport Mat
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint32_t

cdef class BoundingBox:
    cdef public int left, top, right, bottom
    cdef public float class_score, box_score, score
    cdef public int class_id

cdef extern from "nn/onnx.h":
    cdef cppclass ONNXNetwork:
        ONNXNetwork()
        ONNXNetwork(string model, uint32_t max_batch_size)
        pair[Mat, vector[float]] predict(Mat input)

cdef ONNXNetwork* get_model(str model_path, uint32_t max_batch_size)

cdef list get_bounding_boxes(vector[float] features)
cdef list nms(list bounding_boxes, float threshold)
cdef float iou(box_a, box_b)
