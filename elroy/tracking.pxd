from elroy.nn cimport ONNXNetwork
from elroy.opencv cimport Mat
cimport numpy as np

cdef list LABELS
cdef np.ndarray COLORS

cdef class ObjectTracker:
    cdef str model_path
    cdef str object_type
    cdef ONNXNetwork* network
    cdef list trail

    cdef tuple detect(self, Mat frame)
    cdef str get_label(self, int label_id)
    cdef list get_trail(self)
