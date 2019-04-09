# distutils: sources = src/nn/onnx.cpp  src/nn/network.cpp src/nn/util/gpu_allocator.cpp
import cv2
import numpy as np
from libcpp.vector cimport vector
from libcpp.utility cimport pair
cimport elroy.nn as nn
cimport elroy.util as util

cdef list LABELS = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
cdef np.ndarray COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

cdef class ObjectTracker(object):

    def __init__(self, str model_path, str object_type = 'person'):
        self.model_path = model_path
        self.network = new ONNXNetwork(model_path.encode('ascii'), 1)
        self.object_type = 'person'
        self.trail = []

    cdef tuple detect(self, Mat frame):
        cdef:
            cdef pair[Mat, vector[float]] result
            np.ndarray image
            list boxes
        result = self.network.predict(frame)
        image = util.to_array(result.first)
        boxes = nn.get_bounding_boxes(result.second)
        people_coords = []
        for box in boxes:
            if box.class_id == 14:
                people_coords.append([(box.left + box.right) / 2, (box.top + box.bottom) / 2])
        if len(people_coords) > 0:
            if len(self.trail) > 0:
                self.trail.pop(0)
            people_mean = np.array(people_coords).mean(axis=0)
            self.trail.append(tuple(people_mean))
        return image, boxes

    cdef str get_label(self, int label_id):
        return LABELS[label_id]

    cdef list get_trail(self):
        return self.trail
