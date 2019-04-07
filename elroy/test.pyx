# distutils: sources = src/cv/camera.cpp

import pyfakewebcam
import math
import cv2
from cpython.ref cimport PyObject
import numpy as np
cimport numpy as np
from camera cimport Camera
from opencv cimport Mat
from libcpp.vector cimport vector
from libcpp.pair cimport pair

import tqdm
import elroy.util as util
cimport elroy.util as util
import elroy.nn as nn
cimport elroy.nn as nn

cdef Camera* camera = new Camera(1, 5, 320, 240);
import time
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def sigmoid(x):
      return 1 / (1 + math.exp(-x))

LABELS = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                            dtype="uint8")

def main(model):
    fake = pyfakewebcam.FakeWebcam('/dev/video2', 416, 416)
    cdef:
        pair[Mat, vector[float]] result
        Mat frame, mat
        np.ndarray image
        vector[float] output
    camera.start()
    net = nn.get_model(model, 1)
    # cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    people_coords = []
    trail = []
    while True:
        try:
            frame = camera.read()
            result = net.predict(frame)
            image = util.to_array(result.first)
            people_coords = []
            for box in nn.get_bounding_boxes(result.second):
                if LABELS[box.class_id] == "person":
                    people_coords.append([(box.left + box.right) / 2, (box.top + box.bottom) / 2])
                cv2.rectangle(image, (box.left, box.top), (box.right, box.bottom), (0, 255, 0), 2)
                text = "{}: {:.4f}".format(LABELS[box.class_id], box.class_score) 
                cv2.putText(image, text, (box.left, box.top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            image = image[..., ::-1]
            overlay = image.copy()
            if len(people_coords) > 0:
                people_mean = np.array(people_coords).mean(axis=0).astype(np.int32)
                trail.append(people_mean)
                if len(trail) > 50:
                    trail.pop(0)
            if len(trail) > 0:
                for coord in trail[::-1]:
                    cv2.circle(overlay, tuple(coord), 5, (255, 0, 0), cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
            fake.schedule_frame(image.astype(np.uint8))
        except KeyboardInterrupt:
            break
    camera.stop()
