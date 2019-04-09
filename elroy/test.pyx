# distutils: sources = src/common.cpp src/cv/camera.cpp src/cv/fake_webcam.cpp

import pyfakewebcam
import cv2
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
from elroy.arduino import Serial
from elroy.tracking cimport ObjectTracker
from elroy.fake_webcam cimport FakeWebcam

cdef extern from "common.h":
    void init_logging()

cdef Camera* camera = new Camera(1, 640, 480);

def main(model):
    init_logging();
    serial = Serial()
    camera.start()
    cdef ObjectTracker tracker = ObjectTracker(model)
    cdef FakeWebcam* fake
    fake = new FakeWebcam(camera, "/dev/video2".encode('ascii'))
    fake.start()
    # fake = pyfakewebcam.FakeWebcam('/dev/video2', 416, 416)
    cdef:
        np.ndarray image
        list boxes
    while True:
        try:
            frame = camera.read()
            image, boxes = tracker.detect(frame)
            # for box in boxes:
                # left, top, right, bottom = (
                    # int(416 * box.left),
                    # int(416 * box.top),
                    # int(416 * box.right),
                    # int(416 * box.bottom),
                # )
                # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                # text = "{}: {:.4f}".format(tracker.get_label(box.class_id), box.class_score)
                # cv2.putText(image, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # overlay = image.copy()
            # for coord in tracker.get_trail()[::-1]:
                # coord = tuple(map(lambda x: int(416 * x), coord))
                # cv2.circle(overlay, coord, 5, (255, 0, 0), cv2.FILLED)
            # image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
            # fake.schedule_frame(image.astype(np.uint8))
            if len(tracker.get_trail()) > 0:
                serial.write(tracker.get_trail()[-1])
        except KeyboardInterrupt:
            break
    fake.stop()
    serial.close()
    camera.stop()
