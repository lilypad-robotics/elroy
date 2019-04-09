# distutils: sources = src/nn/onnx.cpp src/nn/network.cpp src/nn/util/gpu_allocator.cpp
import numpy as np
cimport numpy as np
cimport elroy.util as util
from elroy.opencv cimport Mat

cdef ONNXNetwork* get_model(str model_path, uint32_t max_batch_size):
    return new ONNXNetwork(model_path.encode('ascii'), max_batch_size)

def sigmoid(x):
      return 1 / (1 + np.exp(-x))
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

cdef class BoundingBox:
    def __init__(self, left, top, right, bottom,
                box_score, class_score, score, class_id):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

        self.box_score = box_score
        self.class_score = class_score
        self.score = score
        self.class_id = class_id

anchors = (
    np.array([1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52])
    .reshape([5, 2]).T
)
boxes = np.tile(np.arange(5)[...,None, None], [1, 13, 13]).ravel()
x, y = np.meshgrid(range(13), range(13))
x = np.tile(x, [5, 1, 1]).reshape(-1)
y = np.tile(y, [5, 1, 1]).reshape(-1)

cdef list get_bounding_boxes(vector[float] features,
                            float detection_threshold=0.2,
                            float nms_threshold=0.2):
    cdef:
        np.ndarray results_tensor
        np.ndarray left, right, width, height
        np.ndarray confidence, class_probs, best_class, best_prob
        np.ndarray total_confidence
        np.ndarray best
        list results
    results_tensor = util.to_array(Mat(features)).reshape([5, 25, 13, 13]).swapaxes(0, 1)

    confidence = sigmoid(results_tensor[4]).reshape(-1)
    class_probs = softmax(results_tensor[5:], 0).reshape([20, -1])
    best_class = class_probs.argmax(axis=0)
    best_prob = class_probs[best_class, np.arange(845)]
    total_confidence = best_prob * confidence
    best = total_confidence > detection_threshold

    best_class = best_class[best]
    total_confidence = total_confidence[best]
    best_prob = best_prob[best]
    best_boxes = boxes[best]

    results_tensor = results_tensor.reshape([25, -1])[:, best]

    left = ((x[best] + sigmoid(results_tensor[0])) / 13).reshape(-1)
    top = ((y[best] + sigmoid(results_tensor[1])) / 13).reshape(-1)
    width = ((np.exp(results_tensor[2]) + anchors[0][best_boxes]) / 13).reshape(-1) / 2
    height = ((np.exp(results_tensor[3]) + anchors[1][best_boxes]) / 13).reshape(-1) / 2
    results = []
    for i in range(results_tensor.shape[1]):
        results.append(BoundingBox(left[i] - width[i],
                                   top[i] - height[i],
                                   left[i] + width[i],
                                   top[i] + height[i],
                                   confidence[i], best_prob[i],
                                   total_confidence[i], best_class[i]))
    results.sort(key=lambda x: x.score, reverse=True)
    results = nms(results, nms_threshold)
    return results

cdef list nms(list bounding_boxes, float threshold):
    if len(bounding_boxes) == 0:
        return []
    nms_predictions = []
    nms_predictions.append(bounding_boxes[0])
    cdef int i = 1
    while i < len(bounding_boxes):
        n_boxes_to_check = len(nms_predictions)
        to_delete = False
        j = 0
        while j < n_boxes_to_check:
            if (bounding_boxes[i].class_id != nms_predictions[j].class_id):
                j = j + 1
                continue
            curr_iou = iou(bounding_boxes[i], nms_predictions[j])
            if(curr_iou > threshold):
                to_delete = True
            j = j + 1
        if to_delete == False:
            nms_predictions.append(bounding_boxes[i])
        i = i + 1
    return nms_predictions

cdef float iou(box_a, box_b):
    cdef:
        float xa, ya
        float xb, yb
        float intersection_area
        float a_area, b_area
        float iou

    xa = max(box_a.left, box_b.left)
    ya = max(box_a.top, box_b.top)
    xb = min(box_a.right, box_b.right)
    yb = min(box_a.bottom, box_b.bottom)

    # Compute the area of intersection
    intersection_area = (xb - xa) * (yb - ya)

    # Compute the area of both rectangles
    a_area = (box_a.right - box_a.left) * (box_a.bottom - box_a.top)
    b_area = (box_b.right - box_b.left) * (box_b.bottom - box_b.top)

    # Compute the IOU
    iou = intersection_area / float(a_area + b_area - intersection_area)
    return iou
