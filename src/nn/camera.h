#include <thread>
#include "common.h"
#include "image_queue.h"

cv::Mat test();

class Camera {
public:
    Camera(int device);

    cv::Mat get_frame();

private:
    int device;
    cv::VideoCapture* cap;
    cv::Mat current_frame;
};
