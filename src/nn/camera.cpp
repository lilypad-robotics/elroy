#include "camera.h"

void test() {
    std::cout << "Camera" << std::endl;
    ConsumerProducerQueue<cv::Mat> *image_queue = new ConsumerProducerQueue<cv::Mat>(10, false);
}
