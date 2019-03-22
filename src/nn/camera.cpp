#include "camera.h"

cv::Mat test() {
    std::cout << "Camera" << std::endl;
    Camera* camera = new Camera(1);
    return camera->get_frame();
}

int main() {
    test();
}

Camera::Camera(int device) { 
    this->device = device;
    this->cap = new cv::VideoCapture(this->device);
    this->cap->read(current_frame);
}

cv::Mat Camera::get_frame() {
    this->cap->read(current_frame);
    return this->current_frame;
}
