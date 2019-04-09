#include "cv/camera.h"

Camera::Camera(int device) : Camera(device, 640, 480) { }

Camera::Camera(int device, unsigned int width, unsigned int height) : current_frame(height, width, CV_8UC3) { 
    this->device = device;
    this->width = width;
    this->height = height;
    this->cap = std::make_shared<cv::VideoCapture>(this->device);
    this->recording = false;
    this->cap->set(CV_CAP_PROP_FRAME_WIDTH, width);
    this->cap->set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

unsigned int Camera::get_width() {
    return this->width;
}

unsigned int Camera::get_height() {
    return this->height;
}

void Camera::capture() {
    while (this->recording) {
        this->load_frame();
    }
}

void Camera::load_frame() {
    std::unique_lock<std::shared_timed_mutex> lock(this->mutex, std::defer_lock);
    this->cap->read(this->current_frame);
}

void Camera::start() {
    this->recording = true;
    this->recording_thread = std::thread(&Camera::capture, this);
}

void Camera::stop() {
    this->recording = false;
    this->recording_thread.join();
}

cv::Mat Camera::read() {
    std::shared_lock<std::shared_timed_mutex> lock(this->mutex, std::defer_lock);
    return this->current_frame.clone();
}
