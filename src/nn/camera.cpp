#include "camera.h"

Camera::Camera(int device) : Camera(device, 50) { }

Camera::Camera(int device, unsigned int max_queue_size) : Camera(device, 50, 640, 480) { }

Camera::Camera(int device, unsigned int max_queue_size, unsigned int width, unsigned int height) { 
    this->device = device;
    this->cap = new cv::VideoCapture(this->device);
    this->max_queue_size = max_queue_size;
    this->recording = false;
    this->cap->set(CV_CAP_PROP_FRAME_WIDTH, width);
    this->cap->set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

void Camera::capture() {
    cv::Mat frame;
    while (this->recording) {
        this->cap->read(frame);

        if (!frame.empty()) {
            std::lock_guard<std::mutex> g(this->queue_mutex);
            if (frame_queue.size() < max_queue_size) {
                frame_queue.push(frame.clone());
            }
            else {
                frame_queue.pop();
                frame_queue.push(frame.clone());
            }
        }
    }
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
	{
		std::lock_guard<std::mutex> g(queue_mutex);
		if (!frame_queue.empty()){
			current_frame = frame_queue.front();
			frame_queue.pop();
		}
	}
    return current_frame;
}
