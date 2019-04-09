#ifndef CV_CAMERA_H
#define CV_CAMERA_H

#include <queue>
#include <shared_mutex>
#include <mutex>
#include <thread>

#include "cv/common.h"

class Camera {
public:
    Camera(const int device);
    Camera(const int device, const unsigned int width, const unsigned int height);

    void capture();
    void start();
    void stop();

    unsigned int get_width();
    unsigned int get_height();

    cv::Mat read();

private:
    unsigned int width;
    unsigned int height;
    bool recording;
    bool started_recording;
    int device;

    std::shared_ptr<cv::VideoCapture> cap;
    cv::Mat current_frame;
    mutable std::shared_timed_mutex mutex;
    std::thread recording_thread;

    void load_frame();
};

#endif
