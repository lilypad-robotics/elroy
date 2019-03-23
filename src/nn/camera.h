#include <queue>
#include <mutex>
#include <thread>

#include "common.h"

cv::Mat test();

class Camera {
public:
    Camera(int device);
    Camera(int device, unsigned int max_queue_size);
    Camera(int device, unsigned int max_queue_size, unsigned int width, unsigned int height);

    void capture();
    void start();
    void stop();
    cv::Mat read();

private:
    bool recording;
    int device;
    unsigned int max_queue_size;
    cv::VideoCapture* cap;
    cv::Mat current_frame;
    std::mutex queue_mutex;
    std::queue<cv::Mat> frame_queue;
    std::thread recording_thread;
};
