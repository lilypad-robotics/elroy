#ifndef CV_FAKE_WEBCAM_H
#define CV_FAKE_WEBCAM_H

#include <thread>

#include "camera.h"

class FakeWebcam {
    public:
        FakeWebcam();
        FakeWebcam(Camera* camera, std::string output_device);

        void start();
        void stop();

        void pipe();
    private:
        Camera* camera;
        uint32_t width;
        uint32_t height;
        uint32_t framesize;
        bool piping;
        std::string output_device;
        std::thread pipe_thread;

        int video_out;
        uint8_t *buffer;
};
#endif
