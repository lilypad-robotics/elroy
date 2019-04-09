#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <chrono>

#include "libyuv.h"
#include "glog/logging.h"
#include "fake_webcam.h"

FakeWebcam::FakeWebcam() : FakeWebcam(nullptr, "") {
};

FakeWebcam::FakeWebcam(Camera* camera, std::string output_device) {
    this->camera = camera;
    this->output_device = output_device;
    this->piping = false;

    this->width = this->camera->get_width();
    this->height = this->camera->get_height();
}


void set_format(v4l2_format* format, unsigned int width, unsigned int height,
        uint32_t multiplier) {
    format->type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    format->fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format->fmt.pix.width = width;
    format->fmt.pix.height = height;
    format->fmt.pix.field = V4L2_FIELD_NONE;
    format->fmt.pix.bytesperline = width * multiplier;
    format->fmt.pix.sizeimage = height * width * multiplier;
    format->fmt.pix.colorspace = V4L2_COLORSPACE_JPEG;
}


void FakeWebcam::start() {
    std::cout << "Starting fake webcam" << std::endl;
    int retcode = 0;

    this->video_out = open(output_device.c_str(), O_WRONLY | O_SYNC);
    struct v4l2_capability vid_caps;
    struct v4l2_format vid_format;
    uint32_t multiplier = 2;
    this->framesize = this->height * this->width * multiplier;

    retcode = ioctl(this->video_out, VIDIOC_QUERYCAP, &vid_caps);
    if (retcode == -1) {
        LOG(ERROR) << "Could not query video capabilities";
        return;
    }
    memset(&vid_format, 0, sizeof(vid_format));
    set_format(&vid_format, this->width, this->height, multiplier);
    retcode = ioctl(this->video_out, VIDIOC_S_FMT, &vid_format);
    if (retcode == -1) {
        LOG(ERROR) << "Could not set video format";
        return;
    }

    this->piping = true;
    this->pipe_thread = std::thread(&FakeWebcam::pipe, this);
}

void FakeWebcam::stop() {
    std::cout << "Stopping fake webcam" << std::endl;
    this->piping = false;
    this->pipe_thread.join();
    close(this->video_out);
}

void FakeWebcam::pipe() {
    while (this->piping) {
        cv::Mat frame = this->camera->read();
        cv::Mat next_frame;
        cv::cvtColor(frame, next_frame, CV_BGR2RGBA);
        int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
        cv::Mat rgba(next_frame.size(), next_frame.type());
        cv::mixChannels(&next_frame, 1, &rgba, 1, from_to, 4);
        const uint8_t* argb_data = rgba.data;
        int argb_stride = rgba.cols * 4;
        cv::Mat yuyv(rgba.rows, rgba.cols, CV_8UC2);
        uint8_t* yuyv_data = yuyv.data;
        int yuyv_stride = width * 2;
        libyuv::ARGBToYUY2(argb_data, argb_stride, yuyv_data, yuyv_stride, rgba.cols, rgba.rows);
        write(this->video_out, yuyv_data, this->framesize);
    }
}
