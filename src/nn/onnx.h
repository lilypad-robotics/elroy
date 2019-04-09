#ifndef NN_ONNX_H
#define NN_ONNX_H

#include <vector>
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"

#include "nn/network.h"
#include "nn/util/gpu_allocator.h"

class ONNXNetwork : public NeuralNetwork {
    public:
        ONNXNetwork();
        ONNXNetwork(std::string model, uint32_t max_batch_size);
        std::pair<cv::Mat, std::vector<float>> predict(cv::Mat input);
    private:
        void load_model(std::string model);
        void setup_context();
        void wrap_input_layer(std::vector<cv::cuda::GpuMat>* input_channels);
        Mat preprocess(const Mat& img, std::vector<GpuMat>* input_channels);
        void postprocess();

        uint32_t max_batch_size;
        float* input_layer;
        float* output_layer;
        float* output_cpu;

        GPUAllocator* allocator;

        nvinfer1::DimsCHW input_dim;
        nvinfer1::DimsCHW output_dim;
        cv::Size input_cv_size;
};

#endif
