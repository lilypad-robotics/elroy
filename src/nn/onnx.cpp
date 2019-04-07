#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "nn/onnx.h"

using namespace nvinfer1;
using namespace nvonnxparser;
using GpuMat = cv::cuda::GpuMat;
using namespace cv;

ONNXNetwork::ONNXNetwork() : ONNXNetwork("", 1) { 
}

ONNXNetwork::ONNXNetwork(std::string model, uint32_t max_batch_size) : NeuralNetwork() {
    this->allocator = new GPUAllocator(1024 * 1024 * 128);
    this->load_model(model);
    this->setup_context();
}

void ONNXNetwork::load_model(std::string model) {
    std::cout << "Loading model: " << model << std::endl;
    this->runtime = createInferRuntime(this->logger);
    CHECK(this->runtime) << "Failed to create runtime";

    std::stringstream model_stream;
    model_stream.seekg(0, model_stream.beg);
    std::ifstream model_cache(model.c_str());
    CHECK(model_cache) << "Model cache is empty";

    model_stream << model_cache.rdbuf();
    model_cache.close();
    model_stream.seekg(0, std::ios::end);
    const int model_size = model_stream.tellg();
    std::cout << "Loaded model: " << model_size << " bytes" << std::endl;
    model_stream.seekg(0, std::ios::beg);
    void* model_mem = malloc(model_size);
    model_stream.read((char*)model_mem, model_size);
    nvonnxparser::IPluginFactory* plugin_factory = nvonnxparser::createPluginFactory(this->logger);
    this->engine = runtime->deserializeCudaEngine(model_mem, model_size, plugin_factory);
    free(model_mem);
}

void ONNXNetwork::setup_context() {
    this->context = this->engine->createExecutionContext();
    CHECK(this->context) << "Failed to set up context.";
    int input_index = this->engine->getBindingIndex("image");
    this->input_dim = static_cast<DimsCHW&&>(engine->getBindingDimensions(input_index));
    this->input_cv_size = Size(this->input_dim.w(), this->input_dim.h());
    size_t input_size = this->input_dim.c() * this->input_dim.h() * this->input_dim.w() * sizeof(float);
    cudaError_t st = cudaMalloc(&this->input_layer, input_size);
    CHECK_EQ(st, cudaSuccess) << "Could not allocate input layer.";

    LOG(INFO) << input_cv_size << " " << input_size;
    int output_index = engine->getBindingIndex("grid");
    this->output_dim = static_cast<DimsCHW&&>(engine->getBindingDimensions(output_index));
    LOG(INFO) << this->output_dim.c() << " " << this->output_dim.h() << " " << this->output_dim.w();
    size_t output_size = this->output_dim.c() * this->output_dim.h() * this->output_dim.w() * sizeof(float);
    st = cudaMalloc(&this->output_layer, output_size);
    CHECK_EQ(st, cudaSuccess) << "Could not allocate output layer.";
}

std::pair<cv::Mat, std::vector<float>> ONNXNetwork::predict(cv::Mat input) {
    std::vector<GpuMat> input_channels;
    this->wrap_input_layer(&input_channels);
    cv::Mat processed = this->preprocess(input, &input_channels);
    void* buffers[2] = { this->input_layer, this->output_layer };
    this->context->execute(1, buffers);
    size_t output_size = this->output_dim.c() * this->output_dim.h() * this->output_dim.w();
    std::vector<float> output(output_size);
    cudaError_t st = cudaMemcpy(output.data(), this->output_layer, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (st != cudaSuccess)
        throw std::runtime_error("could not copy output layer back to host");
    std::pair<cv::Mat, std::vector<float>> result(processed,  output);
    return result;
}

void ONNXNetwork::wrap_input_layer(std::vector<GpuMat>* input_channels) {
    int width = this->input_dim.w();
    int height = this->input_dim.h();
    float* input_data = this->input_layer;
    for (int i = 0; i < this->input_dim.c(); ++i) {
        GpuMat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

cv::Mat ONNXNetwork::preprocess(const Mat& img, std::vector<GpuMat>* input_channels) {
    int c = this->input_dim.c();
    int w = this->input_dim.w();
    int h = this->input_dim.h();

    // Switching color order in channels
    //cv::Mat rgb;
    //cv::cvtColor(img, rgb, CV_BGR2RGB);

    // Resizing image
    cv::Mat resized;
    float scale = min(float(w)/img.cols,float(h)/img.rows);
    cv::Size scale_size = cv::Size(img.cols * scale, img.rows * scale);
    cv::resize(img, resized, scale_size);
    //cv::resize(img, resized, this->input_cv_size);

    // Cropping image
    cv::Mat cropped(h, w, CV_8UC3, 127);
    cv::Rect rect((w - scale_size.width)/2, (h - scale_size.height)/2, scale_size.width, scale_size.height); 
    resized.copyTo(cropped(rect));
    //cv::Mat cropped;
    //cropped = resized;

    cv::Mat final_img;
    cropped.convertTo(final_img, CV_32FC3);

    GpuMat img_gpu(final_img, this->allocator);

    //HWC to CHW
    cv::cuda::split(img_gpu, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == this->input_layer)
        << "Input channels are not wrapping the input layer of the network.";
    return final_img;
}
