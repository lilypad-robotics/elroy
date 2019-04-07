#include <iostream>
#include <sstream>
#include <fstream>

#include "nn/tf.h"

using namespace nvinfer1;
using namespace nvuffparser;

TFNetwork::TFNetwork() : TFNetwork("") { 
}

TFNetwork::TFNetwork(std::string model) : NeuralNetwork() {
    std::cout << "Model " << model << std::endl;
    this->parser = createUffParser();
    this->plugin_factory = new SSDPluginFactory();
    this->parser->setPluginFactory(this->plugin_factory);
    this->parser->registerInput("Input", DimsCHW(3, 300, 300), UffInputOrder::kNCHW);
    this->parser->registerOutput("MarkOutput_0");
    INetworkDefinition* network = this->builder->createNetwork();
    this->parser->parse(model.c_str(), *network, DataType::kFLOAT);
    std::stringstream tensorrt_model_stream;
    tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);
}

int TFNetwork::predict(cv::Mat input) {
    return 1;
}
