#include "nn/network.h"

#include "NvUffParser.h"
#include "nn/plugin/factory.h"

using namespace nvinfer1;
using namespace nvuffparser;

class TFNetwork : public NeuralNetwork {
public:
    TFNetwork();
    TFNetwork(std::string model);
    int predict(cv::Mat input);
private:
    IUffParser* parser;
    SSDPluginFactory* plugin_factory;
};
