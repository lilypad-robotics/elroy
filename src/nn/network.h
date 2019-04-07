#include <vector>
#include <opencv2/core.hpp>

#include "log.h"
#include "NvInfer.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    virtual std::pair<cv::Mat, std::vector<float>> predict(cv::Mat input) = 0;
protected:
    Logger logger;
    IBuilder* builder;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
};
