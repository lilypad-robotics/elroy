#include "nn/network.h"

#include "NvCaffeParser.h"

    using namespace nvinfer1;
    using namespace nvcaffeparser1;

    class CaffeNetwork : public NeuralNetwork {
public:
    CaffeNetwork();
    CaffeNetwork(std::string model, std::string weights);
    int predict(cv::Mat input);
private:
    ICaffeParser* parser;
};

void convertCaffeToTensorRT(
    const char * deploy_file,                // path to deploy.prototxt file
    const char * weights_file,               // path to caffemodel file
	IBuilder* builder,
	ICaffeParser* parser,
    const std::vector<std::string>& outputs, // network outputs
    size_t max_batch_size,                   // batch size - NB must be at least as large as the batch we want to run with
    bool enable_fp_16,                       // if true and natively supported, use 16-bit floating-point
    std::ostream& output_stream,             // where to serialize the converted model
    Logger& logger                           // custom logger
);
