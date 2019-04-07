#include <iostream>
#include <sstream>
#include <fstream>

#include "nn/caffe.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

CaffeNetwork::CaffeNetwork() : CaffeNetwork("", "") { 
}

CaffeNetwork::CaffeNetwork(std::string model, std::string weights) : NeuralNetwork() {
    std::cout << "Model " << model << std::endl;
    this->parser = createCaffeParser();
    std::vector<std::string> outputs({"detection_out"});
    std::stringstream tensorrt_model_stream;
    tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);
    convertCaffeToTensorRT(model.c_str(), weights.c_str(), this->builder, this->parser, outputs, 1, true,
            tensorrt_model_stream, this->logger);
}

int CaffeNetwork::predict(cv::Mat input) {
    return 1;
}

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
){
    INetworkDefinition* network = builder->createNetwork();
    if (!network)
    {
        std::cout << "\n[tensorrt-time] Failed to create network definition!\n";
        exit(EXIT_FAILURE);
    }

    // Check whether 16-bit floating-point is natively supported.
    const bool has_fp_16 = builder->platformHasFastFp16();
    // Create a 16-bit model if supported and enabled.
    const bool use_fp_16 = has_fp_16 && enable_fp_16;
    DataType data_type = use_fp_16 ? DataType::kHALF : DataType::kFLOAT;

    // The third parameter is the network definition that the parser will populate.
    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(deploy_file, weights_file, *network, data_type);
    if (!blobNameToTensor)
    {
        std::cout << "\n[tensorrt-time] Failed to parse Caffe model!\n";
        exit(EXIT_FAILURE);
    }

    // As the Caffe model has no notion of outputs, we need to specify
    // explicitly which tensors the engine should generate.
    for (auto& output : outputs)
    {
        const char * output_name = output.c_str();
        ITensor* tensor = blobNameToTensor->find(output_name);
        if (!tensor)
        {
            std::cerr << "\n[tensorrt-time] Failed to retrieve tensor for output \'" << output_name << "\'!\n";
            exit(EXIT_FAILURE);
        }
        network->markOutput(*tensor);
    }

    // Build the engine.
    builder->setMaxBatchSize(max_batch_size);

    // Set up the network for paired-fp16 format if supported and enabled.
    builder->setHalf2Mode(use_fp_16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
    {
        std::cerr << "\n[tensorrt-time] Failed to build CUDA engine!\n";
        exit(EXIT_FAILURE);
    }

    // We no longer need the network, nor do we need the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then shut everything down.
#if NV_TENSORRT_MAJOR > 1
	  nvinfer1::IHostMemory* serMem = engine->serialize();
	  if( !serMem )
	  {
        std::cerr << "\n[tensorrt-time] failed to serialize CUDA engine!\n";
        exit(EXIT_FAILURE);
	  }
    output_stream.write((const char*)serMem->data(), serMem->size());
#else
    engine->serialize(output_stream);
#endif

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}
