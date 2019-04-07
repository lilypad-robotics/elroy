#include "NvInferPlugin.h"
#include "NvUffParser.h"

class SSDPluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory {
public:
    virtual bool isPlugin(const char* name) override;
    virtual nvinfer1::IPlugin* createPlugin(const char* name, const nvinfer1::Weights *weights, int nbWeights, const nvuffparser::FieldCollection fc) override;
    virtual nvinfer1::IPlugin* createPlugin(const char* name, const void* serialData, size_t serialLength) override; 
};
