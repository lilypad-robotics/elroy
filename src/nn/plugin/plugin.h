#include <cassert>
#include <memory>
#include <cublas_v2.h>
#include <cudnn.h>
#include <unordered_map>
#include <vector>

#include "NvInferPlugin.h"
#include "NvUffParser.h"

#include "nn/plugin/common.h"

using namespace nvuffparser;
using namespace nvinfer1;
using namespace plugin;

extern DetectionOutputParameters detectionOutputParam;

class FlattenConcat : public IPlugin
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch);

    FlattenConcat(const void* data, size_t length);

    ~FlattenConcat();
    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    int initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override;
    size_t getSerializationSize() override;
    void serialize(void* buffer) override;
    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

private:
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize;
    bool mIgnoreBatch{false};
    int mConcatAxisID, mOutputConcatAxis, mNumInputs;
    int* mInputConcatAxis;
    nvinfer1::Dims mCHW;
    cublasHandle_t mCublas;
};

// Integration for serialization.
class PluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactory
{
public:
    std::unordered_map<std::string, int> concatIDs = {
        std::make_pair("_concat_box_loc", 0),
        std::make_pair("_concat_box_conf", 1)};

        virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const nvuffparser::FieldCollection fc) override
        {
            assert(isPlugin(layerName));

            const nvuffparser::FieldMap* fields = fc.fields;
            int nbFields = fc.nbFields;

            if(!strcmp(layerName, "_PriorBox"))
            {
                assert(mPluginPriorBox == nullptr);
                assert(nbWeights == 0 && weights == nullptr);

                float minScale = 0.2, maxScale = 0.95;
                int numLayers;
                std::vector<float> aspectRatios;
                std::vector<int> fMapShapes;
                std::vector<float> layerVariances;

                for(int i = 0; i < nbFields; i++)
                {
                    const char* attr_name = fields[i].name;
                    if (strcmp(attr_name, "numLayers") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        numLayers = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "minScale") == 0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        minScale = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "maxScale") == 0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        maxScale = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "aspectRatios")==0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        int size = fields[i].length;
                        aspectRatios.reserve(size);
                        const double *aR = static_cast<const double*>(fields[i].data);
                        for(int j=0; j < size; j++)
                        {
                            aspectRatios.push_back(*aR);
                            aR++;
                        }
                    }
                    else if(strcmp(attr_name, "featureMapShapes")==0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        int size = fields[i].length;
                        fMapShapes.reserve(size);
                        const int *fMap = static_cast<const int*>(fields[i].data);
                        for(int j=0; j<size; j++){
                            fMapShapes.push_back(*fMap);
                            fMap++;
                        }
                    }
                    else if(strcmp(attr_name, "layerVariances")==0)
                    {
                        assert(fields[i].type == FieldType::kFLOAT);
                        int size = fields[i].length;
                        layerVariances.reserve(size);
                        const double *lVar = static_cast<const double*>(fields[i].data);
                        for(int j=0; j<size; j++){
                            layerVariances.push_back(*lVar);
                            lVar++;
                        }
                    }
                }
                // Num layers should match the number of feature maps from which boxes are predicted.
                assert(numLayers > 0);
                assert((int)fMapShapes.size() == numLayers);
                assert(aspectRatios.size() > 0);
                assert(layerVariances.size() == 4);

                // Reducing the number of boxes predicted by the first layer.
                // This is in accordance with the standard implementation.
                std::vector<float> firstLayerAspectRatios;

                int numFirstLayerARs = 3;
                for(int i = 0; i < numFirstLayerARs; ++i){
                    firstLayerAspectRatios.push_back(aspectRatios[i]);
                }
                // A comprehensive list of box parameters that are required by anchor generator
                GridAnchorParameters boxParams[numLayers];
                for(int i = 0; i < numLayers ; i++)
                {
                    if(i == 0)
                        boxParams[i] = {minScale, maxScale, firstLayerAspectRatios.data(), (int)firstLayerAspectRatios.size(), fMapShapes[i], fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
                    else
                        boxParams[i] = {minScale, maxScale, aspectRatios.data(), (int)aspectRatios.size(), fMapShapes[i], fMapShapes[i], {layerVariances[0], layerVariances[1], layerVariances[2], layerVariances[3]}};
                }

                mPluginPriorBox = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDAnchorGeneratorPlugin(boxParams, numLayers), nvPluginDeleter);
                return mPluginPriorBox.get();
            }
            else if(concatIDs.find(std::string(layerName)) != concatIDs.end())
            {
                const int i = concatIDs[layerName];
                assert(mPluginFlattenConcat[i] == nullptr);
                assert(nbWeights == 0 && weights == nullptr);
                mPluginFlattenConcat[i] = std::unique_ptr<FlattenConcat>(new FlattenConcat(concatAxis[i], ignoreBatch[i]));
                return mPluginFlattenConcat[i].get();
            }
            else if(!strcmp(layerName, "_concat_priorbox"))
            {
                assert(mPluginConcat == nullptr);
                assert(nbWeights == 0 && weights == nullptr);
                mPluginConcat = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createConcatPlugin(2, true), nvPluginDeleter);
                return mPluginConcat.get();
            }
            else if(!strcmp(layerName, "_NMS"))
            {

                assert(mPluginDetectionOutput == nullptr);
                assert(nbWeights == 0 && weights == nullptr);

                 // Fill the custom attribute values to the built-in plugin according to the types
                for(int i = 0; i < nbFields; ++i)
                {
                    const char* attr_name = fields[i].name;
                    if (strcmp(attr_name, "iouThreshold") == 0)
                    {
                        detectionOutputParam.nmsThreshold =(float)(*(static_cast<const double*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "numClasses") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.numClasses = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "maxDetectionsPerClass") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.topK = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "scoreConverter") == 0)
                    {
                        std::string scoreConverter(static_cast<const char*>(fields[i].data), fields[i].length);
                        if(scoreConverter=="SIGMOID")
                            detectionOutputParam.confSigmoid = true;
                    }
                    else if(strcmp(attr_name, "maxTotalDetections") == 0)
                    {
                        assert(fields[i].type == FieldType::kINT32);
                        detectionOutputParam.keepTopK = (int)(*(static_cast<const int*>(fields[i].data)));
                    }
                    else if(strcmp(attr_name, "scoreThreshold") == 0)
                    {
                        detectionOutputParam.confidenceThreshold = (float)(*(static_cast<const double*>(fields[i].data)));
                    }
                }
                mPluginDetectionOutput = std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createSSDDetectionOutputPlugin(detectionOutputParam), nvPluginDeleter);
                return mPluginDetectionOutput.get();
            }
            else
            {
              assert(0);
              return nullptr;
            }
        }

    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));

        if (!strcmp(layerName, "_PriorBox"))
        {
            assert(mPluginPriorBox == nullptr);
            mPluginPriorBox = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createSSDAnchorGeneratorPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginPriorBox.get();
        }
        else if (concatIDs.find(std::string(layerName)) != concatIDs.end())
        {
            const int i = concatIDs[layerName];
            assert(mPluginFlattenConcat[i] == nullptr);
            mPluginFlattenConcat[i] = std::unique_ptr<FlattenConcat>(new FlattenConcat(serialData, serialLength));
            return mPluginFlattenConcat[i].get();
        }
        else if (!strcmp(layerName, "_concat_priorbox"))
        {
            assert(mPluginConcat == nullptr);
            mPluginConcat = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createConcatPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginConcat.get();
        }
        else if (!strcmp(layerName, "_NMS"))
        {
            assert(mPluginDetectionOutput == nullptr);
            mPluginDetectionOutput = std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
            return mPluginDetectionOutput.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    bool isPlugin(const char* name) override
    {
        return !strcmp(name, "_PriorBox")
            || concatIDs.find(std::string(name)) != concatIDs.end()
            || !strcmp(name, "_concat_priorbox")
            || !strcmp(name, "_NMS")
            || !strcmp(name, "mbox_conf_reshape");
    }

    // The application has to destroy the plugin when it knows it's safe to do so.
    void destroyPlugin()
    {
        for (unsigned i = 0; i < concatIDs.size(); ++i)
        {
            mPluginFlattenConcat[i].reset();
        }
        mPluginConcat.reset();
        mPluginPriorBox.reset();
        mPluginDetectionOutput.reset();
    }

    void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginPriorBox{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginDetectionOutput{nullptr, nvPluginDeleter};
    std::unique_ptr<INvPlugin, void (*)(INvPlugin*)> mPluginConcat{nullptr, nvPluginDeleter};
    std::unique_ptr<FlattenConcat> mPluginFlattenConcat[2]{nullptr, nullptr};
};
