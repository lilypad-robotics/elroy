#include <iostream>
#include <cstring>

#include "factory.h"

bool SSDPluginFactory::isPlugin(const char* name) {
    std::cout << "Plugin " << name << std::endl;
    if (strcmp(name, "_NMS") == 0) {
        return true;
    } else if (strcmp(name, "_concat_box_loc") == 0) {
        return true;
    } else if (strcmp(name, "_concat_box_conf") == 0) {
        return true;
    } else if (strcmp(name, "_concat_priorbox") == 0) {
        return true;
    } else if (strcmp(name, "_PriorBox") == 0) {
        return true;
    } 

    return false;
}

nvinfer1::IPlugin* SSDPluginFactory::createPlugin(const char* name, const nvinfer1::Weights *weights, int nbWeights, const nvuffparser::FieldCollection fc) { 
    std::cout << "Create1 " << name << std::endl;
    return nullptr;
}

nvinfer1::IPlugin* SSDPluginFactory::createPlugin(const char* name, const void* serialData, size_t serialLength) { 
    std::cout << "Create2 " << name << std::endl;
    return nullptr;
}
