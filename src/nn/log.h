#include <iostream>
#include "glog/logging.h"
#include "NvInfer.h"

using namespace nvinfer1;

class Logger : public ILogger           
{
public:
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};
