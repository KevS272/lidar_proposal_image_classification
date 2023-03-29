#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <NvInfer.h>

#include <lidar_proposal_image_classification/common.hpp>

class Engine {
public:
    Engine();
    ~Engine();
public:
    void ReadPlan(const char *path, std::vector<char> &plan);
    void Init(const std::vector<char> &plan);
    void Infer(const std::vector<float> &input, std::vector<float> &output, int img_size, int batch_size, int num_classes);
    void DiagBindings();
private:
    bool m_active;
    Logger m_logger;
    UniquePtr<nvinfer1::IRuntime> m_runtime;
    UniquePtr<nvinfer1::ICudaEngine> m_engine;
    UniquePtr<nvinfer1::IExecutionContext> context;
};