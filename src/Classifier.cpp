#include <lidar_proposal_image_classification/Classifier.hpp>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <NvInfer.h>

// wrapper class for inference engine
Engine::Engine(): m_active(false) { }

Engine::~Engine() { }

void Engine::Init(const std::vector<char> &plan) {
    assert(!m_active);
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (m_runtime == nullptr) {
        Error("Error creating infer runtime");
    }
    m_engine.reset(m_runtime->deserializeCudaEngine(plan.data(), plan.size(), nullptr));
    if (m_engine == nullptr) {
        Error("Error deserializing CUDA engine");
    }
    m_active = true;

    context.reset(m_engine->createExecutionContext());
    if (context == nullptr) {
        Error("Error creating execution context");
    }
}

void Engine::Infer(const std::vector<float> &input, std::vector<float> &output, int img_size, int batch_size, int num_classes) {
    assert(m_active);
    context->setBindingDimensions(0, nvinfer1::Dims4(batch_size, 3, img_size, img_size));
    CudaBuffer<float> inputBuffer;
    inputBuffer.Init(batch_size * 3 * img_size * img_size);
    assert(inputBuffer.Size() == input.size());
    inputBuffer.Put(input.data());
    CudaBuffer<float> outputBuffer;
    outputBuffer.Init(num_classes*batch_size);
    void *bindings[2];
    bindings[0] = inputBuffer.Data();
    bindings[1] = outputBuffer.Data();
    bool ok = context->executeV2(bindings);
    output.resize(outputBuffer.Size());
    outputBuffer.Get(output.data());
}

void Engine::DiagBindings() {
    int nbBindings = static_cast<int>(m_engine->getNbBindings());
    printf("TensorRT Classifier Engine Bindings: %d\n", nbBindings);
    for (int i = 0; i < nbBindings; i++) {
        const char *name = m_engine->getBindingName(i);
        bool isInput = m_engine->bindingIsInput(i);
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        std::string fmtDims = FormatDims(dims);
        printf("  [%d] \"%s\" %s [%s]\n", i, name, isInput ? "input" : "output", fmtDims.c_str());
    }
}

void Engine::ReadPlan(const char *path, std::vector<char> &plan) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    ifs.seekg(0, ifs.end);
    size_t size = ifs.tellg();
    plan.resize(size);
    ifs.seekg(0, ifs.beg);
    ifs.read(plan.data(), size);
    ifs.close();
}