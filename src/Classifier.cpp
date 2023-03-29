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
    printf("Bindings: %d\n", nbBindings);
    for (int i = 0; i < nbBindings; i++) {
        const char *name = m_engine->getBindingName(i);
        bool isInput = m_engine->bindingIsInput(i);
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        std::string fmtDims = FormatDims(dims);
        printf("  [%d] \"%s\" %s [%s]\n", i, name, isInput ? "input" : "output", fmtDims.c_str());
    }
}

// I/O utilities

void ReadClasses(const char *path, std::vector<std::string> &classes) {
    std::string line;
    std::ifstream ifs(path, std::ios::in);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }
    ifs.close();
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

void ReadInput(const char *path, std::vector<float> &input) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        Error("Cannot open %s", path);
    }
    size_t size = 3 * 224 * 224;
    input.resize(size);
    ifs.read(reinterpret_cast<char *>(input.data()), size * sizeof(float));
    ifs.close();
}

// main program

// int main(int argc, char *argv[]) {
//     if (argc != 3) {
//         fprintf(stderr, "Usage: trt_infer_plan <plan_path> <input_path>\n");
//         return 1;
//     }
//     const char *planPath = argv[1];
//     const char *inputPath = argv[2];

//     printf("Start %s\n", planPath);

//     std::vector<std::string> classes;
//     ReadClasses("imagenet_classes.txt", classes);

//     std::vector<char> plan;
//     ReadPlan(planPath, plan);

//     std::vector<float> input;
//     ReadInput(inputPath, input);

//     std::vector<float> output;

//     Engine engine;
//     engine.Init(plan);
//     engine.DiagBindings();
//     engine.Infer(input, output);

//     Softmax(static_cast<int>(output.size()), output.data());
//     PrintOutput(output, classes);

//     return 0;
// }