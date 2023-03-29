#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cmath>
#include <string>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <lidar_proposal_image_classification/common.hpp>
#include <iostream>

// error handling

void Error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(1);
}

// wall clock

Timer::Timer() { }

Timer::~Timer() { }

void Timer::Start() {
    start = std::chrono::steady_clock::now();
}

void Timer::StartTotal() {
    start_total = std::chrono::steady_clock::now();
}

void Timer::Stop(const std::string &message) {
    end = std::chrono::steady_clock::now();
    if(print_timer){
        std::cout << "[LiProIC][TIME] " << message << ": " <<  (std::chrono::duration<double, std::milli>(end - start)).count()  << "ms"  <<std::endl;
    }
}

void Timer::StopTotal() {
    end_total = std::chrono::steady_clock::now();
    if(print_timer){
        std::cout << "[LiProIC][TIME TOTAL] " << (std::chrono::duration<double, std::milli>(end_total - start_total)).count()  << "ms"  <<std::endl;
    }
}

// CUDA helpers

void CallCuda(cudaError_t stat) {
    if (stat != cudaSuccess) {
        Error("%s", cudaGetErrorString(stat));
    }
}

void *Malloc(int size) {
    void *ptr = nullptr;
    CallCuda(cudaMalloc(&ptr, size));
    return ptr;
}

void Free(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void Memget(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void Memput(void *dst, const void *src, int size) {
    CallCuda(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

// general helpers

std::vector<float> new_softmax(const std::vector<float>& x) {
    // Compute exponentials of input vector elements
    std::vector<float> exp_x(x.size());
    float max_x = x[0];
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] > max_x) {
            max_x = x[i];
        }
    }
    float exp_sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        exp_x[i] = std::exp(x[i] - max_x);
        exp_sum += exp_x[i];
    }
    // Compute softmax output vector
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = exp_x[i] / exp_sum;
    }
    return y;
}

// TensorRT helpers

std::string FormatDims(const nvinfer1::Dims &dims) {
    std::string result;
    char buf[64];
    int nbDims = static_cast<int>(dims.nbDims);
    for (int i = 0; i < nbDims; i++) {
        if (i > 0) {
            result += " ";
        }
        sprintf(buf, "%d", static_cast<int>(dims.d[i]));
        result += buf;
    }
    return result;
}

// logger

Logger::Logger():
        m_severityLevel(nvinfer1::ILogger::Severity::kWARNING) { }

Logger::~Logger() { }

nvinfer1::ILogger::Severity Logger::SeverityLevel() const {
    return m_severityLevel;
}

void Logger::SetSeverityLevel(nvinfer1::ILogger::Severity level) {
    m_severityLevel = level;
}

void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    if (severity > m_severityLevel) {
        return;
    }
    fprintf(stderr, "%s: %s\n", GetSeverityString(severity), msg);
}

const char *Logger::GetSeverityString(nvinfer1::ILogger::Severity severity) {
    using T = nvinfer1::ILogger::Severity;
    switch (severity) {
    case T::kINTERNAL_ERROR:
        return "INTERNAL_ERROR";
    case T::kERROR:
        return "ERROR";
    case T::kWARNING:
        return "WARNING";
    case T::kINFO:
        return "INFO";
    case T::kVERBOSE:
        return "VERBOSE";
    default:
        return "?";
    }
}
