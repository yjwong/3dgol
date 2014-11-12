#include <iostream>
#include <stdexcept>
#include "util.h"

namespace GameOfLife {

void Util::cudaThrowError(const std::string message,
        const cudaError_t error) {
    std::string out;
    out.append(message)
        .append(" (")
        .append(cudaGetErrorName(error))
        .append(": ")
        .append(cudaGetErrorString(error))
        .append(").");
    throw std::runtime_error(out);
}

void Util::cudaPrintError(const std::string message,
        const cudaError_t error) {
    std::cout << message << " (" << cudaGetErrorName(error) << ": " <<
        cudaGetErrorString(error) << ")." << std::endl;
}

}

/* vim: set ts=4 sw=4 et: */
