#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

#include "util.h"
#include "kernels.h"
#include "3dgol.h"

namespace GameOfLife {

GameOfLife::GameOfLife(int sz, int r1, int r2, int r3, int r4, std::string fl) :
    sz(sz), r1(r1), r2(r2), r3(r3), r4(r4), fl(fl) {
    // Set up CUDA.
    cudaCheck();
    this->field = new Field(sz);
    
    // Initialize the field.
    this->initializeField();
}

GameOfLife::~GameOfLife() {
    delete this->field;
}

void GameOfLife::cudaCheck() {
    int deviceCount;
    cudaError_t result;
   
    // Check if we have a CUDA device.
    result = cudaGetDeviceCount(&deviceCount);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to get number of CUDA-capable devices",
                result);
    }

    std::cout << "Total number of CUDA-capable devices: " << deviceCount <<
        std::endl;

    // Print CUDA device properties.
    int device;
    cudaDeviceProp deviceProperties;
    result = cudaGetDevice(&device);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to get CUDA device", result);
    }

    result = cudaGetDeviceProperties(&deviceProperties, device);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to get CUDA device properties", result);
    }

    std::cout << "CUDA device " << device << " selected: " <<
        deviceProperties.name << " (Compute Capability " <<
        deviceProperties.major << "." << deviceProperties.minor << ")" <<
        std::endl;
}

void GameOfLife::initializeField() {
    // Read from fl if not empty.
    if (fl.empty()) {
        std::srand(std::time(0));

        for (int i = 0; i < this->sz; i++) {
            for (int j = 0; j < this->sz; j++) {
                for (int k = 0; k < this->sz; k++) {
                    int random = std::rand();
                    this->field->at(i, j, k) = random % 2;
                }
            }
        }

    } else {
        this->field = new Field(this->sz, fl);
        std::cout << "Initial values read from file successfully." <<
            std::endl;
    }
    
    //this->field->dump();
}

Field* GameOfLife::getField() {
    return this->field;
}

int GameOfLife::getFieldSize() {
    return this->sz;
}

void GameOfLife::iterate() {
    // Run the kernel.
    Field* outField = new Field(this->sz);
    golPerformSimulation(this->field, outField, this->r1, this->r2,
            this->r3, this->r4);
    delete this->field;
    this->field = outField;
}

void GameOfLife::iterate(int count) {
    Field* outField = new Field(this->sz);
    golPerformSimulationMulti(this->field, outField, this->r1, this->r2,
            this->r3, this->r4, count);
    delete this->field;
    this->field = outField;
}

}


/* vim: set ts=4 sw=4 et: */
