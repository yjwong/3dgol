#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

#include "util.h"
#include "field.h"

namespace GameOfLife {

Field::Field(int size) : _size(size) {
    // Allocate memory on the host.
    int allocationSize = this->_getAllocationSize();
    cudaError_t result = cudaMallocHost(&this->field, allocationSize);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to allocate memory for field", result);
    }

    //this->field = new int[size * size * size];
}

Field::Field(int size, const std::string fileName) : _size(size) {
    // Allocate memory on the host.
    int allocationSize = this->_getAllocationSize();
    cudaError_t result = cudaMallocHost(&this->field, allocationSize);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to allocate memory for field", result);
    }

    //this->field = new int[size * size * size];

    // Read data from file.
    std::ifstream file (fileName.c_str());
    if (file.fail()){ 
        std::string errorStr;
        errorStr.append("Failed opening file ")
            .append(fileName)
            .append(" for reading");
        throw std::runtime_error(errorStr);
    }

    std::string line;
    int i = 0, j = 0;
    while (std::getline(file, line)) {
        std::istringstream iss (line);
        int k = 0;

        // If the first number is -1, then we skip to the next plane.
        int item;
        iss >> item;
        if (item == -1) {
            i++;
            j = 0;
            continue;
        } else {
            this->at(i, j, k) = item;
            k++;
        }

        // Process the rest of the items.
        while (iss >> item) {
            this->at(i, j, k) = item;
            k++;
        }

        j++;
    }
}

Field::~Field() {
    cudaError_t result = cudaFreeHost(this->field);
    if (result != cudaSuccess) {
        Util::cudaPrintError("Failed to free memory for field", result);
    }

    //delete[] this->field;
}

int Field::_getAllocationSize() {
    return this->_size * this->_size * this->_size * sizeof(int);
}

int& Field::at(int x, int y, int z) {
    if (x >= this->_size || y >= this->_size || z >= this->_size) {
        throw std::range_error("Index supplied was out of range");
    }

    return this->field[
        x * this->_size * this->_size +
        y * this->_size + z
    ];
}

int* Field::data() {
    return this->field;
}

int Field::size() {
    return this->_size;
}

void Field::dump() {
    for (int i = 0; i < this->_size; i++) {
        for (int j = 0; j < this->_size; j++) {
            for (int k = 0; k < this->_size; k++) {
                std::cout << "(" << i << ", " << j << ", " << k << "): " <<
                    this->at(i, j, k) << std::endl;
            }
        }
    }
}

void Field::toStream(std::ostream& stream) {
    for (int i = 0; i < this->_size; i++) {
        for (int j = 0; j < this->_size; j++) {
            for (int k = 0; k < this->_size; k++) {
                stream << this->at(i, j, k) << " "; 
            }

            stream << std::endl;
        }

        stream << "-1" << std::endl;
    }
}

void Field::toFile(const std::string fileName) {
    std::ofstream file (fileName.c_str());
    this->toStream(file);
}

int* Field::allocateOnDevice() {
    // Allocate memory on the GPU.
    cudaError_t result;
    size_t allocationSize = this->_getAllocationSize();
    result = cudaMalloc(&this->fieldOnDevice, allocationSize);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to allocate memory on device", result);
    }
    
    return this->fieldOnDevice;
}

void Field::transferToDevice() {
    if (this->fieldOnDevice == NULL) {
        throw std::runtime_error("Memory not allocated before transfer to "
                "device.");
    }

    // Transfer the field over.
    size_t allocationSize = this->_getAllocationSize();
    cudaError_t result = cudaMemcpy(this->fieldOnDevice, this->field,
            allocationSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to copy field to device", result);
    }
}

void Field::transferFromDevice() {
    if (fieldOnDevice == NULL) {
        throw std::runtime_error("Memory not allocated before transfer from "
                "device.");
    }
    
    // Transfer the field back.
    size_t allocationSize = this->_getAllocationSize();
    cudaError_t result = cudaMemcpy(this->field, this->fieldOnDevice,
            allocationSize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        Util::cudaThrowError("Failed to copy field from device", result);
    }
}

int Field::operator[](const int& index) {
    return this->field[index];
}

}

/* vim: set ts=4 sw=4 et: */
