#ifndef GAME_OF_LIFE_UTIL_H
#define GAME_OF_LIFE_UTIL_H

#include <string>
#include <cuda_runtime.h>

namespace GameOfLife {

class Util {
public:
    static void cudaThrowError(const std::string message,
            const cudaError_t error);
    static void cudaPrintError(const std::string message,
            const cudaError_t error);
};

}

#endif /* GAME_OF_LIFE_UTIL_H */

/* vim: set ts=4 sw=4 et: */
