#ifndef GAME_OF_LIFE_KERNELS_H
#define GAME_OF_LIFE_KERNELS_H

#include "field.h"

namespace GameOfLife {

extern "C" void golPerformSimulation(Field* field, Field* outField, int r1,
        int r2, int r3, int r4);
extern "C" void golPerformSimulationMulti(Field* field, Field* outField,
        int r1, int r2, int r3, int r4, int count);

}

#endif /* GAME_OF_LIFE_KERNELS_H */

/* vim: set ts=4 sw=4 et: */
