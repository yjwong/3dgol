#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <string>
#include "field.h"

class GLFWwindow;
namespace GameOfLife {

class GameOfLife {
private:
    int sz, r1, r2, r3, r4;
    std::string fl;
    Field* field;
    
    void cudaCheck();
    void initializeField();

public:
	GameOfLife (int sz, int r1, int r2, int r3, int r4, std::string fl);
    ~GameOfLife();

    Field* getField();
    int getFieldSize();
    void iterate();
    void iterate(int count);
};

}

#endif /* GAME_OF_LIFE_H */

/* vim: set ts=4 sw=4 et: */
