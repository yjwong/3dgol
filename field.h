#ifndef GAME_OF_LIFE_FIELD_H
#define GAME_OF_LIFE_FIELD_H

#include <string>

namespace GameOfLife {

class Field {
private:
    Field();
    int _size;
    int* field;
    int* fieldOnDevice;

    int _getAllocationSize();

public:
    Field(int size);
    Field(int size, const std::string fileName);
    ~Field();

    int& at(int x, int y, int z);
    int* data();
    int size();

    void dump();
    void toStream(std::ostream& stream);
    void toFile(const std::string fileName);
    
    int* allocateOnDevice();
    void transferToDevice();
    void transferFromDevice();

    int operator[](const int& index);
};

}

#endif /* GAME_OF_LIFE_FIELD_H */

/* vim: set ts=4 sw=4 et: */
