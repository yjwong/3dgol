#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include "display.h"
#include "3dgol.h"

static const int EXIT_OK = 0;
static const int EXIT_INSUFFICIENT_ARGS = 1;
static const int EXIT_INVALID_ARGS = 2;
static const int EXIT_UNKNOWN_FAILURE = 3;

int cstrToInt(char* data) {
	std::istringstream ss(data);
	int num;
	if (!(ss >> num)) {
		throw std::domain_error("Supplied argument is not an integer.");
	}

	return num;
}

int main(int argc, char* argv[]) {
	// Check for arguments.
	if (argc != 9 && argc != 10) {
		std::cout << argv[0] << ": Insufficient arguments to the program." <<
			std::endl;
		std::cout << argv[0] << ": Required arguments are (in order): " <<
			std::endl;
		std::cout << argv[0] << ":     dp, sz, r1, r2, r3, r4, sp, gen" <<
			std::endl;
		std::cout << argv[0] << ": Optional arguments are: fl" << std::endl;
		return EXIT_INSUFFICIENT_ARGS;
	}

	// Initialize arguments.
	int dp, sz, r1, r2, r3, r4, sp, gen;
	std::string fl;
	try {
		dp = cstrToInt(argv[1]);
		sz = cstrToInt(argv[2]);
		r1 = cstrToInt(argv[3]);
		r2 = cstrToInt(argv[4]);
		r3 = cstrToInt(argv[5]);
		r4 = cstrToInt(argv[6]);
		sp = cstrToInt(argv[7]);
		gen = cstrToInt(argv[8]);
	} catch (std::domain_error &e) {
		std::cout << argv[0] << ": All arguments except for fl must be " <<
			"integers." << std::endl;
		return EXIT_INVALID_ARGS;
	}

	// Parse the last optional argument.
	if (argc == 10) {
		fl.append(argv[9]);
	}

	// Initialize the Game of Life.
	GameOfLife::GameOfLife gameOfLife (sz, r1, r2, r3, r4, fl);
    
    // Optimized case: minimize copying to and fro the GPU.
    if (dp == 0 && gen != 0) {
        gameOfLife.iterate(gen);
    } else {
        // Unoptimized case.
        for (int i = 0; i < gen || gen == 0; i++) {
            gameOfLife.iterate();
            if (dp != 0 && i % dp == 0) {
                gameOfLife.getField()->toStream(std::cout);
            }

            if (sp != 0) {
                // We'd have liked to use the C++11 features here, but the CUDA
                // frontend does not support it.
                //std::this_thread::sleep_for(std::chrono::milliseconds(sp));
                usleep(sp * 1000);
            }
        }
    }

    gameOfLife.getField()->toFile("final.txt");
    return EXIT_OK;
}

/* vim: set ts=4 sw=4 et: */
