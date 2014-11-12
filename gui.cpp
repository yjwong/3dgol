#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

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
	if (argc != 7 && argc != 8) {
		std::cout << argv[0] << ": Insufficient arguments to the program." <<
			std::endl;
		std::cout << argv[0] << ": Required arguments are (in order): " <<
			std::endl;
		std::cout << argv[0] << ":     sz, r1, r2, r3, r4, sp" <<
			std::endl;
		std::cout << argv[0] << ": Optional arguments are: fl" << std::endl;
		return EXIT_INSUFFICIENT_ARGS;
	}

	// Initialize arguments.
	int sz, r1, r2, r3, r4, sp;
	std::string fl;
	try {
		sz = cstrToInt(argv[1]);
		r1 = cstrToInt(argv[2]);
		r2 = cstrToInt(argv[3]);
		r3 = cstrToInt(argv[4]);
		r4 = cstrToInt(argv[5]);
		sp = cstrToInt(argv[6]);
	} catch (std::domain_error &e) {
		std::cout << argv[0] << ": All arguments except for fl must be " <<
			"integers." << std::endl;
		return EXIT_INVALID_ARGS;
	}

	// Parse the last optional argument.
	if (argc == 8) {
		fl.append(argv[7]);
	}

	// Initialize the Game of Life.
	GameOfLife::GameOfLife program (sz, r1, r2, r3, r4, fl);
    GameOfLife::Display display (&program, sp);
    return EXIT_OK;
}

/* vim: set ts=4 sw=4 et: */
