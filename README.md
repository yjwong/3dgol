3D Game Of Life
===============

This project is part of CS3210 Parallel Computing, National University of
Singapore. It implements Conway's Game of Life using CUDA kernels.

Pre-Requisites
--------------

Few libraries are required for this project:

 * [GLFW](http://www.glfw.org/)
 * [GLEW](http://glew.sourceforge.net/)
 * [GLM](http://glm.g-truc.net/0.9.5/index.html)
 * [nVidia CUDA SDK](http://www.nvidia.com/object/cuda_home_new.html)

Additionally, [CMake](http://www.cmake.org/) is needed to build the project.

If you're using Ubuntu, CMake and the libraries can be installed by issuing the
following command:

    sudo apt-get install cmake libglfw3-dev libglew-dev libglm-dev

The nVidia CUDA SDK can be downloaded from nVidia's website.

Compilation
-----------

 1. `mkdir build && cd build`
 2. `cmake ..`
 3. `make -jN`, where N is the number of parallel jobs

When done, the resulting binaries should be found as `3dgol` and `3dgol_gui`.

Running
-------

`3dgol` is the command-line implementation. `3dgol_gui` is the implementation
with a 3D display. Run each command without parameters to learn about the
required and optional parameters.

