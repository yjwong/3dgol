# CMakeLists file for CS3210 Parallel Computing Project 2
# 3D Game of Life Implementation on GPU
# (C) 2014 Wong Yong Jie

cmake_minimum_required (VERSION 2.8)
project (3dgol)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Options for this project.
option (USE_SYSTEM_GLFW "Use the GLFW library installed on the system" OFF)
option (USE_SYSTEM_GLM "Use the GLM library installed on the system" ON)

# Find the dependencies for this project.
find_package (CUDA REQUIRED)
find_package (OpenGL REQUIRED)
find_package (GLEW REQUIRED)
find_package (GLM REQUIRED)

if (USE_SYSTEM_GLFW)
    find_package (GLFW REQUIRED)
    include_directories (${GLFW_INCLUDE_DIRS})
else (USE_SYSTEM_GLFW)
    add_subdirectory ("glfw")
    include_directories ("glfw/include")
    set (GLFW_LIBRARIES "glfw ${GLFW_LIBRARIES}")
endif (USE_SYSTEM_GLFW)

if (USE_SYSTEM_GLM)
    find_package (GLM REQUIRED)
    include_directories (${GLM_INCLUDE_DIRS})
else (USE_SYSTEM_GLM)
    include_directories ("glm")
endif (USE_SYSTEM_GLM)

# Add required include paths.
include_directories (${OPENGL_INCLUDE_DIR})
include_directories (${GLEW_INCLUDE_DIRS})

# Add the executables.
cuda_add_executable (3dgol kernels.cu
    main.cpp 
    field.cpp
    util.cpp
    3dgol.cpp)
cuda_add_executable (3dgol_gui kernels.cu
    gui.cpp
    field.cpp
    display.cpp
    util.cpp
    3dgol.cpp)

# Link the executable with the required libraries.
target_link_libraries (3dgol
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES})
target_link_libraries (3dgol_gui
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES})

# vim: set ts=4 sw=4 et:
