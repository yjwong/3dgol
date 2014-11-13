#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <GL/glew.h>

#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>

#include "field.h"
#include "3dgol.h"
#include "display.h"

namespace GameOfLife {

GLfloat Display::LIGHT_AMBIENT[] = { 0.25f, 0.25f, 0.25f, 1.0f };
GLfloat Display::LIGHT_DIFFUSE[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat Display::LIGHT_SPECULAR[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat Display::LIGHT_POSITION[] = { 100.0f, 50.0f, 100.0f };

GLfloat Display::CUBE_VERTICES[] = {
    // Left Face
    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    // Right Face
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,

     1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,

    // Top Face
    -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,

     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,

    // Bottom Face
    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,

     1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,

    // Back Face
    -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,

     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,

    // Front Face
    -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

     1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f
};

GLfloat Display::CUBE_COLORS[] = {
    // Left Face
    0.956f, 0.262f, 0.212f,
    0.956f, 0.262f, 0.212f,
    0.956f, 0.262f, 0.212f,
    0.956f, 0.262f, 0.212f,
    0.956f, 0.262f, 0.212f,
    0.956f, 0.262f, 0.212f,

    // Right Face
    0.0f, 0.585f, 0.531f,
    0.0f, 0.585f, 0.531f,
    0.0f, 0.585f, 0.531f,
    0.0f, 0.585f, 0.531f,
    0.0f, 0.585f, 0.531f,
    0.0f, 0.585f, 0.531f,

    // Top Face
    0.612f, 0.153f, 0.690f,
    0.612f, 0.153f, 0.690f,
    0.612f, 0.153f, 0.690f,
    0.612f, 0.153f, 0.690f,
    0.612f, 0.153f, 0.690f,
    0.612f, 0.153f, 0.690f,
    
    // Bottom Face
    0.804f, 0.863f, 0.224f,
    0.804f, 0.863f, 0.224f,
    0.804f, 0.863f, 0.224f,
    0.804f, 0.863f, 0.224f,
    0.804f, 0.863f, 0.224f,
    0.804f, 0.863f, 0.224f,

    // Back Face
    0.129f, 0.588f, 0.953f,
    0.129f, 0.588f, 0.953f,
    0.129f, 0.588f, 0.953f,
    0.129f, 0.588f, 0.953f,
    0.129f, 0.588f, 0.953f,
    0.129f, 0.588f, 0.953f,

    // Front Face
    0.913f, 0.117f, 0.388f,
    0.913f, 0.117f, 0.388f,
    0.913f, 0.117f, 0.388f,
    0.913f, 0.117f, 0.388f,
    0.913f, 0.117f, 0.388f,
    0.913f, 0.117f, 0.388f
};

GLfloat Display::CUBE_NORMALS[] = {
    // Left Face
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,

    // Right Face
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,

    // Top Face
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,

    // Bottom Face
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,
    0.0f,  1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,
    0.0f, -1.0f, 0.0f,

    // Back Face
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f,  1.0f,
    0.0f, 0.0f,  1.0f,
    0.0f, 0.0f,  1.0f,

    // Front Face
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f,  1.0f,
    0.0f, 0.0f,  1.0f,
    0.0f, 0.0f,  1.0f
};

GLfloat Display::CUBE_AMBIENT[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat Display::CUBE_DIFFUSE[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat Display::CUBE_SPECULAR[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat Display::CUBE_SHININESS[] = { 32.0f };
GLfloat Display::CUBE_EMISSION[] = { 0.0f, 0.0f, 0.0f, 1.0f };

GLfloat Display::WIRE_CUBE_VERTICES[] = {
    // Side 1
    -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    // Cross Over
    -1.0f, -1.0f,  1.0f,
    // Side 2
     1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,
    // Side 3
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,
    // Side 4
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    // Side 5
    -1.0f, -1.0f,  1.0f
};

GLfloat Display::WIRE_CUBE_COLORS[] = {
    // Side 1
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    // Cross Over
     1.0f,  1.0f,  1.0f,
    // Side 2
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    // Side 3
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    // Side 4
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,
    // Side 5
     1.0f,  1.0f,  1.0f
};

GLfloat Display::AXES_VERTICES[] = {
    // X
    0.0f, 0.0f, 0.0f,
    2.0f, 0.0f, 0.0f,
    // Y
    0.0f, 0.0f, 0.0f,
    0.0f, 2.0f, 0.0f,
    // Z
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 2.0f
};

GLfloat Display::AXES_COLORS[] = {
    // X
    1.0f, 0.0f, 0.0f,
    1.0f, 0.0f, 0.0f,
    // Y
    0.0f, 1.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    // Z
    0.0f, 0.0f, 1.0f,
    0.0f, 0.0f, 1.0f
};

int Display::WINDOW_WIDTH = 800;
int Display::WINDOW_HEIGHT = 600;

Display::Display(GameOfLife* gameOfLife, int sp) :
    gameOfLife(gameOfLife), sp(sp),
    isAxesVisible(true), isRotateInProgress(false),
    rotateModeOriginalX(0.0), rotateModeOriginalY(0.0),
    rotationX(0.0f), rotationY(0.0f) {
    // Print some instructions.
    std::cout << std::endl;
    std::cout << "Keyboard controls: " << std::endl;
    std::cout << "\tn\tNext generation" << std::endl;
    std::cout << "\tw\tWrite current field to \"final.txt\"" << std::endl;
    std::cout << "\tx\tToggle axes" << std::endl;
    std::cout << std::endl;
    std::cout << "Mouse controls: " << std::endl;
    std::cout << "\tPrimary Drag\tRotate around field" << std::endl;
    std::cout << "\tScroll Wheel\tZoom" << std::endl;
    std::cout << std::endl;

    // Set up the error callback in case anything bad happens.
    glfwSetErrorCallback(&Display::onGlfwError);
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    // We should have an OpenGL 3.3 core profile.
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create a new window for us to render into.
    GLFWwindow* window = glfwCreateWindow(Display::WINDOW_WIDTH,
            Display::WINDOW_HEIGHT, "3D Game of Life", NULL, NULL);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }

    // Set the current OpenGL context.
    glfwMakeContextCurrent(window);

    // Initialize OpenGL.
    this->initializeGraphics();

    // Set up event handlers.
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, &Display::onGlfwFramebufferSize);
    glfwSetKeyCallback(window, &Display::onGlfwKey);
    glfwSetMouseButtonCallback(window, &Display::onGlfwMouseButton);
    glfwSetCursorPosCallback(window, &Display::onGlfwCursorPos);
    glfwSetScrollCallback(window, &Display::onGlfwScroll);

    // Set up the timer.
    if (sp != 0) {
        this->time = glfwGetTime();
        this->timeNextGen = this->time + sp / 1000.0;
    }

    // Begin the event loop.
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw a wireframe cube with axes.
        this->drawWireCube();

        // Draw the cubes.
        this->drawCube();

        // Request new generation based on timer.
        this->time = glfwGetTime();
        if (sp != 0 && this->time > this->timeNextGen) {
            this->gameOfLife->iterate();
            this->timeNextGen = this->time + sp / 1000.0;
        }
        
        // Display the output.
        glfwSwapBuffers(window);
        glfwPollEvents(); 
    }

    this->cleanUpGraphics();
    glfwDestroyWindow(window);
    glfwTerminate();


}

void Display::onGlfwError(int error, const char* desc) {
    std::cout << "GLFW Error: " << desc << " (errno " << error << ")" <<
        std::endl;
}

void Display::onGlfwFramebufferSize(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);

    Display* that = static_cast<Display*> (glfwGetWindowUserPointer(window));
    that->sceneProjection = glm::perspective(45.0f / 180 * glm::pi<float>(),
                1.0f * width / height, 0.1f, 100.0f);
}

void Display::onGlfwKey(GLFWwindow* window, int key, int scancode,
        int action, int mods) {
    Display* that = static_cast<Display*> (glfwGetWindowUserPointer(window));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    if (key == GLFW_KEY_N && action == GLFW_PRESS) {
        if (that == NULL) {
            throw std::runtime_error("Window user pointer is null!");
        }

        std::cout << "Moving to next generation..." << std::endl;
        that->gameOfLife->iterate();
    }

    if (key == GLFW_KEY_X && action == GLFW_PRESS) {
        that->isAxesVisible = !that->isAxesVisible;
    }

    if (key == GLFW_KEY_W && action == GLFW_PRESS) {
        that->gameOfLife->getField()->toFile("final.txt");
        std::cout << "Current field written to \"final.txt\"." << std::endl;
    }
}

void Display::onGlfwMouseButton(GLFWwindow* window, int button, int action,
        int mods) {
    if (button == GLFW_MOUSE_BUTTON_1) {
        Display* that = static_cast<Display*> (glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS) {
            double xpos;
            double ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            that->isRotateInProgress = true;
            that->rotateModeOriginalX = xpos;
            that->rotateModeOriginalY = ypos;
        } else {
            that->isRotateInProgress = false;
        }
    }
}

void Display::onGlfwCursorPos(GLFWwindow* window, double xpos, double ypos) {
    Display* that = static_cast<Display*> (glfwGetWindowUserPointer(window));
    if (that->isRotateInProgress) {
        double x = xpos - that->rotateModeOriginalX;
        double y = ypos - that->rotateModeOriginalY;
        that->rotationX += x / 180.0f * glm::pi<float>() * 0.25f;
        that->rotationY += y / 180.0f * glm::pi<float>() * 0.25f;

        // The order here is rather important.
        that->sceneModel = glm::rotate(that->rotationY,
                glm::vec3(1.0f, 0.0f, 0.0f));
        that->sceneModel *= glm::rotate(that->rotationX,
                glm::vec3(0.0f, 1.0f, 0.0f));

        // Save the current position value so we can compute the delta later.
        that->rotateModeOriginalX = xpos;
        that->rotateModeOriginalY = ypos;
    }
}

void Display::onGlfwScroll(GLFWwindow* window, double xoffset,
        double yoffset) {
    Display* that = static_cast<Display*> (glfwGetWindowUserPointer(window));
    if (that->sceneCameraCoord.z + yoffset > 2.0f) {
        that->sceneCameraCoord += glm::vec3(0.0, 0.0, yoffset);
        that->sceneView = glm::lookAt(
                that->sceneCameraCoord,
                that->sceneCameraLookAt,
                that->sceneCameraUp
        );
    }
}

void Display::initializeGraphics() {
    glewExperimental = GL_TRUE;
    GLenum result = glewInit();
    GLint linkResult = GL_FALSE;
    if (result != GLEW_OK) {
        std::string errorStr;
        errorStr.append("Failed to initialize GLEW (")
            .append(reinterpret_cast<const char*>(glewGetErrorString(result)))
            .append(").");
        throw std::runtime_error(errorStr);
    }

    // Sometimes glewInit has a phantom error, so let's clear the flag.
    glGetError();

    // Create a vertex array object.
    glGenVertexArrays(1, &this->vertexArrayObject);
    glBindVertexArray(this->vertexArrayObject);

    // Initialize vertex and fragment shaders.
    GLuint wireCubeVertexShader = createShader(GL_VERTEX_SHADER,
            "../wire_cube.vert");
    GLuint wireCubeFragmentShader = createShader(GL_FRAGMENT_SHADER,
            "../wire_cube.frag");
    GLuint cubeVertexShader = createShader(GL_VERTEX_SHADER, "../cube.vert");
    GLuint cubeFragmentShader = createShader(GL_FRAGMENT_SHADER,
            "../cube.frag");

    // Link the shader program.
    this->wireCubeProgram = glCreateProgram();
    glAttachShader(this->wireCubeProgram, wireCubeVertexShader);
    glAttachShader(this->wireCubeProgram, wireCubeFragmentShader);
    glLinkProgram(this->wireCubeProgram);
    glGetProgramiv(this->wireCubeProgram, GL_LINK_STATUS, &linkResult);
    if (linkResult != GL_TRUE) {
        printShaderInfoLog(this->wireCubeProgram);
        throw std::runtime_error("Failed to link shaders");
    }

    this->cubeProgram = glCreateProgram();
    glAttachShader(this->cubeProgram, cubeVertexShader);
    glAttachShader(this->cubeProgram, cubeFragmentShader);
    glLinkProgram(this->cubeProgram);
    glGetProgramiv(this->cubeProgram, GL_LINK_STATUS, &linkResult);
    if (linkResult != GL_TRUE) {
        printShaderInfoLog(this->cubeProgram);
        throw std::runtime_error("Failed to link shaders");
    }

    // Create a buffer object for the cube.
    glGenBuffers(1, &this->vboCubeVertices);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeVertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::CUBE_VERTICES),
            Display::CUBE_VERTICES, GL_STATIC_DRAW);
    
    glGenBuffers(1, &this->vboCubeColors);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeColors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::CUBE_COLORS),
            Display::CUBE_COLORS, GL_STATIC_DRAW);

    glGenBuffers(1, &this->vboCubeNormals);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeNormals);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::CUBE_NORMALS),
            Display::CUBE_NORMALS, GL_STATIC_DRAW);
    
    glGenBuffers(1, &this->vboWireCubeVertices);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboWireCubeVertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::WIRE_CUBE_VERTICES),
            Display::WIRE_CUBE_VERTICES, GL_STATIC_DRAW);
    
    glGenBuffers(1, &this->vboWireCubeColors);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboWireCubeColors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::WIRE_CUBE_COLORS),
            Display::WIRE_CUBE_COLORS, GL_STATIC_DRAW);

    glGenBuffers(1, &this->vboAxesVertices);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboAxesVertices);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::AXES_VERTICES),
            Display::AXES_VERTICES, GL_STATIC_DRAW);

    glGenBuffers(1, &this->vboAxesColors);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboAxesColors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Display::AXES_COLORS),
            Display::AXES_COLORS, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set up initial view matrix.
    this->rotationX = glm::pi<float>() / 5;
    this->rotationY = glm::pi<float>() / 5;
    this->sceneModel = glm::rotate(this->rotationX, glm::vec3(1.0, 0.0, 0.0));
    this->sceneModel *= glm::rotate(this->rotationY, glm::vec3(0.0, 1.0, 0.0));
    this->sceneCameraCoord = glm::vec3(0.0, 0.0, 5.0);
    this->sceneCameraLookAt = glm::vec3(0.0, 0.0, 0.0);
    this->sceneCameraUp = glm::vec3(0.0, 1.0, 0.0);
    this->sceneView = glm::lookAt(
           this->sceneCameraCoord,
           this->sceneCameraLookAt,
           this->sceneCameraUp
    );
    this->sceneProjection = glm::perspective(45.0f / 180 * glm::pi<float>(),
                1.0f * Display::WINDOW_WIDTH / Display::WINDOW_HEIGHT,
                0.1f, 100.0f);

    // Enable depth test.
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Enable blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Check if the rendering state is sane.
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cout << "glGetError() returned " << error << ", subsequent "
            "commands may fail." << std::endl;
    }
}

void Display::cleanUpGraphics() {
    glDeleteBuffers(1, &this->vboCubeVertices);
    glDeleteBuffers(1, &this->vboCubeColors);
    glDeleteBuffers(1, &this->vboWireCubeVertices);
    glDeleteBuffers(1, &this->vboWireCubeColors);
    glDeleteBuffers(1, &this->vboAxesVertices);
    glDeleteBuffers(1, &this->vboAxesColors);
}

GLuint Display::createShader(GLenum shaderType,
        const std::string shaderFile) {
    GLint compileResult = GL_FALSE;
    GLuint shader = glCreateShader(shaderType);
    std::ifstream shaderFileStream (shaderFile.c_str());
    if (shaderFileStream.fail()) {
        throw std::runtime_error("Failed to open shader from file");
    }

    std::string shaderSrc (
            (std::istreambuf_iterator<char>(shaderFileStream)),
            std::istreambuf_iterator<char>()
    );

    const char* shaderSrcStr = shaderSrc.c_str();
    glShaderSource(shader, 1, &shaderSrcStr, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileResult);
    if (compileResult != GL_TRUE) {
        printShaderInfoLog(shader);
        throw std::runtime_error("Failed to compile shader");
    }

    return shader;
}

void Display::printShaderInfoLog(GLhandleARB obj) {
    int length = 0;
    int charsWritten = 0;
    
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &length);
    char* infoLog = new char[length];

    if (length > 0) {
        glGetShaderInfoLog(obj, length, &charsWritten, infoLog);
        std::cout << infoLog << std::endl;
    }

    delete[] infoLog;
}

void Display::drawWireCube() {
    // Use the wireframe cube program.
    glUseProgram(this->wireCubeProgram);

    // Set up the color attribute.
    glBindFragDataLocation(this->wireCubeProgram, 0, "out_color");

    // Set up model view projection.
    GLint uniformModel = glGetUniformLocation(this->wireCubeProgram,
            "model");
    GLint uniformView = glGetUniformLocation(this->wireCubeProgram,
            "view");
    GLint uniformProjection = glGetUniformLocation(this->wireCubeProgram,
            "projection");

    // Set the uniform values.
    glUniformMatrix4fv(uniformModel, 1, GL_FALSE,
            glm::value_ptr(this->sceneModel));
    glUniformMatrix4fv(uniformView, 1, GL_FALSE,
            glm::value_ptr(this->sceneView));
    glUniformMatrix4fv(uniformProjection, 1, GL_FALSE,
            glm::value_ptr(this->sceneProjection));

    // Prepare the vertex buffer.
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboWireCubeVertices);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, this->vboWireCubeColors);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

    // Draw our wireframe cube.
    glDrawArrays(GL_LINE_STRIP, 0,
            sizeof(WIRE_CUBE_VERTICES) / sizeof(GLfloat) / 3);

    // Draw the axes.
    if (this->isAxesVisible) {
        glBindBuffer(GL_ARRAY_BUFFER, this->vboAxesVertices);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, this->vboAxesColors);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glDrawArrays(GL_LINES, 0,
                sizeof(AXES_VERTICES) / sizeof(GLfloat) / 3);
    }

    // Done drawing, clean up.
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
}

void Display::drawCube() {
    // Use the cube program.
    glUseProgram(this->cubeProgram);

    // Set up the color attribute.
    glBindFragDataLocation(this->cubeProgram, 0, "out_color");

    // Set up model view projection.
    GLint uniformModel = glGetUniformLocation(this->cubeProgram, "model");
    GLint uniformView = glGetUniformLocation(this->cubeProgram, "view");
    GLint uniformProjection = glGetUniformLocation(this->cubeProgram,
            "projection");
    GLint uniformNormal = glGetUniformLocation(this->cubeProgram,
            "normal_matrix");
    GLint uniformCount = glGetUniformLocation(this->cubeProgram, "count");
    GLint uniformFieldPosition = glGetUniformLocation(this->cubeProgram,
            "field_position");
    GLint uniformAmbient = glGetUniformLocation(this->cubeProgram,
            "mat_ambient");
    GLint uniformDiffuse = glGetUniformLocation(this->cubeProgram,
            "mat_diffuse");
    GLint uniformSpecular = glGetUniformLocation(this->cubeProgram,
            "mat_specular");
    GLint uniformShininess = glGetUniformLocation(this->cubeProgram,
            "mat_shininess");
    GLint uniformEmission = glGetUniformLocation(this->cubeProgram,
            "mat_emission");
    GLint uniformLightAmbient = glGetUniformLocation(this->cubeProgram,
            "light_ambient");
    GLint uniformLightDiffuse = glGetUniformLocation(this->cubeProgram,
            "light_diffuse");
    GLint uniformLightSpecular = glGetUniformLocation(this->cubeProgram,
            "light_specular");
    GLint uniformLightPosition = glGetUniformLocation(this->cubeProgram,
            "light_position");

    // Pre-calculate the normal matrix.
    glm::mat4 normalMatrix = glm::transpose(glm::inverse(
                this->sceneModel * this->sceneView
    ));
   
    // Generate an array with locations of cubes.
    std::vector<glm::vec3> fieldPositions;
    Field* field = this->gameOfLife->getField();
    for (int i = 0; i < field->size(); i++) {
        for (int j = 0; j < field->size(); j++) {
            for (int k = 0; k < field->size(); k++) {
                if (field->at(i, j, k) != 1) {
                    continue;
                }

                // Set the uniform values.
                glUniformMatrix4fv(uniformModel, 1, GL_FALSE,
                        glm::value_ptr(this->sceneModel));
                glUniformMatrix4fv(uniformView, 1, GL_FALSE,
                        glm::value_ptr(this->sceneView));
                glUniformMatrix4fv(uniformProjection, 1, GL_FALSE,
                        glm::value_ptr(this->sceneProjection));
                glUniformMatrix4fv(uniformNormal, 1, GL_FALSE,
                        glm::value_ptr(normalMatrix));
                glUniform1i(uniformCount, this->gameOfLife->getFieldSize());
                glUniform3f(uniformFieldPosition, i, j, k);

                glUniform4fv(uniformAmbient, 1, Display::CUBE_AMBIENT);
                glUniform4fv(uniformDiffuse, 1, Display::CUBE_DIFFUSE);
                glUniform4fv(uniformSpecular, 1, Display::CUBE_SPECULAR);
                glUniform1f(uniformShininess, Display::CUBE_SHININESS[0]);
                glUniform4fv(uniformEmission, 1, Display::CUBE_EMISSION);

                glUniform4fv(uniformLightAmbient, 1, Display::LIGHT_AMBIENT);
                glUniform4fv(uniformLightDiffuse, 1, Display::LIGHT_DIFFUSE);
                glUniform4fv(uniformLightSpecular, 1, Display::LIGHT_SPECULAR);
                glUniform3fv(uniformLightPosition, 1, Display::LIGHT_POSITION);
             
                // Prepare the vertex buffer.
                glEnableVertexAttribArray(0);
                glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeVertices);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

                glEnableVertexAttribArray(1);
                glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeColors);
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

                glEnableVertexAttribArray(2);
                glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeNormals);
                glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);

                // Draw!
                glBindBuffer(GL_ARRAY_BUFFER, this->vboCubeVertices);
                glDrawArrays(GL_TRIANGLES, 0,
                        sizeof(CUBE_VERTICES) / sizeof(GLfloat) / 3);

                // Done with drawing, clean up.
                glDisableVertexAttribArray(1);
                glDisableVertexAttribArray(0);
            }
        }
    }
}

}

/* vim: set ts=4 sw=4 et: */
