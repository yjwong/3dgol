#ifndef GAME_OF_LIFE_DISPLAY_H
#define GAME_OF_LIFE_DISPLAY_H

#include <string>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace GameOfLife {

class GameOfLife;
class Display {
private:
    GameOfLife* gameOfLife;
    int sp;

    static GLfloat CUBE_VERTICES[];
    static GLfloat CUBE_COLORS[];
    static GLfloat WIRE_CUBE_VERTICES[];
    static GLfloat WIRE_CUBE_COLORS[];
    static GLfloat AXES_VERTICES[];
    static GLfloat AXES_COLORS[];
    static int WINDOW_WIDTH;
    static int WINDOW_HEIGHT;

    GLuint wireCubeProgram;
    GLuint cubeProgram;

    GLuint vertexArrayObject;
    GLuint vboCubeVertices;
    GLuint vboCubeColors;
    GLuint vboWireCubeVertices;
    GLuint vboWireCubeColors;
    GLuint vboAxesVertices;
    GLuint vboAxesColors;

    glm::mat4 sceneModel;
    glm::mat4 sceneView;
    glm::mat4 sceneProjection;
    glm::vec3 sceneCameraCoord;
    glm::vec3 sceneCameraLookAt;
    glm::vec3 sceneCameraUp;

    bool isRotateInProgress;
    double rotateModeOriginalX;
    double rotateModeOriginalY;
    float rotationX;
    float rotationY;

    bool isAxesVisible;

    double time; /* double time! */
    double timeNextGen;

    static void onGlfwError(int error, const char* desc);
    static void onGlfwFramebufferSize(GLFWwindow* window, int width, int height);
    static void onGlfwKey(GLFWwindow* window, int key, int scancode,
            int action, int mods);
    static void onGlfwMouseButton(GLFWwindow* window, int button, int action,
            int mods);
    static void onGlfwCursorPos(GLFWwindow* window, double xpos, double ypos);
    static void onGlfwScroll(GLFWwindow* window, double xoffset,
            double yoffset);
    
    GLuint createShader(GLenum shaderType, const std::string shaderFile);
    void printShaderInfoLog(GLhandleARB obj);
    void drawWireCube();
    void drawCube();
    void initializeGraphics();
    void cleanUpGraphics();
    
public:
	Display (GameOfLife* gameOfLife, int sp);
};

}

#endif /* GAME_OF_LIFE_DISPLAY_H */

/* vim: set ts=4 sw=4 et: */
