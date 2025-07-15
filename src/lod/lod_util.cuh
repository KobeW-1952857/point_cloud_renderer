#pragma once
#include <glm/glm.hpp>

// typedefs 
struct CLODPoints {
    glm::ucvec4* cols;
    glm::dvec3* positions;
};

// constants
const float CLOD_FACTOR = 1.5;
const int LEVELS_AMT = 6;
const float ROOT_SPACING = 0.4;
