#pragma once
#include <glm/glm.hpp>


struct TreeConfig  {
    glm::vec3 origin;
    glm::vec3 size;
    int maxDepth;
    
    int threadsPerBlock;
    int minPointsToDivide;

    inline int GetNodesCount() const
    {
        int maxNodes = 0;
        for (int i = 0; i < maxDepth; ++i)
        {
            maxNodes += std::pow(8, i);
        }

        return maxNodes;
    }
};