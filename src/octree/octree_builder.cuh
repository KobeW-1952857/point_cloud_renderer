#pragma once

#include "tree_config.h"
#include "octree.h"

class OctreeBuilderCuda
{
public:
    OctreeBuilderCuda(const TreeConfig& config);
    ~OctreeBuilderCuda();

public:
    void Initialize(const int capacity);
    void Build(glm::dvec3* point, const int count);
    void Reset();
    const Octree& GetTree() const { return *tree; }

private:
    glm::dvec3* pointsExch;
    TreeConfig config;
    Octree* tree;
};