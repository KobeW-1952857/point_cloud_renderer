#pragma once

#include "../util/cuda_operators.cuh"
#include "aabb.h"

struct Octree
{
    int id;
    int startId;
    int endId;
    AABB bounds;

    __host__ __device__ bool Check(const glm::dvec3& point) const
    {
        return bounds.Check(point);
    }

    __host__ __device__ int PointsCount() const
    {
        return endId - startId;
    }
};