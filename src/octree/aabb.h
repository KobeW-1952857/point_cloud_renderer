#pragma once



struct AABB
{
    glm::dvec3 min;
    glm::dvec3 max;

    __host__ __device__ bool Check(const glm::dvec3& point) const
    {
        return point >= min && point < max; 
    }

    __host__ __device__ glm::dvec3 GetCenter() const
    {
        return (min + max) * 0.5f;
    }

    __host__ __device__ glm::dvec3 Size() const
    {
        return max - min;
    }
};