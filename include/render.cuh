#pragma once

#include "glm/fwd.hpp"
#include "happly.h"

__global__ void fillDoubleArrayKernel(double *devArray, const double value,
                                      const int numElements);
// __global__ void naive(unsigned char *output_data, double *depth_buffer,
//                       int width, int height, const glm::mat4 *intr_extr_mat,
//                       const glm::dvec3 *points, const glm::ucvec3 *colors,
//                       size_t n_points);

__global__ void naive(unsigned char *output_data, double *depth_buffer,
            int width, int height, const glm::mat4 *intr_extr_mat,
            const glm::dvec3 *points, const glm::ucvec3 *colors,
            const unsigned int* n_points);