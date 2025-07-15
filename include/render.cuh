#pragma once

#include <cstdint>

#include "glm/fwd.hpp"
#include "types.h"

__global__ void fillDoubleArrayKernel(double *devArray, double value, int numElements);
__global__ void naive(unsigned char *output_data, double *depth_buffer, int width, int height,
                      const glm::mat4 *intr_extr_mat, const glm::dvec3 *points, const glm::ucvec3 *colors,
                      size_t n_points);
__global__ void vertexOrderOptimization(uint64_t *output_data, const float4 *vertices, const uchar4 *colors,
                                        const float cam_proj[16], uint2 image_size, size_t n_points);
__global__ void resolve(glm::ucvec3 *image, const uint64_t *data, size_t count);
__global__ void fillBuffer(uint64_t *devArray, uint64_t value, int numElements);