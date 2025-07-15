#include "glm/fwd.hpp"
#include "glm/glm.hpp"
#include "happly.h"
#include "render.cuh"
#include <cstdio>

__device__ double atomicMin(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  double old_double;

  do {
    old_double = __longlong_as_double(old);
    if (old_double <= val)
      return old_double;
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}
__device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  double old_double;

  do {
    old_double = __longlong_as_double(old);
    if (old_double >= val)
      return old_double;
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Device function to atomically exchange an unsigned char
// 'address' points to an unsigned char within a larger integer (e.g., unsigned
// int) 'val' is the new value to set for the unsigned char
__device__ unsigned char atomicExch(unsigned char *address, unsigned char val) {
  // Determine the containing 32-bit word address
  // This relies on the assumption that the unsigned char is part of an aligned
  // 32-bit word. If not, this approach might lead to issues or be less
  // efficient.
  unsigned int *word_address =
      (unsigned int *)((size_t)address & ~0x3); // Align to 4-byte boundary

  // Calculate the byte offset within the word
  int byte_offset = (size_t)address % 4; // 0, 1, 2, or 3

  unsigned int old_word;
  unsigned int new_word;
  unsigned char old_char_value;

  do {
    old_word = *word_address; // Atomically read the entire 32-bit word

    // Extract the old unsigned char value
    old_char_value = (unsigned char)((old_word >> (byte_offset * 8)) & 0xFF);

    // Construct the new 32-bit word with the updated unsigned char
    new_word = old_word;
    new_word &=
        ~((unsigned int)0xFF << (byte_offset * 8));       // Clear the old byte
    new_word |= ((unsigned int)val << (byte_offset * 8)); // Set the new byte

  } while (atomicCAS(word_address, old_word, new_word) !=
           old_word); // Attempt to swap

  return old_char_value;
}

// Kernel to fill a double array with a specific value
__global__ void fillDoubleArrayKernel(double *devArray, double value,
                                      int numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    devArray[idx] = value;
  }
}

__global__ void naive(unsigned char *output_data, double *depht_buffer,
                      int width, int height, const glm::mat4 *cam_proj,
                      const glm::dvec3 *points, const glm::ucvec3 *colors,
                      size_t n_points) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= n_points) return;

  glm::dvec3 point = points[id];
  glm::ucvec3 color = colors[id];

  // Convert input double3 to homogeneous double4
  glm::vec4 pointHomogeneous(point, 1.0);

  // Perform matrix-vector multiplication
  glm::vec4 result = *cam_proj * pointHomogeneous;

  int u = result.x / result.z;
  int v = result.y / result.z;
  if (u < width && u >= 0 && v < height && v >= 0 && result.z >= 0.0f) {

    double old = atomicMin(&depht_buffer[v * width + u], result.z);
    if (result.z < old) {
      atomicExch(&output_data[(v * width + u) * 3 + 0], color.r);
      atomicExch(&output_data[(v * width + u) * 3 + 1], color.g);
      atomicExch(&output_data[(v * width + u) * 3 + 2], color.b);
    }
  }
}
