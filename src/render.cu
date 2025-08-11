#include <sys/types.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>

#include "glm/fwd.hpp"
#include "render.cuh"
#include "types.h"

__device__ double atomicMin(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  double old_double;

  do {
    old_double = __longlong_as_double(old);
    if (old_double <= val) return old_double;
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ uint64_t atomicMin(uint64_t *address, uint64_t val) {
  uint64_t old;
  do {
    old = *address;
    if (val >= old) return old;
  } while (atomicCAS((unsigned long long *)address, old, val) != old);
  return old;
}

__device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  double old_double;

  do {
    old_double = __longlong_as_double(old);
    if (old_double >= val) return old_double;
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
  unsigned int *word_address = (unsigned int *)((size_t)address & ~0x3);  // Align to 4-byte boundary

  // Calculate the byte offset within the word
  int byte_offset = (size_t)address % 4;  // 0, 1, 2, or 3

  unsigned int old_word;
  unsigned int new_word;
  unsigned char old_char_value;

  do {
    old_word = *word_address;  // Atomically read the entire 32-bit word

    // Extract the old unsigned char value
    old_char_value = (unsigned char)((old_word >> (byte_offset * 8)) & 0xFF);

    // Construct the new 32-bit word with the updated unsigned char
    new_word = old_word;
    new_word &= ~((unsigned int)0xFF << (byte_offset * 8));  // Clear the old byte
    new_word |= ((unsigned int)val << (byte_offset * 8));    // Set the new byte

  } while (atomicCAS(word_address, old_word, new_word) != old_word);  // Attempt to swap

  return old_char_value;
}

__device__ __forceinline__ uint64_t packUCVec3(const glm::ucvec3 &v) {
  return (static_cast<uint64_t>(v.r) << 0) | (static_cast<uint64_t>(v.g) << 8) | (static_cast<uint64_t>(v.b) << 16);
}

__device__ glm::ucvec3 unpackUCVec3(const uint64_t v) {
  return glm::ucvec3(static_cast<unsigned char>((v >> 0) & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
                     static_cast<unsigned char>((v >> 16) & 0xFF));
}

__device__ unsigned long long warpReduceMinULL(unsigned int mask, unsigned long long val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    unsigned long long tmpVal = __shfl_down_sync(mask, val, offset);
    if (val > tmpVal) val = tmpVal;
  }
  return val;
}

// Kernel to fill a double array with a specific value
__global__ void fillDoubleArrayKernel(double *buffer, double value, int numElements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    buffer[idx] = value;
  }
}
__global__ void fillBuffer(uint64_t *buffer, uint64_t value, int numElements) {
  int block_id = blockIdx.x +                         // apartment number on this floor (points across)
                 blockIdx.y * gridDim.x +             // floor number in this building (rows high)
                 blockIdx.z * gridDim.x * gridDim.y;  // building number in this city (panes deep)

  int block_offset = block_id *                             // times our apartment number
                     blockDim.x * blockDim.y * blockDim.z;  // total threads per block (people per apartment)

  int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  int idx = block_offset + thread_offset;  // global person id in the entire apartment complex

  if (idx < numElements) {
    // printf("FILL BUFFER ID: %u, Data: 0x%lX\n", idx, (unsigned long)value);
    buffer[idx] = value;
  }
}

__global__ void naive(unsigned char *output_data, double *depht_buffer, int width, int height,
                      const glm::mat4 *cam_proj, const glm::dvec3 *points, const glm::ucvec3 *colors, size_t n_points) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id > n_points) return;

  glm::dvec3 point = points[id];
  glm::ucvec3 color = colors[id];

  // Convert input double3 to homogeneous double4
  glm::vec4 pointHomogeneous;
  pointHomogeneous.x = point.x;
  pointHomogeneous.y = point.y;
  pointHomogeneous.z = point.z;
  pointHomogeneous.w = 1.0;  // W component for a point

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

__device__ float4 matmul(const float m[16], const float4 &v) {
  float4 result;
  result.x = m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
  result.y = m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7];
  result.z = m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11];
  result.w = m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15];
  return result;
}

__device__ __forceinline__ unsigned int packuchar3(const uchar3 &v) {
  return (static_cast<unsigned int>(v.x) << 0) | (static_cast<unsigned int>(v.y) << 8) |
         (static_cast<unsigned int>(v.z) << 16);
}

__device__ uchar3 unpackuchar3(const unsigned int v) {
  return {static_cast<unsigned char>((v >> 0) & 0xFF), static_cast<unsigned char>((v >> 8) & 0xFF),
          static_cast<unsigned char>((v >> 16) & 0xFF)};
}

__global__ void vertexOrderOptimization(uint64_t *output_data, const float4 *vertices, const uchar4 *colors,
                                        const float cam_proj[16], uint2 image_size, size_t n_points) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id >= n_points) return;

  float4 point = __ldg(&vertices[global_id]);
  uchar4 color = __ldg(&colors[global_id]);

  // Perform matrix-vector multiplication
  float4 result = matmul(cam_proj, point);

  int u = rintf(__fdividef(result.x, result.z));
  int v = rintf(__fdividef(result.y, result.z));
  unsigned int pixID = v * image_size.x + u;

  if (result.z < 0.0f || u < 0 || u >= image_size.x || v < 0 || v >= image_size.y) return;

  unsigned int depth = __float_as_uint(result.z);

  unsigned int same_pixel_mask = __match_any_sync(__activemask(), pixID);
  unsigned int min_depth = __reduce_min_sync(same_pixel_mask, depth);

  bool is_closest_thread = (depth == min_depth);

  if (is_closest_thread) {
    unsigned long long data = ((unsigned long long)depth << 32) | packuchar3({color.x, color.y, color.z});
    atomicMin(&output_data[pixID], data);
    return;
  }
}

__device__ int warpReduceMax(int val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__global__ void findBlockMaxKernel(uint64_t *d_in, uint32_t *d_block_maxes, size_t size) {
  // Shared memory for block-level reduction
  extern __shared__ uint32_t s_data[];

  // Global thread index
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Load one element into shared memory
  s_data[threadIdx.x] = (tid < size) && d_in[tid] != 0xFFFFFFFFFF000000 ? d_in[tid] >> 32 : 0;
  __syncthreads();

  // Block-level reduction using shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_data[threadIdx.x] = max(s_data[threadIdx.x], s_data[threadIdx.x + s]);
    }
    __syncthreads();
  }

  // Write the block's max to global memory
  if (threadIdx.x == 0) {
    d_block_maxes[blockIdx.x] = s_data[0];
  }
}

__global__ void findAbsoluteMaxKernel(uint32_t *d_block_maxes, uint32_t *d_absolute_max, size_t num_blocks) {
  // The first thread finds the max of the block maxes
  int tid = threadIdx.x;
  uint32_t my_max = 0;

  for (int i = tid; i < num_blocks; i += blockDim.x) {
    my_max = max(my_max, d_block_maxes[i]);
  }

  // Use warp reduction to find the final max
  my_max = warpReduceMax(my_max);

  if (threadIdx.x == 0) {
    d_absolute_max[0] = my_max;
  }
}

__global__ void resolve(glm::ucvec3 *image, glm::ucvec3 *depth, const uint32_t *max_depth, const uint64_t *data,
                        size_t count) {
  int block_id = blockIdx.x +                         // apartment number on this floor (points across)
                 blockIdx.y * gridDim.x +             // floor number in this building (rows high)
                 blockIdx.z * gridDim.x * gridDim.y;  // building number in this city (panes deep)

  int block_offset = block_id *                             // times our apartment number
                     blockDim.x * blockDim.y * blockDim.z;  // total threads per block (people per apartment)

  int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

  int id = block_offset + thread_offset;  // global person id in the entire apartment complex

  if (id > count) return;

  uint64_t val = data[id];
  if (val == 0xFFFFFFFFFF000000) return;
  image[id] = unpackUCVec3(val);

  float depth_val = __uint_as_float((uint)(val >> 32));
  unsigned char col = max(__fdiv_rn(__fmul_rn(depth_val, 255.0f), __uint_as_float(*max_depth)), 255.0f);
  depth[id] = {col, col, col};
  // printf("RESOLVE ID: %u, Data: 0x%lX, Color: (%u, %u, %u)\n", id, (unsigned long)val, image[id].x, image[id].y,
  //        image[id].z);
}