#pragma once
#include <glm/glm.hpp>
#include "../util/cuda_util.cuh"
#include "../util/types.cuh"
#include "lod_util.cuh"


__global__ void markVoxelsMinIdx(
    const glm::dvec3* positions,
    const bool* writtenFlags,
    int numPoints,
    glm::dvec3 aabbMin,
    float spacing,
    int3 gridRes,
    uint32_t* voxelOwners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints || writtenFlags[idx]) return;
    
    glm::dvec3 pos = positions[idx];
    glm::ivec3 voxelCoord = glm::floor((pos - aabbMin) / spacing);

    if (voxelCoord.x < 0 || voxelCoord.y < 0 || voxelCoord.z < 0 ||
        voxelCoord.x >= gridRes.x || voxelCoord.y >= gridRes.y || voxelCoord.z >= gridRes.z)
        return;

    size_t flatIdx = voxelCoord.x +
                  voxelCoord.y * gridRes.x +
                  voxelCoord.z * gridRes.x * gridRes.y;

    atomicMin(&voxelOwners[flatIdx], idx);
}


__global__ void sweepWinners(
    const glm::dvec3* positions,
    const glm::ucvec3* colors,
    const uint32_t* voxelOwners,
    bool* writtenFlags,
    int numVoxels,
    CLODPoints output,
    unsigned int* outputCounter,
    uint8_t level
) {
    int voxelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxelIdx >= numVoxels) return;

    uint32_t winnerIdx = voxelOwners[voxelIdx];
    if (winnerIdx == 0xFFFFFFFF) return;

    // TODO: race condition
    if (!writtenFlags[winnerIdx]) {
        writtenFlags[winnerIdx] = true;
        int outIdx = atomicAdd(outputCounter, 1);
        output.positions[outIdx] = positions[winnerIdx];
        output.cols[outIdx] = glm::ucvec4(colors[winnerIdx], level);
    }
}


__global__ void emitRemainingPoints(
    const glm::dvec3* positions,
    const glm::ucvec3* colors,
    bool* writtenFlags,
    int numPoints,
    CLODPoints output,
    unsigned int* outputCounter,
    uint8_t level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    if (!writtenFlags[idx]) {
        writtenFlags[idx] = true;
        int outIdx = atomicAdd(outputCounter, 1);
        output.positions[outIdx] = positions[idx];
        output.cols[outIdx] = glm::ucvec4(colors[idx], level);        
    }
}


void buildCLODLevels_denseGrid(
    glm::dvec3* d_positions,
    glm::ucvec3* d_colors,
    int numPoints,
    glm::dvec3 aabbMin,
    glm::dvec3 aabbMax,
    float rootSpacing,
    int numLevels,
    CLODPoints output,
    unsigned int* outputCounter
) {
    const int blkPoints = optimalBlockSize<markVoxelsMinIdx>(); 
    const int blkVoxels = optimalBlockSize<sweepWinners>();
    dim3 blocksPoints((numPoints + blkPoints - 1) / blkPoints);

    bool* d_writtenFlags;
    cudaMalloc(&d_writtenFlags, sizeof(bool) * numPoints);
    cudaMemset(d_writtenFlags, 0, sizeof(bool) * numPoints);

    for (int level = 0; level < numLevels; ++level) {
        float spacing = rootSpacing / powf(2.f, level);

        if (level == numLevels - 1) {
            // coarsest level accepts all remaining points
            emitRemainingPoints<<<blocksPoints, blkPoints>>>(
                d_positions, d_colors, d_writtenFlags,
                numPoints, output, outputCounter, (uint8_t)level);
            cudaDeviceSynchronize();
            break;
        }

        // grid size
        int3 gridRes = make_int3(
            ceil((aabbMax.x - aabbMin.x) / spacing),
            ceil((aabbMax.y - aabbMin.y) / spacing),
            ceil((aabbMax.z - aabbMin.z) / spacing));
        int numVoxels = gridRes.x * gridRes.y * gridRes.z;
        dim3 blocksVoxels((numVoxels + blkVoxels - 1) / blkVoxels);

        // voxel ownership grid
        uint32_t* d_voxelOwners;
        cudaMalloc(&d_voxelOwners, sizeof(uint32_t) * numVoxels);
        cudaMemset(d_voxelOwners, 0xFF, sizeof(uint32_t) * numVoxels);

        // mark winners
        markVoxelsMinIdx<<<blocksPoints, blkPoints>>>(
            d_positions, d_writtenFlags, numPoints,
            aabbMin, spacing, gridRes, d_voxelOwners);
        
        cudaDeviceSynchronize();

        // sweep winners
        sweepWinners<<<blocksVoxels, blkVoxels>>>(
            d_positions, d_colors, d_voxelOwners, d_writtenFlags,
            numVoxels, output, outputCounter, (uint8_t)level);

        cudaDeviceSynchronize();
        cudaFree(d_voxelOwners);
    }

    cudaFree(d_writtenFlags);
}


// REDUCE
__device__ float pseudoRandom(uint32_t seed) {
    seed ^= 2747636419u;
    seed *= 2654435769u;
    seed ^= seed >> 16;
    seed *= 2654435769u;
    seed ^= seed >> 16;
    seed *= 2654435769u;
    return (seed & 0xFFFFFF) / float(0x1000000); // [0,1)
}


__device__ bool shouldKeepPoint(
    uint8_t level,
    int pointIdx,
    glm::dvec3 pointPos,
    glm::dvec3 cameraPos,
    float rootSpacing,
    float clodFactor
) {
    float distance = glm::length(pointPos - cameraPos);
    float targetSpacing = (distance * clodFactor) / 1000.f;

    float jitter = pseudoRandom(pointIdx);
    float jitteredLevel = (float)level + jitter;

    float pointSpacing = rootSpacing / powf(2.0f, jitteredLevel);

    return pointSpacing >= targetSpacing;
}


// __global__  void filterPointsKernel(CLODPoints points, glm::vec3 camPos, 
//     float rootSpacing, float clodFactor, unsigned int pointAmt, glm::dvec3* pos_buffer, 
//     glm::ucvec3* col_buffer, unsigned int* reducedAmt) {
    
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= pointAmt) return;

//     if (shouldKeepPoint(points.cols[idx].w, idx, points.positions[idx], camPos, rootSpacing, clodFactor)) {
//         int outIdx = atomicAdd(reducedAmt, 1);
//         pos_buffer[outIdx] = points.positions[idx];
//         col_buffer[outIdx] = points.cols[idx];
//     } // else discard point
// }

__global__  void filterPointsKernel(CLODPoints points, glm::vec3 camPos, 
    float rootSpacing, float clodFactor, unsigned int pointAmt, glm::dvec3* pos_buffer, 
    glm::ucvec3* col_buffer, unsigned int* reducedAmt) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pointAmt) return;

    if (shouldKeepPoint(points.cols[idx].w, idx, points.positions[idx], camPos, rootSpacing, clodFactor)) {
        int outIdx = atomicAdd(reducedAmt, 1);
        pos_buffer[outIdx] = points.positions[idx];
        unsigned int level = static_cast<unsigned int>(points.cols[idx].w);
        unsigned uint8_t intensity = static_cast<unsigned char>(min(level * 20, 255u)); // scale factor for visualization
        col_buffer[outIdx] = glm::ucvec3(255u, 0u, intensity);
    }
}
