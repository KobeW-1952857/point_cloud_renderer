#pragma once
#include <glm/glm.hpp>
#include "../util/cuda_util.cuh"
#include "../util/types.cuh"
#include "lod_util.cuh"


__global__ void markVoxelsMinIdx(
    const glm::dvec3* positions,
    const unsigned int* writtenFlags,
    int numPoints,
    glm::dvec3 aabbMin,
    float spacing,
    int3 gridRes,
    uint32_t* voxelOwners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints || writtenFlags[idx] == 1) return;
    
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
    unsigned int* writtenFlags,
    int numVoxels,
    CLODPoints output,
    unsigned int* outputCounter,
    const uint8_t level
) {
    int voxelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxelIdx >= numVoxels) return;

    uint32_t winnerIdx = voxelOwners[voxelIdx];
    if (winnerIdx == 0xFFFFFFFF) return;

    if (atomicCAS(&writtenFlags[winnerIdx], 0, 1) == 0) {
        writtenFlags[winnerIdx] = 1;
        int outIdx = atomicAdd(outputCounter, 1);
        output.positions[outIdx] = positions[winnerIdx];
        output.cols[outIdx] = glm::ucvec4(colors[winnerIdx], level);
    }
}


__global__ void emitRemainingPoints(
    const glm::dvec3* positions,
    const glm::ucvec3* colors,
    unsigned int* writtenFlags,
    int numPoints,
    CLODPoints output,
    unsigned int* outputCounter,
    uint8_t level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    if (atomicCAS(&writtenFlags[idx], 0, 1) == 0) {
        writtenFlags[idx] = 1;
        int outIdx = atomicAdd(outputCounter, 1);
        output.positions[outIdx] = positions[idx];
        output.cols[outIdx] = glm::ucvec4(colors[idx], level);        
    }
}


void buildCLODLevels(
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
    int blkPoints, minGridSizePoints;
    cudaOccupancyMaxPotentialBlockSize(&minGridSizePoints, &blkPoints, markVoxelsMinIdx);
    int blkVoxels, minGridSizeVoxels;
    cudaOccupancyMaxPotentialBlockSize(&minGridSizeVoxels, &blkVoxels, sweepWinners);
    dim3 blocksPoints((numPoints + blkPoints - 1) / blkPoints);

    unsigned int* d_writtenFlags;
    cudaMalloc(&d_writtenFlags, sizeof(unsigned int) * numPoints);
    cudaMemset(d_writtenFlags, 0, sizeof(unsigned int) * numPoints);

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
    const uint8_t level,
    const int pointIdx,
    const glm::dvec3 pointPos,
    const glm::dvec3 cameraPos,
    const float rootSpacing,
    const float clodFactor
) {
    glm::dvec3 diff = pointPos - cameraPos;
    float distSq = glm::dot(diff, diff);
    float targetSpacingSq = ((clodFactor / 1000.f) * (clodFactor / 1000.f)) * distSq;

    float jitter = pseudoRandom(pointIdx);
    float jitteredLevel = (float)level + jitter;

    float pointSpacing = rootSpacing / powf(2.0f, jitteredLevel);
    float pointSpacingSq = pointSpacing * pointSpacing;

    return pointSpacingSq >= targetSpacingSq;

}


__global__ void filterPointsKernel(
        const CLODPoints points,
        const float rootSpacing, const float clodFactor, const unsigned int pointAmt, 
        glm::dvec3* pos_buffer, glm::ucvec3* col_buffer, 
        unsigned int* reducedAmt, const glm::dvec3 camPos
    ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pointAmt) return;

    if (shouldKeepPoint(points.cols[idx].w, idx, points.positions[idx], camPos, rootSpacing, clodFactor)) {
        int outIdx = atomicAdd(reducedAmt, 1);
        pos_buffer[outIdx] = points.positions[idx];
        col_buffer[outIdx] = points.cols[idx];
    } // else discard point
}

// __global__ void filterPointsKernel(
//         CLODPoints points,
//         float rootSpacing, float clodFactor, unsigned int pointAmt, 
//         glm::dvec3* pos_buffer, glm::ucvec3* col_buffer, 
//         unsigned int* reducedAmt, glm::dvec3 camPos
//     ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= pointAmt) return;
    
//     glm::dvec3 pointPos = points.positions[idx];

//     if (shouldKeepPoint(points.cols[idx].w, idx, pointPos, camPos, rootSpacing, clodFactor)) {
//         int outIdx = atomicAdd(reducedAmt, 1);
//         pos_buffer[outIdx] = pointPos;

//         uint8_t level = static_cast<uint8_t>(points.cols[idx].w);
//         glm::ucvec3 color;
//         switch (level) {
//             case 0: color = glm::ucvec3(255, 0, 0); break;      // Red
//             case 1: color = glm::ucvec3(0, 255, 0); break;      // Green
//             case 2: color = glm::ucvec3(0, 0, 255); break;      // Blue
//             case 3: color = glm::ucvec3(255, 255, 0); break;    // Yellow
//             case 4: color = glm::ucvec3(255, 0, 255); break;    // Magenta
//             // case 5: color = glm::ucvec3(0, 255, 255); break;    // Cyan
//             case 5: color = glm::ucvec3(0, 0, 0); break;  
//         }
//         col_buffer[outIdx] = color;
//     }
// }
