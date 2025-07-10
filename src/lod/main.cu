


__global__ void markVoxelsMinIdx(
    const float3* positions,
    const bool* writtenFlags,
    int numPoints,
    float3 aabbMin,
    float spacing,
    int3 gridRes,
    uint32_t* voxelOwners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints || writtenFlags[idx]) return;

    float3 pos = positions[idx];
    int3 voxelCoord = make_int3(floorf((pos - aabbMin) / spacing));

    if (voxelCoord.x < 0 || voxelCoord.y < 0 || voxelCoord.z < 0 ||
        voxelCoord.x >= gridRes.x || voxelCoord.y >= gridRes.y || voxelCoord.z >= gridRes.z)
        return;

    int flatIdx = voxelCoord.x +
                  voxelCoord.y * gridRes.x +
                  voxelCoord.z * gridRes.x * gridRes.y;

    atomicMin(&voxelOwners[flatIdx], idx);
}


__global__ void sweepWinners(
    const float3* positions,
    const uchar3* colors,
    const uint32_t* voxelOwners,
    bool* writtenFlags,
    int numVoxels,
    CLODPoint* output,
    int* outputCounter,
    uint8_t level
) {
    int voxelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxelIdx >= numVoxels) return;

    uint32_t winnerIdx = voxelOwners[voxelIdx];
    if (winnerIdx == 0xFFFFFFFF) return;

    if (!writtenFlags[winnerIdx]) {
        writtenFlags[winnerIdx] = true;
        int outIdx = atomicAdd(outputCounter, 1);
        output[outIdx] = {positions[winnerIdx], colors[winnerIdx], level};
    }
}


__global__ void emitRemainingPoints(
    const float3* positions,
    const uchar3* colors,
    bool* writtenFlags,
    int numPoints,
    CLODPoint* output,
    int* outputCounter,
    uint8_t level
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    if (!writtenFlags[idx]) {
        int outIdx = atomicAdd(outputCounter, 1);
        output[outIdx] = {positions[idx], colors[idx], level};
        writtenFlags[idx] = true;
    }
}


void buildCLODLevels_denseGrid(
    float3* d_positions,
    uchar3* d_colors,
    int numPoints,
    float3 aabbMin,
    float3 aabbMax,
    float rootSpacing,
    int numLevels,
    CLODPoint* d_output,
    int* d_outputCounter
) {
    const int threads = 256;
    const dim3 blocksPoints = dim3((numPoints + threads - 1) / threads);

    bool* d_writtenFlags;
    cudaMalloc(&d_writtenFlags, sizeof(bool) * numPoints);
    cudaMemset(d_writtenFlags, 0, sizeof(bool) * numPoints);

    for (int level = 0; level < numLevels; ++level) {
        float spacing = rootSpacing / powf(2.0f, level);

        if (level == numLevels - 1) {
            // coarsest level: accept all remaining points
            emitRemainingPoints<<<blocksPoints, threads>>>(
                d_positions, d_colors, d_writtenFlags,
                numPoints, d_output, d_outputCounter, (uint8_t)level);
            cudaDeviceSynchronize();
            break;
        }

        // grid size
        int3 gridRes = make_int3(
            ceil((aabbMax.x - aabbMin.x) / spacing),
            ceil((aabbMax.y - aabbMin.y) / spacing),
            ceil((aabbMax.z - aabbMin.z) / spacing));
        int numVoxels = gridRes.x * gridRes.y * gridRes.z;
        dim3 blocksVoxels((numVoxels + threads - 1) / threads);

        // voxel ownership grid
        uint32_t* d_voxelOwners;
        cudaMalloc(&d_voxelOwners, sizeof(uint32_t) * numVoxels);
        cudaMemset(d_voxelOwners, 0xFF, sizeof(uint32_t) * numVoxels);

        // mark winners
        markVoxelsMinIdx<<<blocksPoints, threads>>>(
            d_positions, d_writtenFlags, numPoints,
            aabbMin, spacing, gridRes, d_voxelOwners);

        // sweep winners
        sweepWinners<<<blocksVoxels, threads>>>(
            d_positions, d_colors, d_voxelOwners, d_writtenFlags,
            numVoxels, d_output, d_outputCounter, (uint8_t)level);

        cudaFree(d_voxelOwners);
        cudaDeviceSynchronize();
    }

    cudaFree(d_writtenFlags);
}
