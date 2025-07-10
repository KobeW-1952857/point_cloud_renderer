#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <set>
#include <sstream>
#include <algorithm>
#include "octree_builder.cuh"
#include "octree_kernel.cuh"
#include "../util/tree_tools.h"


OctreeBuilderCuda::OctreeBuilderCuda(const TreeConfig& config)
    : config {config}
{
}

void OctreeBuilderCuda::Initialize(const int capacity)
{
    cudaMalloc((void**)&pointsExch, capacity * sizeof(glm::dvec3));
    // GET_CUDA_ERROR("cudaMalloc() pointsExch");
    // tree:
    const int maxNodes = config.GetNodesCount();
    cudaMallocManaged((void**)&tree, maxNodes * sizeof(Octree));
    cudaDeviceSynchronize();
    // GET_CUDA_ERROR("cudaMallocManaged() tree");
}

void OctreeBuilderCuda::Build(glm::dvec3* points_on_device, const int size)
{
    Reset();

    tree->id = 0;
    tree->bounds.min = config.origin;
    tree->bounds.max = config.origin + config.size;
    tree->startId = 0;
    tree->endId = size;

    std::cout << "Build()" << std::endl;

    auto satrtTime = std::chrono::high_resolution_clock::now();
    const int warpsPerBlock = config.threadsPerBlock / 32;

    OctreeKernel<<<1, config.threadsPerBlock, warpsPerBlock * 8 * sizeof(int)>>>(
        points_on_device, pointsExch, tree, 
        0, config.maxDepth, config.minPointsToDivide
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "Kernel() duration: " << 
        (std::chrono::duration_cast<std::chrono::milliseconds>(endTime - satrtTime).count()) << std::endl;

    // CHECK TREE:
    std::cout << "CHECK tree() ";
    int totalCount = 0;

    Octree* tree2 = tree;

    for (int depth = 0; depth < config.maxDepth; ++depth)
    {
        const auto leafs = GetNodeByDepth<3>(depth);
        for (int leaf = 0; leaf < leafs; ++leaf)
        {
            const Octree* const subTree = &tree2[leaf];

            if ((subTree->PointsCount() < config.minPointsToDivide || 
                depth == config.maxDepth - 1) && subTree->PointsCount() > 0)
            {
                totalCount += subTree->PointsCount();
            }
        }

        tree2 += leafs;
    }

    std::cout << "total points: " << totalCount << " / " << size << "\n";

    if (totalCount != size)
    {
        throw "Invalid tree: totalCount != size\n";
    }
}

void OctreeBuilderCuda::Reset()
{
    std::cout << "Reset()" << std::endl;
    const int maxNodes = config.GetNodesCount();
    
    for (int i = 0; i < maxNodes; ++i)
    {
        tree[i].id = 0;
        tree[i].bounds.min = {0.0f, 0.0f, 0.0f};
        tree[i].bounds.max = {0.0f, 0.0f, 0.0f};
        tree[i].startId = 0;
        tree[i].endId = 0;
    }
}

OctreeBuilderCuda::~OctreeBuilderCuda()
{
    cudaFree(pointsExch);
    cudaFree(tree);
}