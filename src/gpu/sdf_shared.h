// SDF brick struct layouts and constants shared by C++ and GLSL.
#ifndef GPU_SDF_SHARED_H
#define GPU_SDF_SHARED_H

#include "shared.h"

GPU_NAMESPACE_BEGIN

// Capacity of the brick buffers. Unused AABB slots are set to NaN to signify inactive primitives.
GPU_CONST uint MAX_BRICKS = 131072;

// The number of voxels along each axis within a brick
GPU_CONST uint BRICK_VOXELS = 8;

// The number of SDF samples per brick (9x9x9)
GPU_CONST uint BRICK_SAMPLES = 729;

// The number of 32-bit words used to store a given brick's samples
// This works out to 736 bytes, which ensures that no word ever stores
// data from more than one brick.
GPU_CONST uint BRICK_STRIDE_WORDS = 184;

// One brick's axis-aligned bounding box in world space.
// This must match VkAabbPositionsKHR.
struct Aabb
{
    vec3 minValues;
    vec3 maxValues;
};

// Device buffer addresses for the SDF brick buffers
struct SdfSceneAddresses
{
    // Starting address of our buffer of brick Aabb structs
    uint64_t aabbsAddr;

    // Starting address of our buffer of uint brick Morton indices
    uint64_t mortonsAddr;

    // Starting address of our buffer of packed samples (4 samples per 32-bit word)
    uint64_t samplesAddr;

    // Address of a single uint holding the number of non-empty bricks
    uint64_t brickCountAddr;

    // World-space minimum corner of the cubic domain the bricks cover
    vec3 domainMin;

    // World units per voxel
    float voxelSize;

    // Our band size, used for dequantization:
    // distance = (sample / 127) * band
    float band;
};

GPU_NAMESPACE_END

#ifdef __cplusplus
static_assert(sizeof(gpu::Aabb) == 24, "Our Aabb struct must match the size of VkAabbPositionsKHR.");
static_assert(gpu::BRICK_STRIDE_WORDS * 4 >= gpu::BRICK_SAMPLES, "Each brick's samples must fit within its word count.");
#endif

#endif // GPU_SDF_SHARED_H
