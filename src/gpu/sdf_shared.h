// SDF brick struct layouts and constants shared by C++ and GLSL.
#ifndef GPU_SDF_SHARED_H
#define GPU_SDF_SHARED_H

#include "shared.h"

GPU_NAMESPACE_BEGIN

// Capacity of the brick buffers. Unused AABB slots are set to NaN to signify inactive primitives.
GPU_CONST uint MAX_BRICKS = 131072;

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
};

GPU_NAMESPACE_END

#ifdef __cplusplus
static_assert(sizeof(gpu::Aabb) == 24, "Our Aabb struct must match the size of VkAabbPositionsKHR.");
#endif

#endif // GPU_SDF_SHARED_H
