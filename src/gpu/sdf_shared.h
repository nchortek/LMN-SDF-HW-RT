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

// How many voxels span the scene bbox per axis
GPU_CONST uint VOXELS_PER_SCENE_AXIS = 512;

// How many pruning cells span the scene bbox per axis 
GPU_CONST uint PRUNING_CELLS_PER_SCENE_AXIS = 256;



// One brick's axis-aligned bounding box in world space.
// This must match VkAabbPositionsKHR.
struct Aabb
{
    vec3 minValues;
    vec3 maxValues;
};

// Baking inputs
struct BrickBakeParams
{
    // Starting address of our buffer of SDF primitives
    uint64_t primsAddr;

    // Starting address of our buffer of binary SDF operators
    uint64_t binaryOpsAddr;

    // Starting address of our buffer of flattened CSG tree nodes
    uint64_t nodesAddr;

    // Starting address of our buffer of surviving node indices
    uint64_t activeNodesAddr;

    // Per pruning cell: start index of its slice in the active node pool
    uint64_t cellOffsetsAddr;

    // Per pruning cell: length of its slice in the active node pool
    uint64_t cellNumActiveAddr;

    // Per pruning cell: error bound from pruning
    uint64_t cellErrorsAddr;

    // Per brick cell: active node count. Used to skip empty bricks
    uint64_t brickCellsNumActiveAddr;

    // Maximum number of bricks
    int maxBricks;

    // Total number of nodes in CSG tree
    int totalNumNodes;
    
};

// Device buffer addresses for the OUTPUT SDF brick buffers
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

    // Brick baking input params buffer
    uint64_t brickBakeParamsAddr;

    // World-space minimum corner of the cubic domain the bricks cover
    vec3 domainMin;

    // World units per voxel
    float voxelSize;

    // Our band size, used for dequantization:
    // distance = (sample / 127) * band
    float band;

    // Explicit padding at the end (uint64_t members must be 8-byte aligned!)
    // This padding brings up to 64 bytes, which is a multiple of 8.
    float _pad0;
};

struct BakePushConstants
{
    // The device buffer address point to our single instance
    // of the SdfSceneAddresses struct.
    uint64_t sdfSceneAddr;
};


GPU_NAMESPACE_END

#ifdef __cplusplus

static_assert(sizeof(gpu::Aabb) == 24, "Our Aabb struct must match the size of VkAabbPositionsKHR.");
static_assert(gpu::BRICK_STRIDE_WORDS * 4 >= gpu::BRICK_SAMPLES, "Each brick's samples must fit within its word count.");
static_assert(sizeof(gpu::BrickBakeParams) == 8*8 + 2*4, "Size of BrickBakeParams must match expected value.");
static_assert(sizeof(gpu::SdfSceneAddresses) == 5*8 + 12 + 2*4 + 4, "Size of SdfSceneAddresses must match expected value.");
static_assert(sizeof(gpu::BakePushConstants) == 8, "Size of BakePushConstants must match expected value.");

#endif

#endif // GPU_SDF_SHARED_H
