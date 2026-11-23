#ifndef CSG_EVAL_GLSL
#define CSG_EVAL_GLSL

#include "gpu/sdf_shared.h"
#include "gpu/pruning_constants.h"
#include "pruning/extensions.glsl"
#include "pruning/common.glsl"
#include "pruning/eval.glsl"


layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer BrickBakeParamsBuffer {
    BrickBakeParams params;
};

// Finite-difference step size (in voxels) used for CSG normals
const float kNormalStepVoxels = 0.01;

// Worldspace pos -> voxelspace pos [0, VOXELS_PER_SCENE_AXIS)
vec3 worldToVoxel(vec3 worldPos, SdfSceneAddresses sdf) {
    return (worldPos - sdf.domainMin) / sdf.voxelSize;
}

int csgCellIdx(vec3 worldPos, SdfSceneAddresses sdf) {
    // Worldspace pos -> voxelspace pos -> prunespace pos -> prune_idx
    ivec3 voxelspace_pos = ivec3(floor(worldToVoxel(worldPos, sdf)));
    ivec3 prunespace_pos = clamp(voxelspace_pos >> 1, ivec3(0), ivec3(int(PRUNING_CELLS_PER_SCENE_AXIS) - 1));
    int prune_idx = int(get_cell_idx(prunespace_pos, int(PRUNING_CELLS_PER_SCENE_AXIS)));
    return prune_idx;
}

vec3 computeCsgAlbedo(vec3 worldPos, SdfSceneAddresses sdf) {
    if (sdf.brickBakeParamsAddr == uint64_t(0)) {
        return vec3(-1.f);
    }
    return sdf_color_active(worldPos,
        csgCellIdx(worldPos, sdf),
        BrickBakeParamsBuffer(sdf.brickBakeParamsAddr).params);
}

// Computes a surface normal from the pruned CSG grid by normalizing
// the surface gradient.
//
// Returns whether or not a valid normal was able to be computed.
//
// Adapted from IQ's Tetrahedron Technique for SDF Normals:
// https://iquilezles.org/articles/normalsSDF/
bool computeCsgNormal(vec3 worldPos, SdfSceneAddresses sdf, out vec3 normal) {
    normal = vec3(0.0);

    // If we are loading this scene from a .bricks file, there is not a
    // a CSG to evaluate
    if (sdf.brickBakeParamsAddr == uint64_t(0))
    {
        return false;
    }

    BrickBakeParams params = BrickBakeParamsBuffer(sdf.brickBakeParamsAddr).params;
    int cellIdx = csgCellIdx(worldPos, sdf);
    float worldStepSize = kNormalStepVoxels * sdf.voxelSize;
    bool surfaceExistsAtAllSamples = false;

    float f0 = sdf_active(
        worldPos + (vec3(1.0, -1.0, -1.0) * worldStepSize),
        cellIdx,
        params,
        surfaceExistsAtAllSamples);
    float f1 = sdf_active(
        worldPos + (vec3(-1.0, -1.0, 1.0) * worldStepSize),
        cellIdx,
        params,
        surfaceExistsAtAllSamples);
    float f2 = sdf_active(
        worldPos + (vec3(-1.0, 1.0, -1.0) * worldStepSize),
        cellIdx,
        params,
        surfaceExistsAtAllSamples);
    float f3 = sdf_active(
        worldPos + (vec3(1.0, 1.0, 1.0) * worldStepSize),
        cellIdx,
        params,
        surfaceExistsAtAllSamples);

    vec3 gradientDir = vec3(
        (f0 + f3) - (f1 + f2),
        (f2 + f3) - (f0 + f1),
        (f1 + f3) - (f0 + f2));

    if (!surfaceExistsAtAllSamples
        || gradientDir == vec3(0.0))
    {
        return false;
    }

    normal = normalize(gradientDir);
    return true;
}

#endif // CSG_EVAL_GLSL
