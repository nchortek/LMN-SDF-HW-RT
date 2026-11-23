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

#endif // CSG_EVAL_GLSL
