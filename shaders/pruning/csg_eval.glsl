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

int csgCellIdx(vec3 worldPos, SdfSceneAddresses sdf) {
    // Worldspace pos -> voxelspace pos -> prunespace pos -> prune_idx
    ivec3 voxelspace_pos = ivec3((worldPos - sdf.domainMin) / sdf.voxelSize);
    ivec3 prunespace_pos = clamp(voxelspace_pos >> 1, ivec3(0), ivec3(255));
    int prune_idx = int(get_cell_idx(prunespace_pos, int(PRUNING_CELLS_PER_SCENE_AXIS)));
    return prune_idx;
}

#endif // CSG_EVAL_GLSL
