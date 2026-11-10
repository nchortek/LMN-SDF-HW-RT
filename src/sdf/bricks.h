#pragma once

#include "gpu/sdf_shared.h"

#include <cstdint>
#include <vector>

namespace bricksutil
{
    // Contents of a .bricks file
    // Brick i is mortons[i], aabbs[i] and samples[i*729 --> i*729 + 728].
    struct BricksFile
    {
        uint32_t brickCount = 0;

        // The world-space minimum corner (x, y, z) of the cubic domain that the
        // bricks are baked over.
        float domainMin[3]{};
        float voxelSize = 0;

        // The band size used to quantize/dequantize the SDF samples stored
        // at voxel corners.
        // Based on Ray Tracing of SDF Grids (Söderlund et al., 2022).
        float band = 0;

        // brickCount Morton indices
        std::vector<uint32_t> mortons;

        // brickCount world-space AABBs, read straight from the file
        std::vector<gpu::Aabb> aabbs;

        // brickCount * 729 samples
        std::vector<int8_t> samples;
    };

    // Reads a .bricks file into b
    bool readBricks(const char* path, BricksFile& b);

} // namespace bricksutil
