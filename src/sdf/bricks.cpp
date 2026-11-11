#include "bricks.h"

#include <cstdio>

namespace bricksutil
{
    bool readBricks(const char* path, BricksFile& b)
    {
        FILE* f = fopen(path, "rb");
        if (!f)
        {
            fprintf(stderr, "Failed to open bricks file %s\n", path);
            return false;
        }

        bool success = fread(&b.brickCount, 4, 1, f) == 1
            && fread(b.domainMin, 4, 3, f) == 3
            && fread(&b.voxelSize, 4, 1, f) == 1
            && fread(&b.band, 4, 1, f) == 1
            && b.brickCount <= gpu::MAX_BRICKS
            && b.voxelSize > 0.0f;

        if (success)
        {
            b.mortons.resize(b.brickCount);
            b.aabbs.resize(b.brickCount);
            b.samples.resize(size_t(b.brickCount) * 729);

            // The reads below process 24 + brickCount * 757 bytes. By checking EOF last,
            // we are confirming that the file is exactly the expected size.
            success = fread(b.mortons.data(), 4, b.mortons.size(), f) == b.mortons.size()
                && fread(b.aabbs.data(), sizeof(gpu::Aabb), b.aabbs.size(), f) == b.aabbs.size()
                && fread(b.samples.data(), 1, b.samples.size(), f) == b.samples.size()
                && fgetc(f) == EOF;
        }

        fclose(f);

        if (!success)
        {
            fprintf(stderr, "Invalid bricks data in file %s\n", path);
        }

        return success;
    }

} // namespace bricksutil
