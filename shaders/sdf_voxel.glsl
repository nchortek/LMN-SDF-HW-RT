#ifndef SDF_VOXEL_GLSL
#define SDF_VOXEL_GLSL

// The SDF intersection shader passes this (alongside a t-value)
// to the SDF closest hit shader
//
// The Vulkan spec only guarantees this to be 32 bytes.
// If we surpass that we'll need to query it via:
// VkPhysicalDeviceRayTracingPipelinePropertiesKHR::maxRayHitAttributeSize
struct SdfHitAttributes
{
    // The given voxel's integer coordinates within its brick's 8x8x8 grid,
    // packed into a single uint:  unused|zzz|yyy|xxx
    // 
    // Each coordinate only needs three bits because voxel indices range [0, 7]
    uint packedBrickLocalVoxelIdx;

    // The hit position in voxel-local space (coordinates in range [0, 1])
    vec3 voxelLocalPos;
};

uint packBrickLocalVoxelIdx(uvec3 voxelIdx)
{
    // Pack each [0, 7] coordinate into 3 bits
    return voxelIdx.x | (voxelIdx.y << 3) | (voxelIdx.z << 6);
}

uvec3 unpackBrickLocalVoxelIdx(uint packedVoxelIdx)
{
    // 7u is a 3-bit all-set mask
    return uvec3(packedVoxelIdx & 7u,
        (packedVoxelIdx >> 3) & 7u,
        (packedVoxelIdx >> 6) & 7u);
}

// Returns the 8 dequantized samples that lie on the corners of the given voxel within a brick.
//
// globalBrickIdx is the brick's global index
// brickLocalVoxelIdx is the given voxel's integer coordinates within its brick's 8x8x8 grid
float[8] getVoxelSamples(uint64_t samplesAddr, uint globalBrickIdx, uvec3 brickLocalVoxelIdx, float band)
{
    float cornerSamples[8];
    SampleBuffer sampleBuffer = SampleBuffer(samplesAddr);

    // Samples are quantized 8-bit snorm values packed into 32-bit words,
    // but we will read from the sample buffer here as an array of int8s
    const uint brickStartIdx = globalBrickIdx * BRICK_STRIDE_WORDS * 4u;
    const uint samplesPerAxis = BRICK_VOXELS + 1u;

    // Each sample is stored quantized to the range [-127, 127], and we want
    // to return the dequantized value in the range [-band, band]
    const float scale = band / 127.0;

    for (uint k = 0u; k < 2u; k++)
    {
        for (uint j = 0u; j < 2u; j++)
        {
            for (uint i = 0u; i < 2u; i++)
            {
                uvec3 brickLocalSampleIdx = brickLocalVoxelIdx + uvec3(i, j, k);
                uint flatBrickLocalSampleIdx =
                    (((brickLocalSampleIdx.z * samplesPerAxis) + brickLocalSampleIdx.y) * samplesPerAxis)
                    + brickLocalSampleIdx.x;

                cornerSamples[i + (2u * j) + (4u * k)] =
                    float(int(sampleBuffer.samples[brickStartIdx + flatBrickLocalSampleIdx])) * scale;
            }
        }
    }

    return cornerSamples;
}

#endif