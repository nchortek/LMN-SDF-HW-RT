#ifndef SDF_VOXEL_GLSL
#define SDF_VOXEL_GLSL

// Returns the 8 dequantized samples that lie on the corners of the given voxel within a brick.
//
// globalBrickIdx is the brick's global index
// localVoxelIdx is the given voxel's integer coordinates within its brick's 8x8x8 grid
float[8] getVoxelSamples(uint64_t samplesAddr, uint globalBrickIdx, uvec3 localVoxelIdx, float band)
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
                uvec3 localSampleIdx = localVoxelIdx + uvec3(i, j, k);
                uint flatLocalSampleIdx =
                    (((localSampleIdx.z * samplesPerAxis) + localSampleIdx.y) * samplesPerAxis)
                    + localSampleIdx.x;

                cornerSamples[i + (2u * j) + (4u * k)] =
                    float(int(sampleBuffer.samples[brickStartIdx + flatLocalSampleIdx])) * scale;
            }
        }
    }

    return cornerSamples;
}

#endif