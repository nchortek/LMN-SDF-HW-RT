#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "gpu/shared.h"
#include "common.glsl"
#include "sdf_voxel.glsl"
#include "pruning/csg_eval.glsl"
#include "rng.glsl"

// This shader will fill the payload, which is then read by raygen after calling traceRayEXT.
// The location must match the payload location raygen passes to traceRayEXT.
layout(location = 0) rayPayloadInEXT IntersectionInfo payload;

// This gets passed from the SDF intersection shader alongside the t-value of an intersection
hitAttributeEXT SdfHitAttributes hitAttributes;

layout(push_constant, scalar) uniform PushConstantBlock
{
    PushConstants pc;
};

// Given the 8 corner samples of a voxel and a voxel-local surface position (coords in the
// range [0, 1]), compute the surface normal from the gradient of the surface, as defined
// by the trilinear interpolation of the voxel's corner samples.
//
// Adapted from Soderlund et al 2022 Eqs. 1 and 9-11
vec3 computeSurfaceNormal(float voxelSamples[8], vec3 voxelLocalSurfacePos)
{
    float y0 = mix(
        voxelSamples[1] - voxelSamples[0],
        voxelSamples[3] - voxelSamples[2],
        voxelLocalSurfacePos.y);
    float y1 = mix(
        voxelSamples[5] - voxelSamples[4],
        voxelSamples[7] - voxelSamples[6],
        voxelLocalSurfacePos.y);
    float dfdx = mix(y0, y1, voxelLocalSurfacePos.z);

    float x0 = mix(
        voxelSamples[2] - voxelSamples[0],
        voxelSamples[3] - voxelSamples[1],
        voxelLocalSurfacePos.x);
    float x1 = mix(
        voxelSamples[6] - voxelSamples[4],
        voxelSamples[7] - voxelSamples[5],
        voxelLocalSurfacePos.x);
    float dfdy = mix(x0, x1, voxelLocalSurfacePos.z);

    x0 = mix(
        voxelSamples[4] - voxelSamples[0],
        voxelSamples[5] - voxelSamples[1],
        voxelLocalSurfacePos.x);
    x1 = mix(
        voxelSamples[6] - voxelSamples[2],
        voxelSamples[7] - voxelSamples[3],
        voxelLocalSurfacePos.x);
    float dfdz = mix(x0, x1, voxelLocalSurfacePos.y);

    return vec3(dfdx, dfdy, dfdz);
}

vec3 computeCsgAlbedo(vec3 worldPos, SdfSceneAddresses sdf) {
    if (sdf.brickBakeParamsAddr == uint64_t(0)) {
        return vec3(-1.f);
    }
    return sdf_color_active(worldPos, 
                            csgCellIdx(worldPos, sdf), 
                            BrickBakeParamsBuffer(sdf.brickBakeParamsAddr).params);
}

void main()
{
    SceneAddresses sceneAddresses = SceneAddressesBuffer(pc.sceneAddr).deviceAddresses;
    SdfSceneAddresses sdfAddresses = SdfSceneAddressesBuffer(sceneAddresses.sdfAddr).sdfDeviceAddresses;

    float voxelSamples[8] = getVoxelSamples(
        sdfAddresses.samplesAddr,
        uint(gl_PrimitiveID),
        unpackBrickLocalVoxelIdx(hitAttributes.packedBrickLocalVoxelIdx),
        sdfAddresses.band);

    vec3 objectNor = computeSurfaceNormal(voxelSamples, hitAttributes.voxelLocalPos);

    // Set degenerate normals to point toward the camera to avoid NaNs 
    // produced by normalization
    if (objectNor == vec3(0.0))
    {
        objectNor = -gl_ObjectRayDirectionEXT;
    }

    // This is equivalent to transpose(inverse(modelMatrix))
    mat3 normalMatrix = transpose(mat3(gl_WorldToObjectEXT));
    vec3 worldNor = normalize(normalMatrix * objectNor);

    payload.worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.worldShadingNor = worldNor;
    payload.worldGeometricNor = worldNor;
    payload.uv = vec2(0.0);
    payload.materialIndex = gl_InstanceCustomIndexEXT;
    payload.primitiveId = gl_PrimitiveID;
    payload.albedo = computeCsgAlbedo(payload.worldPos, sdfAddresses);
    payload.t = gl_HitTEXT;
}
