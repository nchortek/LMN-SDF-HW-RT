#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "gpu/shared.h"
#include "common.glsl"

// This shader will fill the payload, which is then read by raygen after calling traceRayEXT.
// The location must match the payload location raygen passes to traceRayEXT.
layout(location = 0) rayPayloadInEXT IntersectionInfo payload;

void main()
{
    // sdf_brick.rint reports the entered AABB face as the hit kind:
    // 0: +x, 1: -x, 2: +y, 3: -y, 4: +z, 5: -z
    uint axis = gl_HitKindEXT / 2u;
    vec3 objectNor = vec3(0.0);
    objectNor[axis] = ((gl_HitKindEXT % 2u) == 0u) ? 1.0 : -1.0;

    // This is equivalent to transpose(inverse(modelMatrix))
    mat3 normalMatrix = transpose(mat3(gl_WorldToObjectEXT));
    vec3 worldNor = normalize(normalMatrix * objectNor);

    payload.worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.worldShadingNor = worldNor;
    payload.worldGeometricNor = worldNor;
    payload.uv = vec2(0.0);
    payload.materialIndex = gl_InstanceCustomIndexEXT;
    payload.primitiveId = gl_PrimitiveID;
    payload.t = gl_HitTEXT;
}
