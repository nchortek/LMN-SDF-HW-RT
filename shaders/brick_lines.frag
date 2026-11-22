#version 450

#include "gpu/sdf_shared.h"

layout(push_constant, scalar) uniform WireframePushConstantBlock{
    WireframePushConstants pc;
};


#if BRICK_OVERLAY_DEPTH
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer OverlayDepthBuffer{
    float depths[];
};

//so lines lying on the surface aren't hidden by float error
const float kDepthBias = 0.002;
#endif

layout(location = 0) in vec3 inColor;

layout(location = 0) out vec4 outColor;

void main(){
#if BRICK_OVERLAY_DEPTH
    //The trace may run at a different resolution than the swapchain( the blit may stretch it)
    const uvec2 renderPixel = uvec2(gl_FragCoord.xy * pc.fragToRender);
    const float sceneDepth = OverlayDepthBuffer(pc.depthAddr).depths[renderPixel.y*pc.renderWidth + renderPixel.x];

    //perpsective projection, clipW is view space depth and gl_Fragcoord.w is 1/clipW 
    const float lineDepth = 1.0/gl_FragCoord.w;

    if(lineDepth > sceneDepth*(1.0 + kDepthBias)){
        discard;
    }
#endif

    //drawing wireframes
    outColor = vec4(inColor * pc.color.rgb, 1.0);
}