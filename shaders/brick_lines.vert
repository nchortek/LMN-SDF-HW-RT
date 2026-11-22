#version 460

#include "common.glsl"
#include "rng.glsl"

//a push constant is a small block of data that you write directly into the command buffer with vkCmdPushConstants
//vulkan glsl only allows uniforms inside blocks (uniform buffer boudn with a descriptor set, or a push constant block)

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer AabbRef  {
    Aabb aabbs[];
};

layout(push_constant, scalar) uniform WireframePushConstantBlock{
    WireframePushConstants pc;
};

ivec2 cornerTable[12] = ivec2[](
    ivec2(0, 1), ivec2(2, 3), ivec2(4, 5), ivec2(6, 7),
    ivec2(0, 2), ivec2(1, 3), ivec2(4, 6), ivec2(5, 7),
    ivec2(0, 4), ivec2(1, 5), ivec2(2, 6), ivec2(3, 7)
);

layout(location = 0) out vec3 vertColor;

void main(){
    //drawing wireframes

    int brick = gl_VertexIndex / 24;
    int localIndex = gl_VertexIndex % 24;

    int edge = localIndex/2; //which edge of the local box we'll pick
    int endpoint = localIndex%2; //which component of the local box

    ivec2 cornerIdxPair = cornerTable[edge];
    int cornerIdx = cornerIdxPair[endpoint];

    vec3 bmin = vec3(AabbRef(pc.aabbAddr).aabbs[brick].minValues);
    vec3 bmax = vec3(AabbRef(pc.aabbAddr).aabbs[brick].maxValues);
    
    float bcenterheight = (bmin.y + bmax.y)*0.5;

    //discard any brick wireframes that lie outside the slice range
    if(bcenterheight < pc.yParams.x || bcenterheight > pc.yParams.y){
        vertColor = vec3(0.0);
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); //setting it outside of the viewport so it gets culled
        return;
    }

    float centerNormalized = clamp((bcenterheight - pc.yParams.z) / (pc.yParams.w - pc.yParams.z), 0.0, 1.0);

    if(pc.colorMode == OverlayColorBrickId){
        uint hash = utilhash(uint(brick)); //same as raygen.rgen
        vertColor = vec3(hash & 0xFFu, (hash >> 8) & 0xFFu, (hash >> 16) & 0xFFu) / 255.0; 
    }
    else if(pc.colorMode == OverlayColorAlbedo){
        vertColor = vec3(1.0); //frag multiplies by pc.color.rgb, so set to white
    }
    else if(centerNormalized < 0.5){
        vertColor = mix(vec3(0.0,0.0,1.0), vec3(0.0,1.0,0.0), centerNormalized*2);
    }
    else{
        vertColor = mix(vec3(0.0,1.0,0.0), vec3(1.0,0.0,0.0), centerNormalized*2 - 1);
    }


    //turn corner number into actual position
    //corner = 5: binary 101 = (bmax.x, bmin.y, bmax.z)
    vec3 cornerPos;
    cornerPos.x = ((cornerIdx >> 0) & 1) == 0 ? bmin.x : bmax.x;
    cornerPos.y = ((cornerIdx >> 1) & 1) == 0 ? bmin.y : bmax.y;
    cornerPos.z = ((cornerIdx >> 2) & 1) == 0 ? bmin.z : bmax.z;

    gl_Position = pc.viewProj * vec4(cornerPos, 1.0);

    //Vulkan's clip sapce has +Y pointing down, GLM was written for OpenGL, so glm::perspective assumes +Y is up
    gl_Position.y *= -1.0;
}