#include "brickoverlay.h"

#include "context.h"
#include "pipeline.h"

#include <cstdio>
#include <string>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

bool BrickOverlay:: init(VulkanContext& context, VkFormat colorFormat){
    m_context = &context;

    //init push constants
    m_pushConstants = {};
    m_pushConstants.viewProj = glm::mat4(1.0f);
    m_pushConstants.color = glm::vec4(1.0f);
    m_pushConstants.yParams = glm::vec4(-1e30f, 1e30f, 0.0f, 1.0f);

    if(!createPipeline(colorFormat)){
        fprintf(stderr, "BrickOverlay createpipieline failed\n");
        destroy();
        return false;
    }
    return true;
}

void BrickOverlay::destroy(){
    if(m_context == nullptr){
        return;
    }

    VkDevice device = m_context->getDevice();
    vkDestroyPipeline(device, m_pipeline, nullptr);
    m_pipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(device, m_layout, nullptr);
    m_layout = VK_NULL_HANDLE;
    m_context = nullptr;
}

bool BrickOverlay::createPipeline(VkFormat colorFormat){
    VkDevice device = m_context->getDevice();
    
    std::vector<uint32_t> vertCode;
    std::vector<uint32_t> fragCode;
    if(!Pipeline::readSpirvFile(std::string(SHADER_DIR) + "/brick_lines.vert.spv", vertCode) ||
       !Pipeline::readSpirvFile(std::string(SHADER_DIR) + "/brick_lines.frag.spv", fragCode)){
        
        fprintf(stderr, "Failed to read shader file in brickoverlay\n");
        return false;
    }

    VkShaderModule vert_module = VK_NULL_HANDLE;
    VkShaderModule frag_module = VK_NULL_HANDLE;
    if (!Pipeline::createVkShaderModule(device, vertCode, vert_module) ||
        !Pipeline::createVkShaderModule(device, fragCode, frag_module)){
        
        vkDestroyShaderModule(device, vert_module, nullptr);
        vkDestroyShaderModule(device, frag_module, nullptr);
        fprintf(stderr, "BrickOverlay: failed to create shader modules\n");
        return false;
    }


    VkPipelineShaderStageCreateInfo vert_stage_info = {};
    vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage_info.module = vert_module;
    vert_stage_info.pName = "main"; //entry point

    VkPipelineShaderStageCreateInfo frag_stage_info = {};
    frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage_info.module = frag_module;
    frag_stage_info.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stages[] = { vert_stage_info, frag_stage_info };

    VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount = 0;
    vertex_input_info.vertexAttributeDescriptionCount = 0;

    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; //draw lines instead of triangles
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state = {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo color_blending = {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &colorBlendAttachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;
    
    VkPushConstantRange range = {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(gpu::WireframePushConstants)
    };

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 0;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &range;

    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &m_layout) != VK_SUCCESS)
    {
        m_layout = VK_NULL_HANDLE;
        vkDestroyShaderModule(device, vert_module, nullptr);
        vkDestroyShaderModule(device, frag_module, nullptr);
        fprintf(stderr, "BrickOverlay: failed to create pipeline layout\n");
        return false;
    }

    std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamic_info = {};
    dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_info.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
    dynamic_info.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &colorFormat;
    rendering_info.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    rendering_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    
    VkGraphicsPipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &rendering_info;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_info;
    pipeline_info.pDepthStencilState = nullptr; //no depth attachment
    pipeline_info.layout = m_layout;
    pipeline_info.renderPass = VK_NULL_HANDLE;  //dynamic rendering
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    
    VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline);

    vkDestroyShaderModule(device, frag_module, nullptr);
    vkDestroyShaderModule(device, vert_module, nullptr);

    if (result != VK_SUCCESS)
    {
        m_pipeline = VK_NULL_HANDLE;
        fprintf(stderr, "BrickOverlay: failed to create graphics pipeline\n");
        return false;
    }

    return true;
}

void BrickOverlay::setCamera(const gpu::CameraParams& camera, VkExtent2D renderExtent){
    //pixelLegnth = 2 * tan(FOV/2)/resolution
    const float tanHalfY = 0.5*camera.pixelLength.y*float(renderExtent.height);
    const float tanHalfX = 0.5*camera.pixelLength.x*float(renderExtent.width);

    const float fovY = 2.0*std::atan(tanHalfY);
    const float aspect = tanHalfX/tanHalfY;

    const glm::mat4 view = glm::lookAtRH(camera.position, camera.position + camera.view, camera.up);
    const glm::mat4 proj = glm::perspectiveRH_ZO(fovY, aspect, kNear, kFar);

    m_pushConstants.viewProj = proj*view;
}

bool BrickOverlay::recordDrawCommands(VkCommandBuffer commandBuffer, VkExtent2D extent, uint64_t aabbAddr, uint32_t brickCount){
    if(!m_enabled || brickCount ==0 || m_pipeline == VK_NULL_HANDLE){
        return false;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = float(extent.width);
    viewport.height = float(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = {0,0};
    scissor.extent = extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    m_pushConstants.aabbAddr = aabbAddr;

#if BRICK_OVERLAY_DEPTH
    m_pushConstants.fragToRender = glm::vec2(float(m_renderExtent.width) / float(extent.width), float(m_renderExtent.height) / float(extent.height));
#endif

    vkCmdPushConstants(commandBuffer, m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_pushConstants), &m_pushConstants);

    //12edges * 2 endpoints
    vkCmdDraw(commandBuffer, 24 * brickCount, 1, 0, 0);
    return true;
}