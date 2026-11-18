#include "baking.h"

#include "vk/context.h"
#include "vk/pipeline.h"
#include <cstdio>
#include <vector>
#include <cstring>


//from gpuscene.cpp, made local for convenience
// TODO: move to a utility header!!!! copied 3x! :(
static void recordMemoryBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags2 srcStage,
    VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,
    VkAccessFlags2 dstAccess)
{
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}


bool SdfBaking::init(VulkanContext& context) {
    m_context = &context;
    
    if(!createBakingPipeline()) {
        fprintf(stderr, "SdfBaking: createBakingPipeline failed\n");
        return false;
    }
    if(!createCountReadbackBuffer()) {
        fprintf(stderr, "SdfBaking: createCountReadbackBuffer failed\n");
        return false;
    }

    return true;
}

void SdfBaking::destroy(){
    if(m_context == nullptr){
        return;
    }

    if (m_countStaging != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_context->getAllocator(), m_countStaging, m_countStagingAlloc);
        m_countStaging = VK_NULL_HANDLE;
        m_countStagingAlloc = VK_NULL_HANDLE;
    }

    VkDevice device = m_context->getDevice();
    vkDestroyPipeline(device, m_bakePipeline, nullptr);
    m_bakePipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(device, m_bakeLayout, nullptr);
    m_bakeLayout = VK_NULL_HANDLE;
    m_context = nullptr;
}


bool SdfBaking::createBakingPipeline(){
    VkDevice device = m_context->getDevice();

    //load shader
    std::vector<uint32_t> code; //code holding raw words
    if(!Pipeline::readSpirvFile(std::string(SHADER_DIR) + "/brick_bake.comp.spv", code)){
        fprintf(stderr, "Failed to read SdfBaking spirv.\n");
        return false;
    }

    //create module (vulkans' object for a piece of shader code that has been handed to the driver but not compiled yet - a SPIR-V blob with a handle)
    VkShaderModule module = VK_NULL_HANDLE;
    if(!Pipeline::createVkShaderModule(m_context->getDevice(), code, module)){
        fprintf(stderr, "Failed to create SdfBaking module.\n");
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(gpu::BakePushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_bakeLayout) != VK_SUCCESS){
        m_bakeLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(device, module, nullptr);
        fprintf(stderr, "Failed to create SdfBaking baking pipeline layout\n");
        return false;
    }

    //compute pipeline
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = module;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = m_bakeLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_bakePipeline);

    //module is baked into the pipeline
    vkDestroyShaderModule(device, module, nullptr);
    if(result != VK_SUCCESS){
        m_bakePipeline = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create baking compute pipeline\n");
        return false;
    }

    return true;
}

// Host-visible buffer the brick count gets copied into, so the CPU can read it
bool SdfBaking::createCountReadbackBuffer(){
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(uint32_t);
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    if (vmaCreateBuffer(m_context->getAllocator(), &bufferInfo, &allocInfo, &m_countStaging, &m_countStagingAlloc, &m_countStagingInfo) != VK_SUCCESS) {
        m_countStaging = VK_NULL_HANDLE;
        m_countStagingAlloc = VK_NULL_HANDLE;
        fprintf(stderr, "SdfBaking: failed to create count readback buffer\n");
        return false;
    }

    return true;
}

bool SdfBaking::runBaking(uint64_t sdfSceneAddr, const vkutil::Buffer& brickCountBuffer, const vkutil::Buffer& aabbBuffer, uint32_t& outBrickCount){
    VkCommandBuffer cmd = m_context->beginSingleTimeCommands();
    recordBake(cmd, sdfSceneAddr, brickCountBuffer, aabbBuffer);
    recordCountReadback(cmd, brickCountBuffer);
    m_context->endSingleTimeCommands(cmd);

    outBrickCount = readBrickCount();
    return true;
}

// Init buffers then dispatch bake compute shader
void SdfBaking::recordBake(VkCommandBuffer cmd, uint64_t sdfSceneAddr, const vkutil::Buffer& brickCountBuffer, const vkutil::Buffer& aabbBuffer){
    vkCmdFillBuffer(cmd, brickCountBuffer.bufferHandle, 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, aabbBuffer.bufferHandle, 0, VK_WHOLE_SIZE, 0x7FC00000u); // NANs

    // Buffers must be initialized before compute shader can read/write them
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT  | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bakePipeline);

    // Set single push constant value
    m_pushConstants.sdfSceneAddr = sdfSceneAddr;
    vkCmdPushConstants(cmd, m_bakeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu::BakePushConstants), &m_pushConstants);
        
    // Launch compute shader with one workgroup per brick (1 thread per voxel)
    const uint32_t bricksPerAxis = gpu::VOXELS_PER_SCENE_AXIS / gpu::BRICK_VOXELS; // 64
    vkCmdDispatch(cmd, bricksPerAxis, bricksPerAxis, bricksPerAxis);
}

// Copies the count into the staging buffer
void SdfBaking::recordCountReadback(VkCommandBuffer cmd, const vkutil::Buffer& brickCountBuffer){
    // Shader must finish before the copy reads the count
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,       VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferCopy region{};
    region.size = sizeof(uint32_t);
    vkCmdCopyBuffer(cmd, brickCountBuffer.bufferHandle, m_countStaging, 1, &region);

    // Copy must finish before the CPU reads the staging buffer
    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_HOST_BIT,     VK_ACCESS_2_HOST_READ_BIT);
}

// Call only after the command buffer from recordCountReadback has completed
uint32_t SdfBaking::readBrickCount(){
    // Drop any stale CPU cache lines (no-op on host-coherent memory)
    vmaInvalidateAllocation(m_context->getAllocator(), m_countStagingAlloc, 0, VK_WHOLE_SIZE);
    
    uint32_t count = 0;
    memcpy(&count, m_countStagingInfo.pMappedData, sizeof(uint32_t));

    // Clamp bricks
    if (count > gpu::MAX_BRICKS) {
        fprintf(stderr, "SdfBaking: %u bricks exceeds maximum of %u, clamping!\n", count, gpu::MAX_BRICKS);
        count = gpu::MAX_BRICKS;
    }
    return count;
}