#include "pruning.h"

#include "scene.h"
#include "vk/context.h"
#include "vk/pipeline.h"
#include <cstdio>
#include <iostream>
#include <utility>

//referenced from gpuscene.cpp
bool SdfPruning::init(VulkanContext& context, const Scene& scene){
    m_context = &context;

    m_nodeCount = scene.nodeCount;
    m_gpuRootIdx = scene.gpuRootIdx;
    m_aabbMin = scene.paddedAABBmin;
    m_aabbMax = scene.paddedAABBmax;

    if(!createInputBuffers(scene)){
        fprintf(stderr, "SdfPruning: createInputBuffers failed\n");
        return false;
    }
    if(!createWorkBuffers()){
        fprintf(stderr, "SdfPruning: createWorkBuffers failed\n");
        return false;
    }
    if(!createCullingPipeline()){
        fprintf(stderr, "SdfPruning: createCullingPipeline failed\n");
        return false;
    }

    if(!runPruning()){
        fprintf(stderr, "SdfPruning: runPruning failed\n");
        return false;
    }

    return true;
}

//from gpuscene.cpp, made local for convenience
void SdfPruning::recordMemoryBarrier(
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

bool SdfPruning::runPruning(){
    VkCommandBuffer cmd = m_context->beginSingleTimeCommands();
    
    m_inputIdx = 0;
    m_outputIdx = 1;

    VkBufferCopy region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = m_nodeCount * sizeof(uint16_t)
    };
    vkCmdCopyBuffer(cmd, m_parentsInit.bufferHandle, m_parents[m_inputIdx].bufferHandle, 1, &region);
    vkCmdCopyBuffer(cmd, m_activeNodesInit.bufferHandle, m_activeNodes[m_inputIdx].bufferHandle, 1, &region);
    vkCmdFillBuffer(cmd, m_activeCount.bufferHandle, 0, numLevelSlots * sizeof(int32_t), 0);
    vkCmdFillBuffer(cmd, m_oldToNewCount.bufferHandle, 0, numLevelSlots * sizeof(int32_t), 0);

    recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT  | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pruningPipeline);

    setStaticPushConstants();

    //4^3 -> 16^3 -> 64^3 -> 256^3
    bool firstLvl = true;
    for(int lvl = 2; lvl <= finalGridLvl; lvl += 2){
        setLevelPushConstants(lvl, firstLvl);
        vkCmdPushConstants(cmd, m_pruningLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PruningPushConstants), &m_pushConstants);
        
        //4x4x4 workgroup
        uint32_t groups = (uint32_t)((m_pushConstants.grid_size + 3)/4);
        vkCmdDispatch(cmd, groups, groups, groups);
        
        //next level reads current level's output - everything in the src stage that did src access (transfer writes) must finish and be made visible before anything in the dst stage does dst access
        recordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        firstLvl = false;
        
        if(lvl!=finalGridLvl){ //store final 256^3 result in output slot
            std::swap(m_inputIdx,m_outputIdx); 
        }
    }

    m_context->endSingleTimeCommands(cmd);

    return true;
} 

void SdfPruning::setStaticPushConstants(){
    //static push constants
    m_pushConstants = {};

    m_pushConstants.aabb_min = glm::vec4(m_aabbMin, 0.0f);
    m_pushConstants.aabb_max = glm::vec4(m_aabbMax, 0.0f);
    m_pushConstants.num_nodes = m_nodeCount;

    //input buffer data
    m_pushConstants.prims_ref = m_prims.deviceAddress;
    m_pushConstants.binary_ops_ref = m_binaryOps.deviceAddress;
    m_pushConstants.nodes_ref = m_nodes.deviceAddress;

    //scratch
    m_pushConstants.tmp_ref = m_tmp.deviceAddress;
    m_pushConstants.old_to_new_scratch_ref = m_oldToNewScratch.deviceAddress;

    //lipschitz defaults
    m_pushConstants.max_rel_err = 1.0f;
    m_pushConstants.alpha = 1.0f;
    m_pushConstants.culling_enabled = 1;

    //rendering only, not needed for culling, placeholders for size
    m_pushConstants.mvp_ref = 0;
    m_pushConstants.cam_ref = 0;
    m_pushConstants.resolution = glm::ivec2(0);
}


void SdfPruning::setLevelPushConstants(int lvl, bool firstLvl){
    //level push constants (constants that change per level or pingpong swap)
    m_pushConstants.grid_size = 1 << lvl;
    m_pushConstants.first_lvl = firstLvl? 1: 0;

    m_pushConstants.active_nodes_in_ref = m_activeNodes[m_inputIdx].deviceAddress;
    m_pushConstants.active_nodes_out_ref = m_activeNodes[m_outputIdx].deviceAddress;
    m_pushConstants.parents_in_ref = m_parents[m_inputIdx].deviceAddress;
    m_pushConstants.parents_out_ref = m_parents[m_outputIdx].deviceAddress;
    m_pushConstants.cell_offsets_in_ref = m_cellOffsets[m_inputIdx].deviceAddress;
    m_pushConstants.cell_offsets_out_ref = m_cellOffsets[m_outputIdx].deviceAddress;
    m_pushConstants.num_active_in_ref = m_numActive[m_inputIdx].deviceAddress;
    m_pushConstants.num_active_out_ref = m_numActive[m_outputIdx].deviceAddress;
    m_pushConstants.cell_error_in_ref = m_cellErrors[m_inputIdx].deviceAddress;
    m_pushConstants.cell_error_out_ref = m_cellErrors[m_outputIdx].deviceAddress;

    //atomic bump allocator (next free index where threads claim space from) - every level writes a resh output list starting at index 0
    m_pushConstants.active_count_ref = m_activeCount.deviceAddress + lvl*sizeof(int32_t);
    m_pushConstants.old_to_new_count_ref = m_oldToNewCount.deviceAddress + lvl*sizeof(int32_t);
}

void SdfPruning::destroy(){
    if(m_context == nullptr){
        return;
    }

    VkDevice device = m_context->getDevice();
    vkDestroyPipeline(device, m_pruningPipeline, nullptr);
    m_pruningPipeline = VK_NULL_HANDLE;
    vkDestroyPipelineLayout(device, m_pruningLayout, nullptr);
    m_pruningLayout = VK_NULL_HANDLE;

    vkutil::destroyBuffer(*m_context, m_nodes);
    vkutil::destroyBuffer(*m_context, m_binaryOps);
    vkutil::destroyBuffer(*m_context, m_prims);
    vkutil::destroyBuffer(*m_context, m_parentsInit);
    vkutil::destroyBuffer(*m_context, m_activeNodesInit);

    for(int i = 0; i < 2; i++){
        vkutil::destroyBuffer(*m_context, m_numActive[i]);
        vkutil::destroyBuffer(*m_context, m_cellOffsets[i]);
        vkutil::destroyBuffer(*m_context, m_cellErrors[i]);
        vkutil::destroyBuffer(*m_context, m_activeNodes[i]);
        vkutil::destroyBuffer(*m_context, m_parents[i]);
    }

    vkutil::destroyBuffer(*m_context, m_tmp);
    vkutil::destroyBuffer(*m_context, m_oldToNewScratch);
    vkutil::destroyBuffer(*m_context, m_activeCount);
    vkutil::destroyBuffer(*m_context, m_oldToNewCount);

    m_context = nullptr;
}

//referenced from context.cpp
bool SdfPruning::createInputBuffers(const Scene& scene){
    VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    if (!vkutil::createAndUploadBuffer(*m_context, scene.gpuNodeList.data(), scene.gpuNodeList.size()*sizeof(GPUNode), buffer_usage, m_nodes)){
        fprintf(stderr, "Failed to create the m_nodes buffer on the GPU.\n");
        return false;
    }
 
    if (!scene.binaryOpList.empty() && !vkutil::createAndUploadBuffer(*m_context, scene.binaryOpList.data(), scene.binaryOpList.size()*sizeof(BinaryOp), buffer_usage, m_binaryOps)){
          fprintf(stderr, "Failed to create the binaryops buffer on the GPU.\n");
        return false;
    }
    
    if (!vkutil::createAndUploadBuffer(*m_context, scene.primitiveList.data(), scene.primitiveList.size()*sizeof(Primitive), buffer_usage, m_prims)){
          fprintf(stderr, "Failed to create the primitive buffer on the GPU.\n");
        return false;
    }
    
    if (!vkutil::createAndUploadBuffer(*m_context, scene.parents.data(), scene.parents.size()*sizeof(uint16_t), buffer_usage, m_parentsInit)){
          fprintf(stderr, "Failed to create the parents buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createAndUploadBuffer(*m_context, scene.activeNodes.data(), scene.activeNodes.size()*sizeof(uint16_t), buffer_usage, m_activeNodesInit)){
      fprintf(stderr, "Failed to create the activenodes buffer on the GPU.\n");
        return false;
    }

    return true;
}

//referenced from context.cpp
bool SdfPruning::createWorkBuffers(){
    VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    const VkDeviceSize gridIntSize = numCells * sizeof(int32_t); //num_active, cell_offsets
    const VkDeviceSize gridFloatSize = numCells * sizeof(float); //cell_errors
    const VkDeviceSize activeListSize = maxActiveCount * sizeof(uint16_t); //active_nodes, parents
    const VkDeviceSize tmpSize = maxTmpCount * sizeof(uint32_t);
    const VkDeviceSize oldToNewSize = maxTmpCount * sizeof(uint16_t);
    const VkDeviceSize counterSize = numLevelSlots * sizeof(int32_t); //one int per grid level

    //pingpong
    for(int i = 0; i < 2; i++){
        if (!vkutil::createBuffer(*m_context, gridIntSize, buffer_usage, 0, m_numActive[i])){
            fprintf(stderr, "Failed to create the num_active buffer on the GPU.\n");
            return false;
        }
        if (!vkutil::createBuffer(*m_context, gridIntSize, buffer_usage, 0, m_cellOffsets[i])){
            fprintf(stderr, "Failed to create the cell_offsets buffer on the GPU.\n");
            return false;
        }
        if (!vkutil::createBuffer(*m_context, gridFloatSize, buffer_usage, 0, m_cellErrors[i])){
            fprintf(stderr, "Failed to create the cell_errors buffer on the GPU.\n");
            return false;
        }
        if (!vkutil::createBuffer(*m_context, activeListSize, buffer_usage, 0, m_activeNodes[i])){
            fprintf(stderr, "Failed to create the active_nodes buffer on the GPU.\n");
            return false;
        }
        if (!vkutil::createBuffer(*m_context, activeListSize, buffer_usage, 0, m_parents[i])){
            fprintf(stderr, "Failed to create the parents buffer on the GPU.\n");
            return false;
        }
    }

    //scratch
    if (!vkutil::createBuffer(*m_context, tmpSize, buffer_usage, 0, m_tmp)){
        fprintf(stderr, "Failed to create the tmp buffer on the GPU.\n");
        return false;
    }
    if (!vkutil::createBuffer(*m_context, oldToNewSize, buffer_usage, 0, m_oldToNewScratch)){
        fprintf(stderr, "Failed to create the old_to_new_scratch buffer on the GPU.\n");
        return false;
    }

    //per-level atomic counters
    if (!vkutil::createBuffer(*m_context, counterSize, buffer_usage, 0, m_activeCount)){
        fprintf(stderr, "Failed to create the active_count buffer on the GPU.\n");
        return false;
    }
    if (!vkutil::createBuffer(*m_context, counterSize, buffer_usage, 0, m_oldToNewCount)){
        fprintf(stderr, "Failed to create the old_to_new_count buffer on the GPU.\n");
        return false;
    }

    const VkDeviceSize totalBytes = 2 * (2 * gridIntSize + gridFloatSize + 2 * activeListSize) + tmpSize + oldToNewSize + 2 * counterSize;
    std::cout << "SdfPruning allocated " << (double)totalBytes / 1000000000.0 << " GB of work buffers" << std::endl;  

    return true;
}

bool SdfPruning::createCullingPipeline(){
    VkDevice device = m_context->getDevice();

    //load shader
    std::vector<uint32_t> code; //code holding raw words
    if(!Pipeline::readSpirvFile(std::string(SHADER_DIR) + "/pruning/culling.comp.spv", code)){
        fprintf(stderr, "Failed to read SdfPruning spirv.\n");
        return false;
    }

    //create module (vulkans' object for a piece of shader code that has been handed to the driver but not compiled yet - a SPIR-V blob with a handle)
    VkShaderModule module = VK_NULL_HANDLE;
    if(!Pipeline::createVkShaderModule(m_context->getDevice(), code, module)){
        fprintf(stderr, "Failed to create SdfPruning module.\n");
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PruningPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pruningLayout) != VK_SUCCESS){
        m_pruningLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(device, module, nullptr);
        fprintf(stderr, "Failed to create SdfPruning pruning pipeline layout\n");
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
    pipelineInfo.layout = m_pruningLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pruningPipeline);

    //module is baked into the pipeline
    vkDestroyShaderModule(device, module, nullptr);
    if(result != VK_SUCCESS){
        m_pruningPipeline = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create culling compute pipeline\n");
        return false;
    }

    return true;

}