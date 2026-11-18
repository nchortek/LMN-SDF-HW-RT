#pragma once

#include "vk/utils/buffer.h"
#include "gpu/sdf_shared.h"

#include <cstdint>

class VulkanContext;

class SdfBaking{
public:
    SdfBaking() = default;
    ~SdfBaking()
    {
        destroy();
    }

    // SdfBaking should be created once and passed by reference as needed,
    // so we should disable copies/moves
    SdfBaking(const SdfBaking&) = delete;
    SdfBaking& operator=(const SdfBaking&) = delete;
    SdfBaking(SdfBaking&&) = delete;
    SdfBaking& operator=(SdfBaking&&) = delete;

    bool init(VulkanContext& context);
    void destroy();

    bool runBaking(uint64_t sdfSceneAddr, const vkutil::Buffer& brickCountBuffer, const vkutil::Buffer& aabbBuffer, uint32_t& outBrickCount);

private:
    VulkanContext* m_context = nullptr;

    VkPipelineLayout m_bakeLayout = VK_NULL_HANDLE;
    VkPipeline m_bakePipeline = VK_NULL_HANDLE;

    gpu::BakePushConstants m_pushConstants{};

    // Host-visible 4-byte buffer the brick count is copied into, created once in init
    VkBuffer m_countStaging = VK_NULL_HANDLE;
    VmaAllocation m_countStagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo m_countStagingInfo{};

    bool createBakingPipeline();
    bool createCountReadbackBuffer();

    void recordBake(VkCommandBuffer cmd, uint64_t sdfSceneAddr, const vkutil::Buffer& brickCountBuffer, const vkutil::Buffer& aabbBuffer);
    void recordCountReadback(VkCommandBuffer cmd, const vkutil::Buffer& brickCountBuffer);
    uint32_t readBrickCount();
};