#pragma once

#include <volk.h>
#include <glm/glm.hpp>
#include <cstdint>

#include "gpu/sdf_shared.h"

class VulkanContext;

class BrickOverlay{
public:
    BrickOverlay() = default;
    ~BrickOverlay(){destroy(); }

    BrickOverlay(const BrickOverlay&) = delete;
    BrickOverlay& operator=(const BrickOverlay&) = delete;
    BrickOverlay(BrickOverlay&&) = delete;
    BrickOverlay& operator=(BrickOverlay&&) = delete;

    bool init(VulkanContext& context, VkFormat colorFormat);
    void destroy();

    //false when disabled or empty
    bool recordDrawCommands(VkCommandBuffer commandBuffer, VkExtent2D extent, uint64_t aabbAddr, uint32_t brickCount);

    void setCamera(const gpu::CameraParams& camera, VkExtent2D renderExtent);

    void setEnabled(bool enabled) {
        m_enabled = enabled;
    }
    bool isEnabled() const {
        return m_enabled;
    }
    void setColor(const glm::vec4& color) { 
        m_pushConstants.color = color; 
    }
    void setYParams(const glm::vec4& yParams) {
        m_pushConstants.yParams = yParams; 
    }
    void setColorMode(uint32_t colorMode){
        m_pushConstants.colorMode = colorMode;
    }

#if BRICK_OVERLAY_DEPTH
    //depth buffer and render size are fixed after startup
    void setDepthBuffer(uint64_t depthAddr, VkExtent2D renderExtent){
        m_pushConstants.depthAddr = depthAddr;
        m_pushConstants.renderWidth = renderExtent.width;
        m_renderExtent = renderExtent;
    }
#endif

private:
    static constexpr float kNear = 0.01f;
    static constexpr float kFar = 1000.0f;

    VulkanContext* m_context = nullptr;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    gpu::WireframePushConstants m_pushConstants{};

    bool m_enabled = false; //will toggle with imgui
    
    bool createPipeline(VkFormat colorFormat);

#if BRICK_OVERLAY_DEPTH
    VkExtent2D m_renderExtent{};
#endif
};