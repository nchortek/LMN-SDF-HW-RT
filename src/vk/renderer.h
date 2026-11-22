#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include "gui.h"
#include "gputimer.h"
#include "pipeline.h"
#include "gpu/shared.h"

#include <array>
#include <cstdint>
#include "brickoverlay.h"
#include "vk/utils/buffer.h"

struct GLFWwindow;
class VulkanContext;
class Swapchain;
class GpuScene;

class Renderer
{
public:
    Renderer() = default;
    ~Renderer()
    {
        destroy();
    }

    // Renderer should be created once and passed by reference as needed,
    // so we should disable copies/moves
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    bool init(VulkanContext& context, Swapchain& swapchain, GLFWwindow* window, uint32_t width, uint32_t height, GpuScene& gpuScene);
    void destroy();
    bool drawFrame();

    Gui& getGui()
    {
        return m_gui;
    }

    BrickOverlay& getBrickOverlay(){
        return m_brickOverlay;
    }

    const GpuTimer& getGpuTimer() const
    {
        return m_gpuTimer;
    }

    void setCameraParams(const gpu::CameraParams& camera)
    {
        m_pushConstants.camera = camera;
        m_brickOverlay.setCamera(camera, m_renderExtent);
    }

    void setDepthOfField(bool enabled)
    {
        if (enabled)
        {
            m_pushConstants.flags |= gpu::kFlagDepthOfField;
        }
        else
        {
            m_pushConstants.flags &= ~gpu::kFlagDepthOfField;
        }
    }

    void setDebugView(bool enabled)
    {
        if (enabled)
        {
            m_pushConstants.flags |= gpu::kFlagDebugView;
        }
        else
        {
            m_pushConstants.flags &= ~gpu::kFlagDebugView;
        }
    }

    void setDirectLightingMode(bool enabled)
    {
        if (enabled)
        {
            m_pushConstants.flags |= gpu::kFlagDirectLighting;
        }
        else
        {
            m_pushConstants.flags &= ~gpu::kFlagDirectLighting;
        }
    }

    void setSolidFaceTest(bool enabled)
    {
        if (enabled)
        {
            m_pushConstants.flags |= gpu::kFlagSolidFaceTest;
        }
        else
        {
            m_pushConstants.flags &= ~gpu::kFlagSolidFaceTest;
        }
    }

    void setMaxTraceDepth(uint32_t maxDepth)
    {
        m_pushConstants.maxDepth = maxDepth;
    }

    uint32_t getRenderedFrameCount() const
    {
        return m_pushConstants.renderedFrameCount;
    }

    void resetRenderedFrameCount()
    {
        m_pushConstants.renderedFrameCount = 0;
    }

private:
    static constexpr uint32_t kFramesInFlight = 2;
    static constexpr uint64_t kMaxTimeout = UINT64_MAX;

    VulkanContext* m_context = nullptr;
    Swapchain* m_swapchain = nullptr;
    GpuScene* m_gpuScene = nullptr;
    Pipeline m_pipeline;
    Gui m_gui;
    BrickOverlay m_brickOverlay;
    GpuTimer m_gpuTimer;

    gpu::PushConstants m_pushConstants{};

    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    struct FrameData
    {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        VkFence inFlightFence = VK_NULL_HANDLE;
    };

    std::array<FrameData, kFramesInFlight> m_frames{};
    uint32_t m_frameInFlight = 0;

    struct StorageImage
    {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    StorageImage m_accumImage{};
    StorageImage m_displayImage{};
    VkExtent2D m_renderExtent{};

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_rendererDescriptorSet = VK_NULL_HANDLE;

    bool createCommandPool();
    bool createCommandBuffers();
    bool createSyncObjects();
    bool createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView& imageView);

    bool createStorageImages();
    bool createStorageImage(VkFormat format, VkImageUsageFlags usage, StorageImage& storageImage);
    void destroyStorageImage(StorageImage& storageImage);

#if BRICK_OVERLAY_DEPTH
    vkutil::Buffer m_overlayDepthBuffer{};
    bool createOverlayDepthBuffer();
#endif

    bool createDescriptorPool();
    bool createRendererDescriptorSet(VkAccelerationStructureKHR TLAS);

    bool recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex);

    static void recordImageBarrier(
        VkCommandBuffer commandBuffer,
        VkImage image,
        VkImageLayout oldImageLayout,
        VkImageLayout newImageLayout,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess);
};