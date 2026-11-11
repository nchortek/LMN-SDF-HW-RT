#include "gputimer.h"

#include "context.h"

#include <algorithm>
#include <cstdio>

bool GpuTimer::init(VulkanContext& context, uint32_t framesInFlight)
{
    m_context = &context;

    // Timestamps can only be written on a queue whose family reports valid timestamp bits
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context.getPhysicalDevice(), &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(context.getPhysicalDevice(), &queueFamilyCount, queueFamilies.data());

    const uint32_t validBits = queueFamilies.at(context.getGraphicsQueueFamily()).timestampValidBits;
    if (validBits == 0)
    {
        fprintf(stderr, "The graphics queue doesn't support timestamps, so GPU timing is unavailable.\n");
        destroy();
        return false;
    }

    m_timestampMask = (validBits >= 64) ? ~0ull : ((1ull << validBits) - 1);
    m_timestampPeriod = context.getTimestampPeriod();

    VkQueryPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount = kQueryCount;

    m_queryPools.resize(framesInFlight, VK_NULL_HANDLE);
    m_slotWritten.resize(framesInFlight, false);

    for (VkQueryPool& queryPool : m_queryPools)
    {
        if (vkCreateQueryPool(context.getDevice(), &createInfo, nullptr, &queryPool) != VK_SUCCESS)
        {
            queryPool = VK_NULL_HANDLE;
            fprintf(stderr, "Failed to create a GPU timer query pool.\n");
            destroy();
            return false;
        }
    }

    return true;
}

void GpuTimer::destroy()
{
    if (m_context == nullptr)
    {
        return;
    }

    for (VkQueryPool queryPool : m_queryPools)
    {
        vkDestroyQueryPool(m_context->getDevice(), queryPool, nullptr);
    }

    m_queryPools.clear();
    m_slotWritten.clear();
    m_samplesMs = {};
    m_nextSample = 0;
    m_sampleCount = 0;
    m_context = nullptr;
}

void GpuTimer::beginFrame(VkCommandBuffer commandBuffer, uint32_t frameSlot)
{
    m_currentSlot = frameSlot;
    vkCmdResetQueryPool(commandBuffer, m_queryPools.at(frameSlot), 0, kQueryCount);
    m_slotWritten.at(frameSlot) = true;
}

void GpuTimer::begin(VkCommandBuffer commandBuffer, GpuScope scope)
{
    // ALL_COMMANDS writes the timestamp once every earlier command has finished
    vkCmdWriteTimestamp2(
        commandBuffer,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        m_queryPools.at(m_currentSlot),
        2 * static_cast<uint32_t>(scope));
}

void GpuTimer::end(VkCommandBuffer commandBuffer, GpuScope scope)
{
    vkCmdWriteTimestamp2(
        commandBuffer,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        m_queryPools.at(m_currentSlot),
        2 * static_cast<uint32_t>(scope) + 1);
}

void GpuTimer::collect(uint32_t frameSlot)
{
    // Skip a slot that has never been recorded, or whose results were already collected
    if (!m_slotWritten.at(frameSlot))
    {
        return;
    }

    m_slotWritten.at(frameSlot) = false;

    // No wait flag: the slot's fence has already signaled, so the results are available.
    // If they somehow aren't (e.g. a scope wasn't recorded), this frame's sample is
    // skipped rather than stalling.
    std::array<uint64_t, kQueryCount> ticks{};
    if (vkGetQueryPoolResults(
        m_context->getDevice(),
        m_queryPools.at(frameSlot),
        0,
        kQueryCount,
        sizeof(ticks),
        ticks.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
    {
        return;
    }

    for (uint32_t i = 0; i < kScopeCount; i++)
    {
        // Masking handles timestamps that wrap within their valid bits
        const uint64_t elapsedTicks = (ticks.at(2 * i + 1) - ticks.at(2 * i)) & m_timestampMask;

        // Convert from elapsed ticks to ms (timestampPeriod is in ns/tick)
        m_samplesMs.at(i).at(m_nextSample) = static_cast<double>(elapsedTicks) * m_timestampPeriod * kMsPerNs;
    }

    m_nextSample = (m_nextSample + 1) % kAverageFrameCount;
    m_sampleCount = std::min(m_sampleCount + 1, kAverageFrameCount);
}

double GpuTimer::getAverageMs(GpuScope scope) const
{
    if (m_sampleCount == 0)
    {
        return 0.0;
    }

    // Until the ring buffer fills, the unfilled samples are 0, so dividing by
    // m_sampleCount still gives the average of the collected frames
    double sumMs = 0.0;
    for (double sampleMs : m_samplesMs.at(static_cast<uint32_t>(scope)))
    {
        sumMs += sampleMs;
    }

    return sumMs / m_sampleCount;
}
