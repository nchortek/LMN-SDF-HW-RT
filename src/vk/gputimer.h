#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <vector>

class VulkanContext;

// Every GPU-timed scope. Each scope must be recorded every frame.
// To add a scope, add it here (before Count) and add its name to GpuTimer::kScopeNames.
enum class GpuScope : uint32_t
{
    Frame,
    SdfBlas,
    Tlas,
    Trace,
    Count
};

// Measures GPU time for each GpuScope within each frame's command buffer.
// Each frame in flight gets its own timestamp query pool, so a slot's results are read
// back without stalling right after that slot's fence wait, before the slot is reused.
// Each scope's time is averaged over the last kAverageFrameCount collected frames.
class GpuTimer
{
public:
    static constexpr uint32_t kScopeCount = static_cast<uint32_t>(GpuScope::Count);
    static constexpr uint32_t kAverageFrameCount = 60;

    GpuTimer() = default;
    ~GpuTimer()
    {
        destroy();
    }

    // GpuTimer should be created once and passed by reference as needed,
    // so we should disable copies/moves
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    GpuTimer(GpuTimer&&) = delete;
    GpuTimer& operator=(GpuTimer&&) = delete;

    bool init(VulkanContext& context, uint32_t framesInFlight);
    void destroy();

    // Resets the frame slot's queries. Record it at the start of the slot's command buffer.
    void beginFrame(VkCommandBuffer commandBuffer, uint32_t frameSlot);

    // Write a scope's start and end timestamps into the current frame slot
    void begin(VkCommandBuffer commandBuffer, GpuScope scope);
    void end(VkCommandBuffer commandBuffer, GpuScope scope);

    // Reads the frame slot's results from its last recorded frame, if any.
    // Call it after waiting on the slot's fence, and before recording into the slot again.
    void collect(uint32_t frameSlot);

    static const char* getScopeName(GpuScope scope)
    {
        return kScopeNames.at(static_cast<uint32_t>(scope));
    }

    // Average over the collected frames (up to kAverageFrameCount), or 0 before the first one
    double getAverageMs(GpuScope scope) const;

private:
    // Display names, indexed by GpuScope
    static constexpr std::array<const char*, kScopeCount> kScopeNames = {
        "Frame",
        "SDF BLAS",
        "TLAS",
        "Trace"
    };

    // Each scope uses two queries per pool: its start and end timestamps
    static constexpr uint32_t kQueryCount = 2 * kScopeCount;

    static constexpr double kMsPerNs = 1.0e-6;

    VulkanContext* m_context = nullptr;

    // One pool per frame in flight
    std::vector<VkQueryPool> m_queryPools;
    std::vector<bool> m_slotWritten;
    uint32_t m_currentSlot = 0;

    // Nanoseconds per timestamp tick, and the mask of valid timestamp bits
    double m_timestampPeriod = 0.0;
    uint64_t m_timestampMask = 0;

    // A ring buffer of each scope's last kAverageFrameCount samples, indexed by GpuScope.
    // m_nextSample is the write position shared by all scopes.
    std::array<std::array<double, kAverageFrameCount>, kScopeCount> m_samplesMs{};
    uint32_t m_nextSample = 0;
    uint32_t m_sampleCount = 0;
};
