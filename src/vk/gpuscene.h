#pragma once

#include <volk.h>

#include "vk/utils/buffer.h"
#include "sceneUtils.h"
#include "sdf/bricks.h"

#include <cstdint>
#include <vector>

class VulkanContext;
class GpuTimer;

class GpuScene
{
public:
    GpuScene() = default;
    ~GpuScene()
    {
        destroy();
    }

    // GpuScene should be created once and passed by reference as needed,
    // so we should disable copies/moves
    GpuScene(const GpuScene&) = delete;
    GpuScene& operator=(const GpuScene&) = delete;
    GpuScene(GpuScene&&) = delete;
    GpuScene& operator=(GpuScene&&) = delete;

    bool init(
        VulkanContext& context,
        const sceneutil::FlatScene& flatScene,
        const bricksutil::BricksFile* bricks,
        uint32_t sdfMaterialIdx);
    void destroy();

    uint64_t getSceneAddresses() const
    {
        return m_sceneAddressesBuffer.deviceAddress;
    }

    VkAccelerationStructureKHR getTLAS() const
    {
        return m_TLAS.handle;
    }

    // Records this frame's SDF BLAS and TLAS builds, with the barriers that order them
    // against the previous frame's trace and this frame's trace. This must be called
    // before tracing if the scene contains SDFs.
    // gpuTimer records the GpuScope::SdfBlas and GpuScope::Tlas scopes.
    void recordSdfBuilds(VkCommandBuffer commandBuffer, GpuTimer& gpuTimer);

private:
    VulkanContext* m_context = nullptr;

    vkutil::Buffer m_vertexBuffer{};
    vkutil::Buffer m_indexBuffer{};
    vkutil::Buffer m_geometryBuffer{};
    vkutil::Buffer m_materialBuffer{};
    vkutil::Buffer m_lightBuffer{};
    vkutil::Buffer m_sceneAddressesBuffer{};

    // SDF buffers, only created when the scene has an SDF
    vkutil::Buffer m_aabbBuffer{};
    vkutil::Buffer m_sdfSceneAddressesBuffer{};

    struct AccelStructure
    {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        vkutil::Buffer buffer{};

        // The buffer device address of the acceleration structure,
        // which, unlike vkutil::Buffer.deviceAddress, must be queried
        // via vkGetAccelerationStructureDeviceAddressKHR
        VkDeviceAddress deviceAddress = 0;
    };

    // One TLAS for the scene, one BLAS per mesh, plus one AABB BLAS for the SDF bricks
    std::vector<AccelStructure> m_BLASes;
    AccelStructure m_sdfBLAS{};
    AccelStructure m_TLAS{};

    // Inputs to the per-frame SDF BLAS and TLAS builds. The geometry descriptions are
    // filled once at load and shared by each structure's size query and every build.
    uint32_t m_sdfBrickCount = 0;
    VkAccelerationStructureGeometryKHR m_sdfBlasGeometry{};
    vkutil::Buffer m_tlasInstanceBuffer{};
    uint32_t m_tlasInstanceCount = 0;
    VkAccelerationStructureGeometryKHR m_tlasGeometry{};

    // Shared by the per-frame SDF BLAS and TLAS builds, which run one after the other
    vkutil::Buffer m_scratchBuffer{};

    // Note: Each acceleration structure's size query and build must use the same flags,
    // so each kind of structure gets one constant.
    // 
    // Triangle mesh BLASes are built once at load, so they favor trace speed
    static constexpr VkBuildAccelerationStructureFlagsKHR kMeshBlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    // The SDF BLAS favors build speed, since it will be rebuilt every frame
    static constexpr VkBuildAccelerationStructureFlagsKHR kSdfBlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;

    static constexpr VkBuildAccelerationStructureFlagsKHR kTlasBuildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    bool createBLASes(const sceneutil::FlatScene& flatScene);

    // These create the SDF BLAS and TLAS without building them (recordSdfBuilds builds them
    // every frame), and return the scratch size a build needs
    bool createSdfBLAS(uint32_t brickCount, VkDeviceSize& scratchSize);
    bool createTLAS(const sceneutil::FlatScene& flatScene, uint32_t sdfMaterialIdx, VkDeviceSize& scratchSize);

    // Note: geometries and maxPrimitiveCounts each point to geometryCount elements
    bool createAccelStructure(
        VkAccelerationStructureTypeKHR type,
        uint32_t geometryCount,
        const VkAccelerationStructureGeometryKHR* geometries,
        const uint32_t* maxPrimitiveCounts,
        VkBuildAccelerationStructureFlagsKHR flags,
        AccelStructure& accelStructure,
        VkDeviceSize& scratchSize);

    // Note: geometries and buildRanges each point to geometryCount elements
    void recordAccelStructureBuild(
        VkCommandBuffer commandBuffer,
        VkAccelerationStructureTypeKHR type,
        uint32_t geometryCount,
        const VkAccelerationStructureGeometryKHR* geometries,
        const VkAccelerationStructureBuildRangeInfoKHR* buildRanges,
        VkBuildAccelerationStructureFlagsKHR flags,
        const AccelStructure& accelStructure,
        VkDeviceAddress scratchAddress);

    // Records a global memory barrier. Everything before it in submission order (including
    // earlier frames' command buffers on this queue) is ordered against everything after it
    static void recordMemoryBarrier(
        VkCommandBuffer commandBuffer,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess);

    void destroyAccelStructure(AccelStructure& accelStructure);

    bool createSceneDataBuffers(const sceneutil::FlatScene& flatScene);
    bool createSdfBuffers(const bricksutil::BricksFile& bricks);
    bool createSceneAddressesBuffer(uint32_t lightCount, float totalLightArea);
};