#pragma once

#include <volk.h>

#include "vk/utils/buffer.h"
#include "sceneUtils.h"
#include "sdf/bricks.h"

#include <cstdint>
#include <vector>

class VulkanContext;

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
    bool createSdfBLAS(uint32_t brickCount);
    bool createTLAS(const sceneutil::FlatScene& flatScene, uint32_t sdfMaterialIdx);

    bool createAccelStructure(
        VkAccelerationStructureTypeKHR type,
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<uint32_t>& maxPrimitiveCounts,
        VkBuildAccelerationStructureFlagsKHR flags,
        AccelStructure& accelStructure,
        VkDeviceSize& scratchSize);

    void recordAccelStructureBuild(
        VkCommandBuffer commandBuffer,
        VkAccelerationStructureTypeKHR type,
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& buildRanges,
        VkBuildAccelerationStructureFlagsKHR flags,
        const AccelStructure& accelStructure,
        VkDeviceAddress scratchAddress);

    void destroyAccelStructure(AccelStructure& accelStructure);

    bool createSceneDataBuffers(const sceneutil::FlatScene& flatScene);
    bool createSdfBuffers(const bricksutil::BricksFile& bricks);
    bool createSceneAddressesBuffer(uint32_t lightCount, float totalLightArea);
};