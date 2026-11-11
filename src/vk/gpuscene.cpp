#include "gpuscene.h"

#include "context.h"
#include "pipeline.h"
#include "gpu/sdf_shared.h"

#include <glm/gtc/type_ptr.hpp>

#include <limits>

// Note: sdfMaterialIdx is the SDF's index into flatScene.materials, and is only used when 
// bricks is non-null (i.e. when the scene includes SDFs).
bool GpuScene::init(
    VulkanContext& context,
    const sceneutil::FlatScene& flatScene,
    const bricksutil::BricksFile* bricks,
    uint32_t sdfMaterialIdx)
{
    m_context = &context;

    if (!createSceneDataBuffers(flatScene))
    {
        destroy();
        return false;
    }

    // The SDF buffers must exist before the scene addresses buffer, which stores their address
    if (bricks != nullptr && !createSdfBuffers(*bricks))
    {
        destroy();
        return false;
    }

    if (!createSceneAddressesBuffer(static_cast<uint32_t>(flatScene.lights.size()), flatScene.totalLightArea))
    {
        destroy();
        return false;
    }

    if (!createBLASes(flatScene))
    {
        destroy();
        return false;
    }

    if (bricks != nullptr && !createSdfBLAS(bricks->brickCount))
    {
        destroy();
        return false;
    }

    if (!createTLAS(flatScene, sdfMaterialIdx))
    {
        destroy();
        return false;
    }

    return true;
}

void GpuScene::destroy()
{
    if (m_context == nullptr)
    {
        return;
    }

    destroyAccelStructure(m_TLAS);

    for (AccelStructure& blas : m_BLASes)
    {
        destroyAccelStructure(blas);
    }

    m_BLASes.clear();

    destroyAccelStructure(m_sdfBLAS);

    vkutil::destroyBuffer(*m_context, m_vertexBuffer);
    vkutil::destroyBuffer(*m_context, m_indexBuffer);
    vkutil::destroyBuffer(*m_context, m_geometryBuffer);
    vkutil::destroyBuffer(*m_context, m_materialBuffer);
    vkutil::destroyBuffer(*m_context, m_lightBuffer);
    vkutil::destroyBuffer(*m_context, m_sceneAddressesBuffer);
    vkutil::destroyBuffer(*m_context, m_aabbBuffer);
    vkutil::destroyBuffer(*m_context, m_sdfSceneAddressesBuffer);

    m_context = nullptr;
}

bool GpuScene::createSceneDataBuffers(const sceneutil::FlatScene& flatScene)
{
    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.vertices.data(),
        flatScene.vertices.size() * sizeof(gpu::Vertex),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_vertexBuffer))
    {
        fprintf(stderr, "Failed to create the vertex buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.indices.data(),
        flatScene.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_indexBuffer))
    {
        fprintf(stderr, "Failed to create the index buffer on the GPU.\n");
        return false;
    }

    // A scene with only an SDF has no geometry rows, so it gets no geometry buffer
    // and geometriesAddr stays 0. The built-in meshes always exist, so the vertex
    // and index buffers are never empty.
    if (!flatScene.geometries.empty()
        && !vkutil::createAndUploadBuffer(
            *m_context,
            flatScene.geometries.data(),
            flatScene.geometries.size() * sizeof(gpu::GeometryInfo),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            m_geometryBuffer))
    {
        fprintf(stderr, "Failed to create the geometry buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        flatScene.materials.data(),
        flatScene.materials.size() * sizeof(gpu::GpuMaterial),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_materialBuffer))
    {
        fprintf(stderr, "Failed to create the material buffer on the GPU.\n");
        return false;
    }

    if (!flatScene.lights.empty()
        && !vkutil::createAndUploadBuffer(
            *m_context,
            flatScene.lights.data(),
            flatScene.lights.size() * sizeof(gpu::EmissiveTri),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            m_lightBuffer))
    {
        fprintf(stderr, "Failed to create the light buffer on the GPU.\n");
        return false;
    }

    return true;
}

bool GpuScene::createSdfBuffers(const bricksutil::BricksFile& bricks)
{
    // The AABB buffer is sized by MAX_BRICKS. The first N bricks contain real AABB data, 
    // followed by NaNs in the remaining unused primitive slots. Vulkan treats an AABB 
    // whose minX is NaN as an inactive primitive, which makes this safe.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<gpu::Aabb> aabbs = bricks.aabbs;
    aabbs.resize(
        gpu::MAX_BRICKS,
        gpu::Aabb{
            glm::vec3(nan),
            glm::vec3(nan)
        });

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        aabbs.data(),
        aabbs.size() * sizeof(gpu::Aabb),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_aabbBuffer))
    {
        fprintf(stderr, "Failed to create the SDF AABB buffer on the GPU.\n");
        return false;
    }

    gpu::SdfSceneAddresses sdfSceneAddresses{};
    sdfSceneAddresses.aabbsAddr = m_aabbBuffer.deviceAddress;

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        &sdfSceneAddresses,
        sizeof(sdfSceneAddresses),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_sdfSceneAddressesBuffer))
    {
        fprintf(stderr, "Failed to create the SDF scene addresses buffer on the GPU.\n");
        return false;
    }

    return true;
}

bool GpuScene::createSceneAddressesBuffer(uint32_t lightCount, float totalLightArea)
{
    gpu::SceneAddresses sceneAddresses{};
    sceneAddresses.verticesAddr = m_vertexBuffer.deviceAddress;
    sceneAddresses.indicesAddr = m_indexBuffer.deviceAddress;
    sceneAddresses.geometriesAddr = m_geometryBuffer.deviceAddress;
    sceneAddresses.materialsAddr = m_materialBuffer.deviceAddress;
    sceneAddresses.lightsAddr = m_lightBuffer.deviceAddress;
    sceneAddresses.sdfAddr = m_sdfSceneAddressesBuffer.deviceAddress;
    sceneAddresses.lightCount = lightCount;
    sceneAddresses.totalLightArea = totalLightArea;

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        &sceneAddresses,
        sizeof(sceneAddresses),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_sceneAddressesBuffer))
    {
        fprintf(stderr, "Failed to create the scene addresses buffer on the GPU.\n");
        return false;
    }

    return true;
}

bool GpuScene::createBLASes(const sceneutil::FlatScene& flatScene)
{
    m_BLASes.resize(flatScene.meshRanges.size());

    for (int meshIndex = 0; meshIndex < flatScene.meshRanges.size(); meshIndex++)
    {
        const std::vector<sceneutil::MeshPrimitiveRange>& primitiveRanges = flatScene.meshRanges.at(meshIndex);

        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        geometries.reserve(primitiveRanges.size());

        std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
        buildRanges.reserve(primitiveRanges.size());

        // For a TLAS a "primitive" is a mesh/BLAS instance. For a BLAS it's a triangle.
        std::vector<uint32_t> maxPrimitiveCounts;
        maxPrimitiveCounts.reserve(primitiveRanges.size());

        for (const sceneutil::MeshPrimitiveRange& primitiveRange : primitiveRanges)
        {
            // Describe our vertex data layout so the driver knows how to access the vertex
            // positions specifically
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
            triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triangles.vertexData.deviceAddress = m_vertexBuffer.deviceAddress;
            triangles.vertexStride = sizeof(gpu::Vertex);

            // maxVertex is the highest index the driver will read
            triangles.maxVertex = primitiveRange.vertexCount - 1;

            triangles.indexType = VK_INDEX_TYPE_UINT32;
            triangles.indexData.deviceAddress = m_indexBuffer.deviceAddress;

            // BLAS doesn't require geometry transforms (placement is a TLAS job)
            triangles.transformData = {};

            VkAccelerationStructureGeometryKHR geometry{};
            geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometry.geometry.triangles = triangles;

            // Opaque here means traversal never invokes an any-hit shader, which is
            // okay because we're only using a closest-hit shader for now
            geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

            VkAccelerationStructureBuildRangeInfoKHR buildRange{};
            buildRange.primitiveCount = primitiveRange.indexCount / 3;

            // primitiveOffset is in bytes
            buildRange.primitiveOffset = primitiveRange.indexOffset * sizeof(uint32_t);

            // firstVertex is an element count
            buildRange.firstVertex = primitiveRange.vertexOffset;
            buildRange.transformOffset = 0;

            geometries.push_back(geometry);
            buildRanges.push_back(buildRange);

            // Triangle BLASes are sized exactly for the triangles they're built with
            maxPrimitiveCounts.push_back(buildRange.primitiveCount);
        }

        AccelStructure& blas = m_BLASes.at(meshIndex);
        VkDeviceSize scratchSize = 0;
        if (!createAccelStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometries,
            maxPrimitiveCounts,
            kMeshBlasBuildFlags,
            blas,
            scratchSize))
        {
            fprintf(stderr, "Failed to create BLAS for mesh index %d.\n", meshIndex);
            return false;
        }

        // Create a temporary scratch buffer with the required driver-defined alignment
        vkutil::Buffer scratchBuffer{};
        if (!vkutil::createBuffer(
            *m_context,
            scratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            m_context->getAccelProperties().minAccelerationStructureScratchOffsetAlignment,
            scratchBuffer))
        {
            fprintf(stderr, "Failed to create the BLAS scratch buffer for mesh index %d.\n", meshIndex);
            return false;
        }

        // Build the BLAS with a one-off command buffer
        VkCommandBuffer commandBuffer = m_context->beginSingleTimeCommands();
        recordAccelStructureBuild(
            commandBuffer,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometries,
            buildRanges,
            kMeshBlasBuildFlags,
            blas,
            scratchBuffer.deviceAddress);
        m_context->endSingleTimeCommands(commandBuffer);

        // NCHORTEK TODO: this is only safe because endSingleTimeCommands currently performs
        // a queue-level waitIdle
        vkutil::destroyBuffer(*m_context, scratchBuffer);
    }

    return true;
}

bool GpuScene::createSdfBLAS(uint32_t brickCount)
{
    // One AABB geometry over the whole AABB buffer. Each box is a primitive, so
    // gl_PrimitiveID in the intersection and hit shaders is the brick index.
    VkAccelerationStructureGeometryAabbsDataKHR aabbsData{};
    aabbsData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbsData.data.deviceAddress = m_aabbBuffer.deviceAddress;
    aabbsData.stride = sizeof(gpu::Aabb);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs = aabbsData;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    // For VK_GEOMETRY_TYPE_AABBS_KHR, a primitive is one AABB (one brick)
    VkAccelerationStructureBuildRangeInfoKHR buildRange{};
    buildRange.primitiveCount = brickCount;

    const std::vector<VkAccelerationStructureGeometryKHR> geometries = { geometry };
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges = { buildRange };

    // Sized via MAX_BRICKS, so later builds can use any brick count up to that maximum
    VkDeviceSize scratchSize = 0;
    if (!createAccelStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        geometries,
        { gpu::MAX_BRICKS },
        kSdfBlasBuildFlags,
        m_sdfBLAS,
        scratchSize))
    {
        fprintf(stderr, "Failed to create the SDF BLAS.\n");
        return false;
    }

    // Create a temporary scratch buffer with the required driver-defined alignment
    vkutil::Buffer scratchBuffer{};
    if (!vkutil::createBuffer(
        *m_context,
        scratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_context->getAccelProperties().minAccelerationStructureScratchOffsetAlignment,
        scratchBuffer))
    {
        fprintf(stderr, "Failed to create the SDF BLAS scratch buffer.\n");
        return false;
    }

    // Build the SDF BLAS with a one-off command buffer
    VkCommandBuffer commandBuffer = m_context->beginSingleTimeCommands();
    recordAccelStructureBuild(
        commandBuffer,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        geometries,
        buildRanges,
        kSdfBlasBuildFlags,
        m_sdfBLAS,
        scratchBuffer.deviceAddress);
    m_context->endSingleTimeCommands(commandBuffer);

    // NCHORTEK TODO: this is only safe because endSingleTimeCommands currently performs
    // a queue-level waitIdle
    vkutil::destroyBuffer(*m_context, scratchBuffer);

    return true;
}

bool GpuScene::createTLAS(const sceneutil::FlatScene& flatScene, uint32_t sdfMaterialIdx)
{
    // VkTransformMatrixKHR is row-major 3x4, but glm::mat4 is column-major.
    // Transposing and copying sizeof(VkTransformMatrixKHR) drops the bottom row,
    // which is what VkTransformMatrixKHR expects
    auto toTransformMatrixKHR = [](const glm::mat4& matrix)
    {
        VkTransformMatrixKHR transform{};
        const glm::mat4 transposed = glm::transpose(matrix);
        memcpy(&transform, glm::value_ptr(transposed), sizeof(transform));
        return transform;
    };

    // FlatScene only holds triangle mesh data, so we add one additional instance
    // for the SDF BLAS
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(flatScene.instanceFirstGeometry.size() + 1);

    for (int i = 0; i < flatScene.instances.size(); i++)
    {
        const MeshInstance& meshInstance = flatScene.instances.at(i);

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform = toTransformMatrixKHR(meshInstance.modelMatrix);

        // instanceCustomIndex holds the instance's first row in geometries[]
        // The shader will read geometries[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT],
        // so instances sharing a BLAS can use different materials
        instance.instanceCustomIndex = flatScene.instanceFirstGeometry.at(i);

        instance.mask = gpu::kInstanceMaskAllBitsSet;

        instance.instanceShaderBindingTableRecordOffset = Pipeline::kTriangleHitSlot;

        // Backfaces must hit (both sides of a surface should be treated as valid intersections)
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        instance.accelerationStructureReference = m_BLASes.at(meshInstance.meshIndex).deviceAddress;

        instances.push_back(instance);
    }

    if (m_sdfBLAS.handle != VK_NULL_HANDLE)
    {
        VkAccelerationStructureInstanceKHR instance{};

        // The bricks' AABBs are already in world space, so the SDF uses an identity transform
        instance.transform = toTransformMatrixKHR(glm::mat4(1.0f));

        // instanceCustomIndex holds the SDF's material index, which sdf.rchit reads
        // as gl_InstanceCustomIndexEXT
        instance.instanceCustomIndex = sdfMaterialIdx;

        instance.mask = gpu::kInstanceMaskAllBitsSet;
        instance.instanceShaderBindingTableRecordOffset = Pipeline::kSdfHitSlot;
        instance.accelerationStructureReference = m_sdfBLAS.deviceAddress;

        instances.push_back(instance);
    }

    vkutil::Buffer instanceBuffer{};
    if (!vkutil::createAndUploadBuffer(
        *m_context,
        instances.data(),
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        instanceBuffer))
    {
        fprintf(stderr, "Failed to create the TLAS instance buffer.\n");
        return false;
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceBuffer.deviceAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildRangeInfoKHR buildRange{};

    // primitiveCount counts whatever the geometry type's unit is.
    // For VK_GEOMETRY_TYPE_TRIANGLES_KHR its triangles, for VK_GEOMETRY_TYPE_AABBS_KHR
    // its AABBs (one per brick), and for VK_GEOMETRY_TYPE_INSTANCES_KHR its
    // VkAccelerationStructureInstanceKHR entries (one per MeshInstance, plus one for the SDF)
    buildRange.primitiveCount = static_cast<uint32_t>(instances.size());

    const std::vector<VkAccelerationStructureGeometryKHR> geometries = { geometry };
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges = { buildRange };

    VkDeviceSize scratchSize = 0;
    if (!createAccelStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        geometries,
        { buildRange.primitiveCount },
        kTlasBuildFlags,
        m_TLAS,
        scratchSize))
    {
        fprintf(stderr, "Failed to create the TLAS.\n");
        vkutil::destroyBuffer(*m_context, instanceBuffer);
        return false;
    }

    // Create a temporary scratch buffer with the required driver-defined alignment
    vkutil::Buffer scratchBuffer{};
    if (!vkutil::createBuffer(
        *m_context,
        scratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_context->getAccelProperties().minAccelerationStructureScratchOffsetAlignment,
        scratchBuffer))
    {
        fprintf(stderr, "Failed to create the TLAS scratch buffer.\n");
        vkutil::destroyBuffer(*m_context, instanceBuffer);
        return false;
    }

    // NCHORTEK NOTE: The BLAS builds each ran through endSingleTimeCommands,
    // which waits idle, so they are complete before this build reads them.
    // If the wait idle is every removed, this will need sync handling
    VkCommandBuffer commandBuffer = m_context->beginSingleTimeCommands();
    recordAccelStructureBuild(
        commandBuffer,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        geometries,
        buildRanges,
        kTlasBuildFlags,
        m_TLAS,
        scratchBuffer.deviceAddress);
    m_context->endSingleTimeCommands(commandBuffer);

    // NCHORTEK TODO: this is only safe because endSingleTimeCommands currently performs
    // a queue-level waitIdle
    vkutil::destroyBuffer(*m_context, scratchBuffer);

    // Right now the instanceBuffer is only needed to build the TLAS.
    // Because we never update the TLAS, we can safely destroy it
    // after the TLAS is created
    vkutil::destroyBuffer(*m_context, instanceBuffer);

    return true;
}

// Creates an acceleration structure on the device.
// maxPrimitiveCounts (one per geometry) defines the size of the structure, allowing
// a later build to optionally use fewer primitives. scratchSize returns the scratch
// buffer side needed to build the acceleration structure.
bool GpuScene::createAccelStructure(
    VkAccelerationStructureTypeKHR accelStructType,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<uint32_t>& maxPrimitiveCounts,
    VkBuildAccelerationStructureFlagsKHR flags,
    AccelStructure& accelStruct,
    VkDeviceSize& scratchSize)
{
    VkDevice device = m_context->getDevice();

    // The size query only needs the type, flags and geometries. The scratch buffer
    // and destination accel struct are filled in when the build is recorded
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = accelStructType;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = static_cast<uint32_t>(geometries.size());
    buildInfo.pGeometries = geometries.data();

    // Query the size of the acceleration structure and scratch buffer
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        maxPrimitiveCounts.data(),
        &sizeInfo);

    // The driver builds its BVH into this buffer, so we dont have to
    // do it manually.
    if (!vkutil::createBuffer(
        *m_context,
        sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        0,
        accelStruct.buffer))
    {
        fprintf(stderr, "Failed to create acceleration structure storage buffer.\n");
        return false;
    }

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = accelStruct.buffer.bufferHandle;
    createInfo.offset = 0;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = accelStructType;

    if (vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &accelStruct.handle)
        != VK_SUCCESS)
    {
        accelStruct.handle = VK_NULL_HANDLE;
        fprintf(stderr, "Failed to create acceleration structure.\n");
        return false;
    }

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = accelStruct.handle;
    accelStruct.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

    scratchSize = sizeInfo.buildScratchSize;

    return true;
}

// Records an acceleration structure build command. The provided type, geometries and flags
// parameters must match the ones that the AccelStructure was created with.
void GpuScene::recordAccelStructureBuild(
    VkCommandBuffer commandBuffer,
    VkAccelerationStructureTypeKHR accelStructType,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& buildRanges,
    VkBuildAccelerationStructureFlagsKHR flags,
    const AccelStructure& accelStruct,
    VkDeviceAddress scratchAddress)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = accelStructType;
    buildInfo.flags = flags;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = accelStruct.handle;
    buildInfo.geometryCount = static_cast<uint32_t>(geometries.size());
    buildInfo.pGeometries = geometries.data();
    buildInfo.scratchData.deviceAddress = scratchAddress;

    const VkAccelerationStructureBuildRangeInfoKHR* buildRangesPtr = buildRanges.data();
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, &buildRangesPtr);
}

void GpuScene::destroyAccelStructure(AccelStructure& accelStructure)
{
    vkDestroyAccelerationStructureKHR(m_context->getDevice(), accelStructure.handle, nullptr);
    vkutil::destroyBuffer(*m_context, accelStructure.buffer);
    accelStructure = {};
}