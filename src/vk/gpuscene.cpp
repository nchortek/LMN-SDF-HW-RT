#include "gpuscene.h"

#include "context.h"
#include "gputimer.h"
#include "pipeline.h"
#include "gpu/sdf_shared.h"
#include "sdf/pruning.h"
#include "sdf/baking.h"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

// Note: sdfMaterialIdx is the SDF's index into flatScene.materials, and is only used when 
// bricks is non-null (i.e. when the scene includes SDFs).
bool GpuScene::init(
    VulkanContext& context,
    const sceneutil::FlatScene& flatScene,
    uint32_t sdfMaterialIdx,
    SdfPruning* pruning,
    SdfBaking* baking)
{
    m_context = &context;

    if (!createSceneDataBuffers(flatScene))
{
        destroy();
        return false;
    }

    // SDF data comes from the GPU bake (CSG scenes) OR a .bricks file
    uint32_t sdfBrickCount = 0;
    bool hasSdfBricks = false;

    // Create the empty buffers then bake!
    if (!createEmptySdfBuffers(*pruning) ||
        !baking->runBaking(m_sdfSceneAddressesBuffer.deviceAddress, m_brickCountBuffer, m_aabbBuffer, sdfBrickCount))
    {
        destroy();
        return false;
    }
    printf("SdfBaking: baked %u bricks\n", sdfBrickCount);
    hasSdfBricks = true;


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

    VkDeviceSize sdfBlasScratchSize = 0;
    if (hasSdfBricks && !createSdfBLAS(sdfBrickCount, sdfBlasScratchSize))
    {
        destroy();
        return false;
    }

    VkDeviceSize tlasScratchSize = 0;
    if (!createTLAS(flatScene, sdfMaterialIdx, tlasScratchSize))
    {
        destroy();
        return false;
    }

    // The per-frame SDF BLAS and TLAS builds run one after the other, so one scratch buffer
    // sized for the larger of the two serves both. It's created with the required
    // driver-defined alignment.
    if (!vkutil::createBuffer(
        *m_context,
        std::max(sdfBlasScratchSize, tlasScratchSize),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_context->getAccelProperties().minAccelerationStructureScratchOffsetAlignment,
        m_scratchBuffer))
    {
        fprintf(stderr, "Failed to create the per-frame acceleration structure scratch buffer.\n");
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

    vkutil::destroyBuffer(*m_context, m_tlasInstanceBuffer);
    vkutil::destroyBuffer(*m_context, m_scratchBuffer);

    vkutil::destroyBuffer(*m_context, m_vertexBuffer);
    vkutil::destroyBuffer(*m_context, m_indexBuffer);
    vkutil::destroyBuffer(*m_context, m_geometryBuffer);
    vkutil::destroyBuffer(*m_context, m_materialBuffer);
    vkutil::destroyBuffer(*m_context, m_lightBuffer);
    vkutil::destroyBuffer(*m_context, m_sceneAddressesBuffer);
    vkutil::destroyBuffer(*m_context, m_aabbBuffer);
    vkutil::destroyBuffer(*m_context, m_mortonBuffer);
    vkutil::destroyBuffer(*m_context, m_sampleBuffer);
    vkutil::destroyBuffer(*m_context, m_brickCountBuffer);
    vkutil::destroyBuffer(*m_context, m_sdfSceneAddressesBuffer);
    vkutil::destroyBuffer(*m_context, m_brickBakeParamsBuffer);

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

    // The Morton and sample buffers are sized via gpu::MAX_BRICKS, so baking can write
    // up to MAX_BRICKS bricks each frame, but only the first brickCount entries hold
    // real data
    std::vector<uint32_t> mortons = bricks.mortons;
    mortons.resize(gpu::MAX_BRICKS, 0);

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        mortons.data(),
        mortons.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_mortonBuffer))
    {
        fprintf(stderr, "Failed to create the SDF Morton index buffer on the GPU.\n");
        return false;
    }

    // NCHORTEK TODO: This samples vector and the memcpy for-loop can both be deleted after
    // we no longer rely on the .bricks file
    std::vector<uint32_t> samples(size_t(gpu::MAX_BRICKS) * gpu::BRICK_STRIDE_WORDS, 0);
    for (uint32_t i = 0; i < bricks.brickCount; i++)
    {
        memcpy(samples.data() + size_t(i) * gpu::BRICK_STRIDE_WORDS,
            bricks.samples.data() + size_t(i) * gpu::BRICK_SAMPLES,
            gpu::BRICK_SAMPLES);
    }

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        samples.data(),
        samples.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_sampleBuffer))
    {
        fprintf(stderr, "Failed to create the SDF sample buffer on the GPU.\n");
        return false;
    }

    // NCHORTEK TODO: We'll need to update all these references to bricks.* once
    // we no longer use the .bricks file
    if (!vkutil::createAndUploadBuffer(
        *m_context,
        &bricks.brickCount,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_brickCountBuffer))
    {
        fprintf(stderr, "Failed to create the SDF brick count buffer on the GPU.\n");
        return false;
    }

    gpu::SdfSceneAddresses sdfSceneAddresses{};
    sdfSceneAddresses.aabbsAddr = m_aabbBuffer.deviceAddress;
    sdfSceneAddresses.mortonsAddr = m_mortonBuffer.deviceAddress;
    sdfSceneAddresses.samplesAddr = m_sampleBuffer.deviceAddress;
    sdfSceneAddresses.brickCountAddr = m_brickCountBuffer.deviceAddress;
    sdfSceneAddresses.domainMin = glm::vec3(bricks.domainMin[0], bricks.domainMin[1], bricks.domainMin[2]);
    sdfSceneAddresses.voxelSize = bricks.voxelSize;
    sdfSceneAddresses.band = bricks.band;

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

bool GpuScene::createEmptySdfBuffers(SdfPruning& pruning)
{ 
    // Initialize sdf buffers
    if (!vkutil::createBuffer(*m_context, 
        VkDeviceSize(gpu::MAX_BRICKS) * sizeof(gpu::Aabb), 
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        0, 
        m_aabbBuffer
    )) {
        fprintf(stderr, "Failed to create the SDF AABB buffer on the GPU.\n");
        return false;
    }
    
    if (!vkutil::createBuffer(*m_context, 
        VkDeviceSize(gpu::MAX_BRICKS) * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        0,
        m_mortonBuffer
    )) {
        fprintf(stderr, "Failed to create the SDF morton buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createBuffer(*m_context, 
        VkDeviceSize(gpu::MAX_BRICKS) * gpu::BRICK_STRIDE_WORDS * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        0,
        m_sampleBuffer
    )) {
        fprintf(stderr, "Failed to create the SDF sample buffer on the GPU.\n");
        return false;
    }

    if (!vkutil::createBuffer(*m_context, 
        sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        0,
        m_brickCountBuffer
    )) {
        fprintf(stderr, "Failed to create the SDF brick count buffer on the GPU.\n");
        return false;
    }

    // Upload BrickBakeParams: addresses of the pruning outputs the bake evaluates.
    // runPruning() has already completed (its own submit + wait), so these are ready to read.
    gpu::BrickBakeParams bakeParams{};
    bakeParams.primsAddr = pruning.getPrims().deviceAddress;
    bakeParams.binaryOpsAddr = pruning.getBinaryOps().deviceAddress;
    bakeParams.nodesAddr = pruning.getNodes().deviceAddress;
    bakeParams.activeNodesAddr = pruning.getFinalActiveNodes().deviceAddress;
    bakeParams.cellOffsetsAddr = pruning.getFinalCellOffsets().deviceAddress;
    bakeParams.cellNumActiveAddr = pruning.getFinalNumActive().deviceAddress;
    bakeParams.cellErrorsAddr = pruning.getFinalCellErrors().deviceAddress;
    bakeParams.brickCellsNumActiveAddr = pruning.getBrickNumActive().deviceAddress;
    bakeParams.maxBricks = static_cast<int>(gpu::MAX_BRICKS);
    bakeParams.totalNumNodes = pruning.getNodeCount();

    if (!vkutil::createAndUploadBuffer(
        *m_context,
        &bakeParams,
        sizeof(bakeParams),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_brickBakeParamsBuffer))
    {
        fprintf(stderr, "Failed to create the brick bake params buffer on the GPU.\n");
        return false;
    }

    // Upload SdfSceneAddresses buffer
    glm::vec3 hi = pruning.getAabbMax();
    glm::vec3 lo = pruning.getAabbMin();
    gpu::SdfSceneAddresses sdfSceneAddresses{};
    sdfSceneAddresses.aabbsAddr = m_aabbBuffer.deviceAddress;
    sdfSceneAddresses.mortonsAddr = m_mortonBuffer.deviceAddress;
    sdfSceneAddresses.samplesAddr = m_sampleBuffer.deviceAddress;
    sdfSceneAddresses.brickCountAddr = m_brickCountBuffer.deviceAddress;
    sdfSceneAddresses.domainMin = lo;
    sdfSceneAddresses.voxelSize = (hi.x - lo.x) / gpu::VOXELS_PER_SCENE_AXIS;
    sdfSceneAddresses.band = 2.0f * std::sqrt(3.0f) * sdfSceneAddresses.voxelSize; // MMELK TODO: to helper?
    sdfSceneAddresses.brickBakeParamsAddr = m_brickBakeParamsBuffer.deviceAddress;

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
        const uint32_t geometryCount = static_cast<uint32_t>(geometries.size());
        VkDeviceSize scratchSize = 0;
        if (!createAccelStructure(
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            geometryCount,
            geometries.data(),
            maxPrimitiveCounts.data(),
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
            geometryCount,
            geometries.data(),
            buildRanges.data(),
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

bool GpuScene::createSdfBLAS(uint32_t brickCount, VkDeviceSize& scratchSize)
{
    m_sdfBrickCount = brickCount;

    // One AABB geometry over the whole AABB buffer. Each box is a primitive, so
    // gl_PrimitiveID in the intersection and hit shaders is the brick index.
    VkAccelerationStructureGeometryAabbsDataKHR aabbsData{};
    aabbsData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbsData.data.deviceAddress = m_aabbBuffer.deviceAddress;
    aabbsData.stride = sizeof(gpu::Aabb);

    m_sdfBlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    m_sdfBlasGeometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    m_sdfBlasGeometry.geometry.aabbs = aabbsData;
    m_sdfBlasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    // Sized via MAX_BRICKS, so later builds can use any brick count up to that maximum
    if (!createAccelStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        1,
        &m_sdfBlasGeometry,
        &gpu::MAX_BRICKS,
        kSdfBlasBuildFlags,
        m_sdfBLAS,
        scratchSize))
    {
        fprintf(stderr, "Failed to create the SDF BLAS.\n");
        return false;
    }

    return true;
}

bool GpuScene::createTLAS(const sceneutil::FlatScene& flatScene, uint32_t sdfMaterialIdx, VkDeviceSize& scratchSize)
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

    // The instance buffer is uploaded once and kept, since every frame's TLAS build reads it.
    // The upload waits idle, so it's complete before the first build.
    if (!vkutil::createAndUploadBuffer(
        *m_context,
        instances.data(),
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        m_tlasInstanceBuffer))
    {
        fprintf(stderr, "Failed to create the TLAS instance buffer.\n");
        return false;
    }

    // primitiveCount counts whatever the geometry type's unit is.
    // For VK_GEOMETRY_TYPE_TRIANGLES_KHR its triangles, for VK_GEOMETRY_TYPE_AABBS_KHR
    // its AABBs (one per brick), and for VK_GEOMETRY_TYPE_INSTANCES_KHR its
    // VkAccelerationStructureInstanceKHR entries (one per MeshInstance, plus one for the SDF)
    m_tlasInstanceCount = static_cast<uint32_t>(instances.size());

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = m_tlasInstanceBuffer.deviceAddress;

    m_tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    m_tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    m_tlasGeometry.geometry.instances = instancesData;

    if (!createAccelStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        1,
        &m_tlasGeometry,
        &m_tlasInstanceCount,
        kTlasBuildFlags,
        m_TLAS,
        scratchSize))
    {
        fprintf(stderr, "Failed to create the TLAS.\n");
        return false;
    }

    return true;
}

void GpuScene::recordSdfBuilds(VkCommandBuffer commandBuffer, GpuTimer& gpuTimer)
{
    // Builds read and write both the structure and the scratch buffer through these accesses
    const VkPipelineStageFlags2 buildStage = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    const VkPipelineStageFlags2 traceStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    const VkAccessFlags2 accelRead = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    const VkAccessFlags2 accelWrite = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    // The previous frame's trace may still be reading the SDF BLAS and TLAS, and its
    // builds may still be using the scratch buffer, so both must finish before we rebuild
    recordMemoryBarrier(
        commandBuffer,
        traceStage | buildStage,
        accelRead | accelWrite,
        buildStage,
        accelRead | accelWrite);

    // The SDF BLAS scope is recorded even without an SDF, where it measures roughly zero
    gpuTimer.begin(commandBuffer, GpuScope::SdfBlas);

    if (m_sdfBLAS.handle != VK_NULL_HANDLE)
    {
        // For VK_GEOMETRY_TYPE_AABBS_KHR, a primitive is one AABB (one brick)
        VkAccelerationStructureBuildRangeInfoKHR sdfBuildRange{};
        sdfBuildRange.primitiveCount = m_sdfBrickCount;

        recordAccelStructureBuild(
            commandBuffer,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            1,
            &m_sdfBlasGeometry,
            &sdfBuildRange,
            kSdfBlasBuildFlags,
            m_sdfBLAS,
            m_scratchBuffer.deviceAddress);

        // The TLAS build reads the fresh SDF BLAS and reuses the scratch buffer
        recordMemoryBarrier(
            commandBuffer,
            buildStage,
            accelWrite,
            buildStage,
            accelRead | accelWrite);
    }

    gpuTimer.end(commandBuffer, GpuScope::SdfBlas);

    // The mesh BLASes were built at load through endSingleTimeCommands, which waits idle,
    // so they're complete before any frame's TLAS build reads them
    VkAccelerationStructureBuildRangeInfoKHR tlasBuildRange{};
    tlasBuildRange.primitiveCount = m_tlasInstanceCount;

    gpuTimer.begin(commandBuffer, GpuScope::Tlas);

    recordAccelStructureBuild(
        commandBuffer,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        1,
        &m_tlasGeometry,
        &tlasBuildRange,
        kTlasBuildFlags,
        m_TLAS,
        m_scratchBuffer.deviceAddress);

    gpuTimer.end(commandBuffer, GpuScope::Tlas);

    // Rays must not traverse the TLAS until its build is complete
    recordMemoryBarrier(
        commandBuffer,
        buildStage,
        accelWrite,
        traceStage,
        accelRead);
}

// Creates an acceleration structure on the device.
// maxPrimitiveCounts (one per geometry) defines the size of the structure, allowing
// a later build to optionally use fewer primitives. scratchSize returns the scratch
// buffer side needed to build the acceleration structure.
bool GpuScene::createAccelStructure(
    VkAccelerationStructureTypeKHR accelStructType,
    uint32_t geometryCount,
    const VkAccelerationStructureGeometryKHR* geometries,
    const uint32_t* maxPrimitiveCounts,
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
    buildInfo.geometryCount = geometryCount;
    buildInfo.pGeometries = geometries;

    // Query the size of the acceleration structure and scratch buffer
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vkGetAccelerationStructureBuildSizesKHR(
        device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        maxPrimitiveCounts,
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
    uint32_t geometryCount,
    const VkAccelerationStructureGeometryKHR* geometries,
    const VkAccelerationStructureBuildRangeInfoKHR* buildRanges,
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
    buildInfo.geometryCount = geometryCount;
    buildInfo.pGeometries = geometries;
    buildInfo.scratchData.deviceAddress = scratchAddress;

    // vkCmdBuildAccelerationStructuresKHR takes one build-range array per build info
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, &buildRanges);
}

void GpuScene::recordMemoryBarrier(
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

void GpuScene::destroyAccelStructure(AccelStructure& accelStructure)
{
    vkDestroyAccelerationStructureKHR(m_context->getDevice(), accelStructure.handle, nullptr);
    vkutil::destroyBuffer(*m_context, accelStructure.buffer);
    accelStructure = {};
}