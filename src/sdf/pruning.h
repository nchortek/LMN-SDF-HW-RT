#pragma once

#include "vk/utils/buffer.h"

#include <glm/glm.hpp>
#include <cstdint>

class VulkanContext;
class Scene;

struct PruningPushConstants {
    //glm::mat4 world_to_clip;
    glm::vec4 aabb_min;
    glm::vec4 aabb_max;
    glm::ivec2 resolution;
    uint64_t prims_ref;
    uint64_t binary_ops_ref;
    uint64_t nodes_ref;
    uint64_t parents_in_ref;
    uint64_t parents_out_ref;
    uint64_t active_nodes_in_ref;
    uint64_t active_nodes_out_ref;
    uint64_t cell_offsets_in_ref;
    uint64_t cell_offsets_out_ref;
    uint64_t num_active_in_ref;
    uint64_t num_active_out_ref;
    uint64_t active_count_ref;
    uint64_t cell_error_in_ref;
    uint64_t cell_error_out_ref;
    uint64_t old_to_new_scratch_ref;
    uint64_t old_to_new_count_ref;
    uint64_t tmp_ref;
    uint64_t mvp_ref = 0;
    uint64_t cam_ref = 0;
    int num_nodes;
    int grid_size;
    int first_lvl;
    float max_rel_err = 0;
    float viz_max;
    float alpha;
    int culling_enabled;
    float gamma;
    int num_samples;
};
static_assert(sizeof(PruningPushConstants) == 232);

//referenced from gpuscene.h
//in the original lipschitz repo, the pruning buffers were just fields of one giant RenderData struct in utils.h, and had their own helpers and also contain a lot of unnecessary members
class SdfPruning{
public:
    //capacities for pruning scratch buffers (lipschitz defaults)
    //shader has no overflow guard -> check counters if you want to lower
    static constexpr uint64_t maxActiveCount = 100000000;
    static constexpr uint64_t maxTmpCount = 400000000;

    //final pruning grid is 256 cells per axis (4^3 > 16^3 > 64^3 > 256^3)
    static constexpr int finalGridLvl = 8;
    static constexpr uint64_t numCells = 1ull << (3*finalGridLvl);

    //per-level counters
    static constexpr int numLevelSlots = 10;

    SdfPruning() = default;
    ~SdfPruning()
    {
        destroy();
    }

    // SdfPruning should be created once and passed by reference as needed,
    // so we should disable copies/moves
    SdfPruning(const SdfPruning&) = delete;
    SdfPruning& operator=(const SdfPruning&) = delete;
    SdfPruning(SdfPruning&&) = delete;
    SdfPruning& operator=(SdfPruning&&) = delete;

    bool init(VulkanContext& context, const Scene& scene);
    void destroy();

    bool runPruning();

private:
    VulkanContext* m_context = nullptr;

    int m_nodeCount = 0;
    int m_gpuRootIdx = -1;
    glm::vec3 m_aabbMin = glm::vec3(0.0f);
    glm::vec3 m_aabbMax = glm::vec3(0.0f);

    //input buffers, uploaded once from scene
    vkutil::Buffer m_nodes{};
    vkutil::Buffer m_binaryOps{};
    vkutil::Buffer m_prims{};
    vkutil::Buffer m_parentsInit{};
    vkutil::Buffer m_activeNodesInit{};

    //pingpong
    vkutil::Buffer m_numActive[2]{};
    vkutil::Buffer m_cellOffsets[2]{};
    vkutil::Buffer m_cellErrors[2]{};
    vkutil::Buffer m_activeNodes[2]{};
    vkutil::Buffer m_parents[2]{};

    //scartch
    vkutil::Buffer m_tmp{};
    vkutil::Buffer m_oldToNewScratch{};

    //atomic counters per level
    vkutil::Buffer m_activeCount{};
    vkutil::Buffer m_oldToNewCount{};

    bool createInputBuffers(const Scene& scene);
    bool createWorkBuffers();
    bool createCullingPipeline();

    VkPipelineLayout m_pruningLayout = VK_NULL_HANDLE;
    VkPipeline m_pruningPipeline = VK_NULL_HANDLE;

    int m_inputIdx = 0;
    int m_outputIdx = 1;

    PruningPushConstants m_pushConstants{};

    void setStaticPushConstants();
    void setLevelPushConstants(int lvl, bool firstLvl);

    void recordMemoryBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags2 srcStage,
    VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,
    VkAccessFlags2 dstAccess);
};