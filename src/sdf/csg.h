#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "gpu/pruning_constants.h"

enum PrimitiveType {
    PRIMITIVE_SPHERE = 0,
    PRIMITIVE_BOX = 1,
    PRIMITIVE_CYLINDER = 2,
    PRIMITIVE_CONE = 3
};

struct SphereData {
    glm::vec4 radius;
};

struct BoxData {
    glm::vec4 sizes;
};

struct CylinderData {
    float height;
    float radius;
    float pad0;
    float pad1;
};

struct ConeData {
    float radius;
    float height;
    float pad0, pad1;
};

struct Primitive {
    union {
        SphereData sphere;
        BoxData box;
        CylinderData cylinder;
        ConeData cone;
    };
    glm::vec4 m_row0;
    glm::vec4 m_row1;
    glm::vec4 m_row2;
    glm::vec2 extrude_rounding;
    PrimitiveType type;
    float bevel;
    uint32_t color;

    float pad0, pad1, pad2;
};
static_assert(sizeof(Primitive) == 6*16);

struct BinaryOp {
    uint32_t blend_factor_and_sign;

    BinaryOp() = default;
    // 0 = union, 1 = sub, 2 = inter
    BinaryOp(float k, bool sign, uint32_t op);
};

enum NodeType {
    NODETYPE_BINARY,
    NODETYPE_PRIMITIVE,
};

struct CSGNode {
    union {
        BinaryOp binary_op;
        Primitive primitive;
    };
    NodeType type;
    int left, right;
    bool sign;
};

struct GPUNode {
    NodeType type;
    int idx_in_type;
};
static_assert(sizeof(GPUNode) == 8);

//parses lipschitz csg scene json into a flat node list
void load_json(const char* path, std::vector<CSGNode>& nodes, glm::vec3& aabb_min, glm::vec3& aabb_max);
//flattens the cpu tree into gpu arrays the pruning shader reads
int ConvertToGPUTree(int root_idx, const std::vector<CSGNode>& csg_nodes, std::vector<GPUNode>& gpu_nodes, std::vector<Primitive>& primitives, std::vector<BinaryOp>& binary_ops, std::vector<uint16_t>& parent, std::vector<uint16_t>& active_nodes);
//load_json + pads the AABB to a cube as done in Track A's create_scene
int load_csg_scene(const std::string& path, std::vector<CSGNode>& nodes, glm::vec3& aabb_min, glm::vec3& aabb_max);