#pragma once

#include "csg.h"
#include <queue>
#include <random>
#include <rapidjson/document.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h> 
#include <glm/gtx/string_cast.hpp>

#include "glm/gtx/transform.hpp"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cassert>


BinaryOp::BinaryOp(float k, bool sign, unsigned int op) {
    uint32_t x = (uint32_t)sign;
    op &= 3u;
    x |= (op << 1);

    uint32_t k_bits;

    static_assert(sizeof(float) == sizeof(uint32_t));
    memcpy(&k_bits, &k, sizeof(float));

    // clear lower 3 bits
    k_bits &= ~7u;
    x |= k_bits;

    blend_factor_and_sign = x;
}

void load_json(const char *path, std::vector<CSGNode> &nodes, glm::vec3& aabb_min, glm::vec3& aabb_max) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        abort();
    }
    fseek(fp, 0, SEEK_END);
    size_t s = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(s + 1);
    if (fread(buf.data(), 1, s, fp) != s) {
        fprintf(stderr, "Failed to read file: %s\n", path);
        abort();
    }
    fclose(fp);
    buf[s] = 0;
    rapidjson::Document d;
    d.Parse(buf.data());
    if (d.HasParseError()) {
        rapidjson::ParseErrorCode code = d.GetParseError();
        fprintf(stderr, "JSON parsing failed: error %d\n", code);
        abort();
    }

    auto aabb_min_arr = d["aabb_min"].GetArray();
    aabb_min.x = aabb_min_arr[0].GetFloat();
    aabb_min.y = aabb_min_arr[1].GetFloat();
    aabb_min.z = aabb_min_arr[2].GetFloat();
    auto aabb_max_arr = d["aabb_max"].GetArray();
    aabb_max.x = aabb_max_arr[0].GetFloat();
    aabb_max.y = aabb_max_arr[1].GetFloat();
    aabb_max.z = aabb_max_arr[2].GetFloat();

    struct StackEntry {
        const rapidjson::Value* j;
        glm::mat4 mat;
        int idx;
        bool sign;
    };

    nodes.push_back({});
    std::vector<StackEntry> stack = { { &d, glm::mat4(1), 0, true } };

    while (!stack.empty()) {
        StackEntry e = stack.back();
        stack.pop_back();

        const auto& j = *e.j;

        auto m = j["matrix"].GetArray();
        glm::mat4 node_mat = glm::mat4(
            m[0].GetFloat(), m[4].GetFloat(), m[8].GetFloat(), 0.f,
            m[1].GetFloat(), m[5].GetFloat(), m[9].GetFloat(), 0.f,
            m[2].GetFloat(), m[6].GetFloat(), m[10].GetFloat(), 0.f,
            m[3].GetFloat(), m[7].GetFloat(), m[11].GetFloat(), 1.f
        );
        glm::mat4 mat = e.mat;
        std::string type = j["nodeType"].GetString();
        int node_idx = e.idx;
        if (type == "primitive") {
            CSGNode &node = nodes[node_idx];
            node.type = NODETYPE_PRIMITIVE;
            node.left = -1;
            node.right = -1;
            node.sign = e.sign;

            glm::mat4 world_to_prim = mat * node_mat;
            node.primitive.m_row0 = glm::vec4(world_to_prim[0][0], world_to_prim[1][0], world_to_prim[2][0],
                                              world_to_prim[3][0]);
            node.primitive.m_row1 = glm::vec4(world_to_prim[0][1], world_to_prim[1][1], world_to_prim[2][1],
                                              world_to_prim[3][1]);
            node.primitive.m_row2 = glm::vec4(world_to_prim[0][2], world_to_prim[1][2], world_to_prim[2][2],
                                              world_to_prim[3][2]);

            node.primitive.extrude_rounding.x = j["round_x"].GetFloat();
            node.primitive.extrude_rounding.y = j["round_y"].GetFloat();

            {
                auto c = j["color"].GetArray();
                node.primitive.color = 0;
                node.primitive.color |= std::min((uint32_t)(c[0].GetFloat() * 255.99f), 255u) << 0;
                node.primitive.color |= std::min((uint32_t)(c[1].GetFloat() * 255.99f), 255u) << 8;
                node.primitive.color |= std::min((uint32_t)(c[2].GetFloat() * 255.99f), 255u) << 16;
            }

            std::string prim_type = j["primitiveType"].GetString();
            if (prim_type == "sphere") {
                node.primitive.type = PRIMITIVE_SPHERE;
            } else if (prim_type == "box") {
                node.primitive.type = PRIMITIVE_BOX;
            } else if (prim_type == "cylinder") {
                node.primitive.type = PRIMITIVE_CYLINDER;
            } else if (prim_type == "cone") {
                node.primitive.type = PRIMITIVE_CONE;
            } else {
                fprintf(stderr, "Unknown primitive: %s\n", prim_type.c_str());
                abort();
            }

            switch (node.primitive.type) {
                case PRIMITIVE_SPHERE: {
                    float r = j["radius"].GetFloat();
                    node.primitive.sphere.radius = glm::vec4(r, 0, 0, 0);
                    break;
                }
                case PRIMITIVE_BOX: {
                    auto s = j["sides"].GetArray();
                    float s_x = s[0].GetFloat();
                    float s_y = s[1].GetFloat();
                    float s_z = s[2].GetFloat();
                    node.primitive.box.sizes = glm::vec4(s_x, s_y, s_z, 0);

                    float scale = fmaxf(node.primitive.box.sizes.x, node.primitive.box.sizes.z) * 2;
                    auto c = j["bevel"].GetArray();
                    glm::vec4 corner_rounding;
                    corner_rounding.x = c[0].GetFloat();
                    corner_rounding.y = c[1].GetFloat();
                    corner_rounding.z = c[2].GetFloat();
                    corner_rounding.w = c[3].GetFloat();

    #if 1
                    uint32_t corner_data = 0;
                    corner_data |= (uint32_t)(corner_rounding.x * 255.f * 2.f / scale) << 0;
                    corner_data |= (uint32_t)(corner_rounding.y * 255.f * 2.f / scale) << 8;
                    corner_data |= (uint32_t)(corner_rounding.z * 255.f * 2.f / scale) << 16;
                    corner_data |= (uint32_t)(corner_rounding.w * 255.f * 2.f / scale) << 24;
                    memcpy(&node.primitive.box.sizes.w, &corner_data, sizeof(uint32_t));
    #endif
                    break;
                }
                case PRIMITIVE_CYLINDER: {
                    float h = j["height"].GetFloat();
                    float r = j["radius"].GetFloat();
                    node.primitive.cylinder.height = h;
                    node.primitive.cylinder.radius = r;
                    break;
                }
                case PRIMITIVE_CONE: {
                    float h = j["height"].GetFloat();
                    float r = j["radius"].GetFloat();
                    node.primitive.cone.height = h;
                    node.primitive.cone.radius = r;
                    break;
                }

                default:
                    abort();
            }
        } else if (type == "binaryOperator") {
            std::string blend_mode = j["blendMode"].GetString();
            uint32_t op;
            if (blend_mode == "union") {
                op = OP_UNION;
            } else if (blend_mode == "sub") {
                op = OP_SUB;
            } else if (blend_mode == "inter") {
                op = OP_INTER;
            } else {
                fprintf(stderr, "Unknown blend mode: %s\n", blend_mode.c_str());
                abort();
            }

            int left_idx = nodes.size();
            nodes.push_back({});
            int right_idx = nodes.size();
            nodes.push_back({});

            stack.push_back({ &j["leftChild"], mat * node_mat, left_idx, 1 });
            stack.push_back({ &j["rightChild"], mat * node_mat, right_idx, op != OP_SUB });

            CSGNode &node = nodes[node_idx];
            node.left = left_idx;
            node.right = right_idx;
            node.sign = e.sign;
            float k = j["blendRadius"].GetFloat();

            node.binary_op = BinaryOp(k, op == OP_UNION, op);
        } else {
            fprintf(stderr, "invalid type: %s\n", type.c_str());
            abort();
        }
    }
}

int ConvertToGPUTree(int root_idx, const std::vector<CSGNode>& csg_nodes, std::vector<GPUNode>& gpu_nodes, std::vector<Primitive>& primitives, std::vector<BinaryOp>& binary_ops, std::vector<uint16_t>& parent, std::vector<uint16_t>& active_nodes) {

    std::vector<int> cpu_to_gpu(csg_nodes.size());
    std::vector<int> stack = { root_idx };
    std::vector<int> preorder;
    while (!stack.empty()) {
        int current_idx = stack.back();
        stack.pop_back();

        preorder.push_back(current_idx);
        if (csg_nodes[current_idx].type == NODETYPE_BINARY) {
            stack.push_back(csg_nodes[current_idx].left);
            stack.push_back(csg_nodes[current_idx].right);
        }
    }
    assert(preorder.size() == csg_nodes.size());
    active_nodes.resize(csg_nodes.size());
    for (int i = (int)csg_nodes.size() - 1; i >= 0; i--) {
        int current_idx = preorder[i];
        const CSGNode& node = csg_nodes[current_idx];

        switch (node.type) {
            case NODETYPE_BINARY:
            {
                int gpu_left = cpu_to_gpu[node.left];
                int gpu_right = cpu_to_gpu[node.right];
                binary_ops.push_back(node.binary_op);
                GPUNode gpu_node = {
                        .type = node.type,
                        .idx_in_type = (int)binary_ops.size() - 1,
                };
                gpu_nodes.push_back(gpu_node);
                parent[gpu_left] = (uint16_t)gpu_nodes.size() - 1;
                parent[gpu_right] = (uint16_t)gpu_nodes.size() - 1;
                parent.push_back(0xffff);
                break;
            }
            case NODETYPE_PRIMITIVE:
            {
                primitives.push_back(node.primitive);
                GPUNode gpu_node = {
                        .type = node.type,
                        .idx_in_type = (int)primitives.size() - 1,
                };
                gpu_nodes.push_back(gpu_node);
                parent.push_back(0xffff);
                break;
            }
            default:
                abort();
        }
        uint32_t gpu_idx = gpu_nodes.size()-1;
        cpu_to_gpu[current_idx] = gpu_idx;
        active_nodes[gpu_idx] = gpu_idx;
        if (!node.sign) {
            active_nodes[gpu_idx] |= 1 << 15;
        }
    }

    return cpu_to_gpu[root_idx];
}

int load_csg_scene(const std::string& path, std::vector<CSGNode>& nodes, glm::vec3& aabb_min, glm::vec3& aabb_max){
    // Clear prev scene
    nodes.clear();

    // Load in new scene
    load_json(path.c_str(), nodes, aabb_min, aabb_max);

    // Pad to 512^3 for cubic voxels
    // No more non-cubic cells
    glm::vec3 extent = aabb_max - aabb_min;
    glm::vec3 center = 0.5f * (aabb_min + aabb_max);
    float half = 0.5f * std::max(extent.x, std::max(extent.y, extent.z));
    aabb_min = center - glm::vec3(half);
    aabb_max = center + glm::vec3(half);

    int root_idx = 0;
    return root_idx;
}