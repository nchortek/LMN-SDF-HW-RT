#pragma once

#include "sceneStructs.h"
#include <vector>
#include "sdf/csg.h"

class Scene
{
public:
    Scene(std::string filename);

    std::vector<MeshData> meshes;
    std::vector<MeshInstance> meshInstances;

    // NCHORTEK TODO: I dont think we need geoms anymore
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    // Whether or not the scene contains SDF data
    bool hasSdf = false;
    bool hasCsg = false;

    glm::vec3 paddedAABBmin = glm::vec3(0.0);
    glm::vec3 paddedAABBmax = glm::vec3(0.0);
    
    int gpuRootIdx = -1;
    int nodeCount = -1;

    std::vector<GPUNode> gpuNodeList;
    std::vector<Primitive> primitiveList;
    std::vector<BinaryOp> binaryOpList;

    std::vector<uint16_t> parents;
    std::vector<uint16_t> activeNodes;

    // The .bricks filepath (The JSON file contains a filepath relative to the scene
    // JSON's folder, but we store it here relative to the working directory)
    std::string sdfBricksPath;

    // The index of the SDF material in the Scene::materials vector
    uint32_t sdfMaterialIdx = 0;

private:
    static constexpr uint32_t kCubeMeshIndex = 0;
    static constexpr uint32_t kSphereMeshIndex = 1;
    static constexpr uint32_t kIcosphereSubdivisions = 4;


    void loadFromJSON(const std::string& jsonName);
};
