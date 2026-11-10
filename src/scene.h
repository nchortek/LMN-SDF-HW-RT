#pragma once

#include "sceneStructs.h"
#include <vector>

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
