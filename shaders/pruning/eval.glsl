// CSG tree evaluation, ported from LMN-LipschitzPruning/shaders/eval.glsl.
#ifndef PRUNING_EVAL_GLSL
#define PRUNING_EVAL_GLSL

#define EVAL_STACK_DEPTH 128

// Full-tree SDF: evaluates every node in postfix order. Slow; used for validating the bake.
float sdf(vec3 p, BrickBakeParams params) {
    NodesRef nodes = NodesRef(params.nodesAddr);
    BinaryOpsRef binary_ops = BinaryOpsRef(params.binaryOpsAddr);
    PrimitivesRef prims = PrimitivesRef(params.primsAddr);

    float stack[EVAL_STACK_DEPTH];
    int stack_idx = 0;

    for (int i = 0; i < params.totalNumNodes; i++) {
        Node node = nodes.tab[i];

        float d;
        if (node.type == NODETYPE_BINARY) {
            float left_val = stack[stack_idx-2];
            float right_val = stack[stack_idx-1];
            BinaryOp op = binary_ops.tab[node.idx_in_type];
            float k = BinaryOp_blend_factor(op);
            float s = BinaryOp_sign(op);
            uint typ = BinaryOp_op(op);
            if (typ == OP_SUB) right_val *= -1;
            stack_idx -= 2;
            d = s*(min(s*left_val, s*right_val) - kernel(abs(left_val-right_val), k));
        } else if (node.type == NODETYPE_PRIMITIVE) {
            Primitive prim = prims.tab[node.idx_in_type];
            d = eval_prim(p, prim);
        }

        if (stack_idx >= EVAL_STACK_DEPTH) {
            return 1.0 / 0.0;
        }
        stack[stack_idx++] = d;
    }

    return stack[0];
}

// Pruned SDF: evaluates only the nodes still active in pruning cell cell_idx (256^3 grid).
float sdf_active(vec3 p, int cell_idx, BrickBakeParams params, out bool near_field) {
    IntArrayRef cells_num_active = IntArrayRef(params.cellNumActiveAddr);
    FloatArrayRef cell_errors = FloatArrayRef(params.cellErrorsAddr);

    int num_active = cells_num_active.tab[cell_idx];

    if (num_active == 0) {
        near_field = false;
        return cell_errors.tab[cell_idx];
    }

    IntArrayRef cells_offset = IntArrayRef(params.cellOffsetsAddr);
    ActiveNodesRef active_nodes = ActiveNodesRef(params.activeNodesAddr);
    NodesRef nodes = NodesRef(params.nodesAddr);
    BinaryOpsRef binary_ops = BinaryOpsRef(params.binaryOpsAddr);
    PrimitivesRef prims = PrimitivesRef(params.primsAddr);

    float stack[EVAL_STACK_DEPTH];
    int stack_idx = 0;

    int cell_offset = cells_offset.tab[cell_idx];

    for (int i = 0; i < num_active; i++) {
        ActiveNode active_node = active_nodes.tab[cell_offset + i];
        int node_idx = ActiveNode_index(active_node);

        Node node = nodes.tab[node_idx];
        float d;
        if (node.type == NODETYPE_BINARY) {
            float left_val = stack[stack_idx-2];
            float right_val = stack[stack_idx-1];
            stack_idx -= 2;
            BinaryOp op = binary_ops.tab[node.idx_in_type];
            float k = BinaryOp_blend_factor(op);
            float s = BinaryOp_sign(op);
            d = s*(min(s*left_val, s*right_val) - kernel(abs(left_val-right_val), k));
        } else if (node.type == NODETYPE_PRIMITIVE) {
            Primitive prim = prims.tab[node.idx_in_type];
            d = eval_prim(p, prim);
        }

        d *= ActiveNode_sign(active_node) ? 1 : -1;
        if (stack_idx >= EVAL_STACK_DEPTH) {
            return 1.0 / 0.0;
        }
        stack[stack_idx++] = d;
    }

    near_field = true;

    return stack[0];
}

#endif // PRUNING_EVAL_GLSL
