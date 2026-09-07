extern "C" __global__ void rope_qk_kernel(
    float* q, float* k, const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];
    unsigned int row_offset = meta[3];
    
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    // grid.z spans the heads
    unsigned int h = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int row = row_offset + token_idx;
    unsigned int offset = row * dim + h * head_dim + dim_idx;

    float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
    float v_angle = (float)token_idx * freq;
    float v_cos = cosf(v_angle);
    float v_sin = sinf(v_angle);

    float q0 = q[offset];
    float q1 = q[offset + 1u];
    q[offset]      = q0 * v_cos - q1 * v_sin;
    q[offset + 1u] = q0 * v_sin + q1 * v_cos;

    float k0 = k[offset];
    float k1 = k[offset + 1u];
    k[offset]      = k0 * v_cos - k1 * v_sin;
    k[offset + 1u] = k0 * v_sin + k1 * v_cos;
}
