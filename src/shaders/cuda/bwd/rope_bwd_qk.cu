extern "C" __global__ void rope_bwd_qk_kernel(
    float* d_q, float* d_k, const unsigned int* meta
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

    float dq0 = d_q[offset];
    float dq1 = d_q[offset + 1u];
    d_q[offset]      = dq0 * v_cos + dq1 * v_sin;
    d_q[offset + 1u] = -dq0 * v_sin + dq1 * v_cos;

    float dk0 = d_k[offset];
    float dk1 = d_k[offset + 1u];
    d_k[offset]      = dk0 * v_cos + dk1 * v_sin;
    d_k[offset + 1u] = -dk0 * v_sin + dk1 * v_cos;
}
