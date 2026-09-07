extern "C" __global__ void rope_bwd_kernel(float* d_vec, const unsigned int* meta) {
    unsigned int seq_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];
    
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    // grid.z spans the heads
    unsigned int h = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
    float dx0 = d_vec[offset];
    float dx1 = d_vec[offset + 1u];

    float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
    float v_angle = (float)token_idx * freq;
    float v_cos = cosf(v_angle);
    float v_sin = sinf(v_angle);

    d_vec[offset]      = dx0 * v_cos + dx1 * v_sin;
    d_vec[offset + 1u] = -dx0 * v_sin + dx1 * v_cos;
}
