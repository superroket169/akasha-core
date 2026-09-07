extern "C" __global__ void rope_offset_kernel(
    float* vec, const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];
    unsigned int pos_offset = meta[3];
    
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    // grid.z spans the heads
    unsigned int h = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int abs_pos = token_idx + pos_offset;
    unsigned int offset = token_idx * dim + h * head_dim + dim_idx;

    float x0 = vec[offset];
    float x1 = vec[offset + 1u];

    float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
    float v_angle = (float)abs_pos * freq;
    float v_cos = cosf(v_angle);
    float v_sin = sinf(v_angle);

    vec[offset]      = x0 * v_cos - x1 * v_sin;
    vec[offset + 1u] = x0 * v_sin + x1 * v_cos;
}
