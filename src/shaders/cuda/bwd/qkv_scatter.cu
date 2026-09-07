extern "C" __global__ void qkv_scatter_kernel(
    const float* q, const float* k, const float* v, float* dst,
    const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int full_dim = meta[1];
    unsigned int head_dim = meta[2];
    unsigned int head_offset = meta[3];
    
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= seq_len || col >= head_dim) return;
    
    (void)head_offset;

    unsigned int width = head_dim;
    unsigned int src_idx = row * width + col;
    unsigned int dst_row = row * full_dim;

    dst[dst_row + col] = q[src_idx];
    dst[dst_row + width + col] = k[src_idx];
    dst[dst_row + 2u * width + col] = v[src_idx];
}
