extern "C" __global__ void qkv_split_kernel(
    const float* src, float* q, float* k, float* v,
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
    unsigned int src_row = row * full_dim;
    unsigned int dst_idx = row * width + col;

    q[dst_idx] = src[src_row + col];
    k[dst_idx] = src[src_row + width + col];
    v[dst_idx] = src[src_row + 2u * width + col];
}
