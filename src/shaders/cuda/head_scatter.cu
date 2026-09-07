extern "C" __global__ void head_scatter_kernel(
    const float* src, float* dst, const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int full_dim = meta[1];
    unsigned int head_dim = meta[2];
    unsigned int head_offset = meta[3];
    
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= seq_len || col >= head_dim) return;
    
    unsigned int src_idx = row * head_dim + col;
    unsigned int dst_idx = row * full_dim + head_offset + col;
    dst[dst_idx] = src[src_idx];
}
