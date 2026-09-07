extern "C" __global__ void cache_write_kernel(
    const float* src, float* dst, const unsigned int* meta
) {
    unsigned int row_count = meta[0];
    unsigned int width = meta[1];
    unsigned int dst_row_offset = meta[2];

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= row_count || col >= width) return;
    
    unsigned int src_idx = row * width + col;
    unsigned int dst_idx = (dst_row_offset + row) * width + col;
    dst[dst_idx] = src[src_idx];
}
