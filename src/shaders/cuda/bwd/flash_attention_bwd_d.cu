extern "C" __global__ void flash_attention_bwd_d_kernel(
    const float* d_o, const float* o, float* d_sum,
    const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];

    unsigned int row_offset = meta[4];
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int num_heads = dim / head_dim;
    if (row >= seq_len || head >= num_heads) return;

    unsigned int off = (row_offset + row) * dim + head * head_dim;

    float d_i = 0.0f;
    for (unsigned int d = 0; d < head_dim; d++) {
        d_i += d_o[off + d] * o[off + d];
    }

    d_sum[row * num_heads + head] = d_i;
}
