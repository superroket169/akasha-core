extern "C" __global__ void embedding_bwd_kernel(
    const unsigned int* tokens, const float* grad_output, float* grad_table,
    const unsigned int* meta
) {
    unsigned int vocab_size = meta[0];
    unsigned int embed_dim = meta[1];
    unsigned int seq_len = meta[2];
    
    unsigned int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = blockIdx.y;
    
    if (token_idx >= seq_len || dim_idx >= embed_dim) return;

    unsigned int token_id = tokens[token_idx];
    if (token_id >= vocab_size) return;

    unsigned int target_idx = token_id * embed_dim + dim_idx;
    float grad_val = grad_output[token_idx * embed_dim + dim_idx];
    atomicAdd(&grad_table[target_idx], grad_val);
}
