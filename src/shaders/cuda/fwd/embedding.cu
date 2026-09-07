extern "C" __global__ void embedding_kernel(
    const unsigned int* tokens, const float* weight, float* output,
    const unsigned int* meta
) {
    unsigned int vocab_size = meta[0];
    unsigned int embed_dim = meta[1];
    unsigned int seq_len = meta[2];
    
    unsigned int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = blockIdx.y;
    
    if (token_idx >= seq_len || dim_idx >= embed_dim) return;

    unsigned int token_id = tokens[token_idx];
    unsigned int out_idx = token_idx * embed_dim + dim_idx;
    if (token_id < vocab_size) {
        output[out_idx] = weight[token_id * embed_dim + dim_idx];
    } else {

        // Output contract: every element gets written, even for an invalid
        // token id - otherwise pool garbage leaks into the hidden state.
        
        output[out_idx] = 0.0f;
    }
}
