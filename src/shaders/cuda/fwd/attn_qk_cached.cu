extern "C" __global__ void attn_qk_cached_kernel(
    const float* q, const float* k_cache, float* scores, const unsigned int* meta
) {
    unsigned int attn_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];

    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int h = blockIdx.y;
    if (j >= attn_len) return;

    unsigned int q_off = h * head_dim;
    unsigned int k_off = j * dim + q_off;
    float sum = 0.0f;
    for (unsigned int c = 0; c < head_dim; c++) {
        sum += q[q_off + c] * k_cache[k_off + c];
    }
    scores[h * attn_len + j] = sum;
}
