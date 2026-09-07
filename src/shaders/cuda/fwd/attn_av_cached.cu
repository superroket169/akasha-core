extern "C" __global__ void attn_av_cached_kernel(
    const float* scores, const float* v_cache, float* out, const unsigned int* meta
) {
    unsigned int attn_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];

    unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    unsigned int s_off = (d / head_dim) * attn_len;
    float sum = 0.0f;
    for (unsigned int j = 0; j < attn_len; j++) {
        sum += scores[s_off + j] * v_cache[j * dim + d];
    }
    out[d] = sum;
}
