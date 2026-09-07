extern "C" __global__ void flash_attention_bwd_dkdv_kernel(
    const float* q, const float* k, const float* v, const float* o, const float* d_o,
    const float* l_cache, float* d_k, float* d_v,
    const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];

    float scale = __uint_as_float(meta[3]);
    
    unsigned int row_offset = meta[4];
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int num_heads = dim / head_dim;
    if (col >= seq_len || head >= num_heads) return;

    unsigned int head_off = head * head_dim;
    unsigned int kv_off = (row_offset + col) * dim + head_off;

    float dk_acc[128];
    float dv_acc[128];
    for (unsigned int d = 0; d < head_dim; d++) { dk_acc[d] = 0.0f; dv_acc[d] = 0.0f; }

    for (unsigned int i = col; i < seq_len; i++) {
        unsigned int qo_off = (row_offset + i) * dim + head_off;
        float l_i = l_cache[i * num_heads + head];

        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q[qo_off + d] * k[kv_off + d];
        }
        score *= scale;
        float p = expf(score - l_i);

        float d_i = 0.0f;
        float dp = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            d_i += d_o[qo_off + d] * o[qo_off + d];
            dp += d_o[qo_off + d] * v[kv_off + d];
        }
        float d_s = p * (dp - d_i);

        for (unsigned int d = 0; d < head_dim; d++) {
            dv_acc[d] += p * d_o[qo_off + d];
            dk_acc[d] += d_s * q[qo_off + d];
        }
    }

    for (unsigned int d = 0; d < head_dim; d++) {
        d_k[kv_off + d] = dk_acc[d] * scale;
        d_v[kv_off + d] = dv_acc[d];
    }
}
