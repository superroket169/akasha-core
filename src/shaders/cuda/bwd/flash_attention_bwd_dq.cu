extern "C" __global__ void flash_attention_bwd_dq_kernel(
    const float* q, const float* k, const float* v, const float* o, const float* d_o,
    const float* l_cache, float* d_q,
    const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int dim = meta[1];
    unsigned int head_dim = meta[2];

    float scale = __uint_as_float(meta[3]);
    
    unsigned int row_offset = meta[4];
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int num_heads = dim / head_dim;
    
    if (row >= seq_len || head >= num_heads) return;

    unsigned int head_off = head * head_dim;
    unsigned int q_off = (row_offset + row) * dim + head_off;
    unsigned int o_off = (row_offset + row) * dim + head_off;
    float l_i = l_cache[row * num_heads + head];

    float d_i = 0.0f;
    for (unsigned int d = 0; d < head_dim; d++) {
        d_i += d_o[o_off + d] * o[o_off + d];
    }

    float dq_acc[128];
    for (unsigned int d = 0; d < head_dim; d++) dq_acc[d] = 0.0f;

    for (unsigned int j = 0; j <= row; j++) {
        unsigned int kv_off = (row_offset + j) * dim + head_off;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q[q_off + d] * k[kv_off + d];
        }
        score *= scale;
        float p = expf(score - l_i);

        float dp = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dp += d_o[o_off + d] * v[kv_off + d];
        }
        float d_s = p * (dp - d_i);

        for (unsigned int d = 0; d < head_dim; d++) {
            dq_acc[d] += d_s * k[kv_off + d];
        }
    }

    unsigned int dq_off = (row_offset + row) * dim + head_off;
    for (unsigned int d = 0; d < head_dim; d++) {
        d_q[dq_off + d] = dq_acc[d] * scale;
    }
}
