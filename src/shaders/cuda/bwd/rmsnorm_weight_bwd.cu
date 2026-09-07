extern "C" __global__ void rmsnorm_weight_bwd_kernel(
    const float* dY, const float* X, const float* rsqrt_cache, float* dWeight,
    const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int size = meta[1];
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= size) return;

    float acc = 0.0f;
    for (unsigned int row = 0; row < seq_len; row++) {
        unsigned int offset = row * size;
        float norm_x = X[offset + i] * rsqrt_cache[row];
        acc += dY[offset + i] * norm_x;
    }
    dWeight[i] = dWeight[i] + acc;
}
