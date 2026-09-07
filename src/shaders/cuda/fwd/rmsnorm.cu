extern "C" __global__ void rmsnorm_kernel(
    const float* x, const float* weight, float* output,
    const unsigned int* meta
) {
    unsigned int seq_len = meta[0];
    unsigned int size = meta[1];
    float eps = __uint_as_float(meta[2]);
    
    __shared__ float partial[256];
    unsigned int row = blockIdx.x;
    
    if (row >= seq_len) return;
    unsigned int offset = row * size;
    unsigned int tid = threadIdx.x;

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        float val = x[offset + i];
        local_ss += val * val;
    }
    partial[tid] = local_ss;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    float rsqrt_v = 1.0f / sqrtf((partial[0] / (float)size) + eps);

    for (unsigned int i = tid; i < size; i += 256u) {
        output[offset + i] = x[offset + i] * rsqrt_v * weight[i];
    }
}
