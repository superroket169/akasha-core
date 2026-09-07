extern "C" __global__ void rmsnorm_bwd_kernel(
    const float* dY, const float* X, const float* Weight,
    float* dX, float* rsqrt_cache,
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
        local_ss += X[offset + i] * X[offset + i];
    }
    partial[tid] = local_ss;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float ss = partial[0];
    __syncthreads();

    float rsqrt_v = 1.0f / sqrtf((ss / (float)size) + eps);
    if (tid == 0u) rsqrt_cache[row] = rsqrt_v;

    float local_sum_grad = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        float norm_x = X[offset + i] * rsqrt_v;
        float dy_w = dY[offset + i] * Weight[i];
        local_sum_grad += dy_w * norm_x;
    }
    partial[tid] = local_sum_grad;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float sum_grad = partial[0];
    __syncthreads();

    for (unsigned int i = tid; i < size; i += 256u) {
        float norm_x = X[offset + i] * rsqrt_v;
        float dy_w = dY[offset + i] * Weight[i];
        dX[offset + i] = rsqrt_v * (dy_w - (norm_x * sum_grad / (float)size));
    }
}
