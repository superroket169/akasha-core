extern "C" __global__ void grad_sumsq_kernel(
    const float* grad, float* partials, const unsigned int* meta
) {
    unsigned int len = meta[0];
    unsigned int out_offset = meta[1];

    __shared__ float partial[256];
    unsigned int tid = threadIdx.x;
    unsigned int stride = gridDim.x * 256u;

    float acc = 0.0f;
    for (unsigned int i = blockIdx.x * 256u + tid; i < len; i += stride) {
        float v = grad[i];
        acc += v * v;
    }
    partial[tid] = acc;
    __syncthreads();

    for (unsigned int s = 128u; s > 0u; s >>= 1) {
        if (tid < s) { partial[tid] += partial[tid + s]; }
        __syncthreads();
    }

    if (tid == 0u) { partials[out_offset + blockIdx.x] = partial[0]; }
}
