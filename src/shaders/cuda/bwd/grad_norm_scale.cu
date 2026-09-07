extern "C" __global__ void grad_norm_scale_kernel(
    const float* partials, float* scale, const unsigned int* meta
) {
    unsigned int num_partials = meta[0];
    float max_norm = __uint_as_float(meta[1]);

    __shared__ float partial[256];
    unsigned int tid = threadIdx.x;

    float acc = 0.0f;
    for (unsigned int i = tid; i < num_partials; i += 256u) {
        acc += partials[i];
    }
    partial[tid] = acc;
    __syncthreads();

    for (unsigned int s = 128u; s > 0u; s >>= 1) {
        if (tid < s) { partial[tid] += partial[tid + s]; }
        __syncthreads();
    }

    if (tid == 0u) {
        float norm = sqrtf(partial[0]);
        scale[0] = (norm > max_norm) ? max_norm / (norm + 1e-6f) : 1.0f;
    }
}
