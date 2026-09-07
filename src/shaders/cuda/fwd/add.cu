extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, unsigned int n) {
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
