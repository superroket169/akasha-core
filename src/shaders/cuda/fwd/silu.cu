extern "C" __global__ void silu_kernel(float* x, unsigned int n) {
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        x[idx] = val / (1.0f + expf(-val));
    }
}
