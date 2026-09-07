extern "C" __global__ void silu_out_kernel(const float* x, float* y, unsigned int n) {
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}
