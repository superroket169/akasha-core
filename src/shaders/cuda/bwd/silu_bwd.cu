extern "C" __global__ void silu_bwd_kernel(const float* x, const float* dY, float* dX, unsigned int n) {
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        float grad_silu = sig + val * sig * (1.0f - sig);
        dX[idx] = dY[idx] * grad_silu;
    }
}
