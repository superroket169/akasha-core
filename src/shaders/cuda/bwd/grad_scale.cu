extern "C" __global__ void grad_scale_kernel(
    float* grad, const float* scale, const unsigned int* meta
) {
    unsigned int len = meta[0];
    // 2D grid, linearized
    unsigned int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < len) { grad[idx] = grad[idx] * scale[0]; }
}
