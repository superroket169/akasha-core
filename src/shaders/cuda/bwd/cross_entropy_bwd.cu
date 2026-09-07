// One thread per element, in place: probs become grad_logits.
extern "C" __global__ void cross_entropy_bwd_kernel(
    float* probs, const unsigned int* targets, const float* d_losses,
    const unsigned int* meta
) {
    unsigned int vocab_size = meta[0];
    unsigned int num_rows = meta[1];

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y;
    if (col >= vocab_size || row >= num_rows) return;

    unsigned int idx = row * vocab_size + col;
    float g = probs[idx];
    if (col == targets[row]) g -= 1.0f;
    probs[idx] = g * d_losses[row];
}
