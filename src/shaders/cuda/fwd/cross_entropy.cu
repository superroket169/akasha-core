// Workgroup-per-row, in place: logits become softmax probs; see the wgsl twin.
extern "C" __global__ void cross_entropy_kernel(
    float* logits, const unsigned int* targets, float* losses,
    const unsigned int* meta
) {
    unsigned int vocab_size = meta[0];
    unsigned int num_rows = meta[1];

    __shared__ float partial[256];
    __shared__ float row_max;
    __shared__ float row_sum;

    unsigned int row = blockIdx.x;
    if (row >= num_rows) return;
    unsigned int offset = row * vocab_size;
    unsigned int tid = threadIdx.x;

    float local_max = -3.4028235e38f;
    for (unsigned int i = tid; i < vocab_size; i += 256u) {
        local_max = fmaxf(local_max, logits[offset + i]);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    if (tid == 0u) row_max = partial[0];
    __syncthreads();
    float max_val = row_max;

    float local_sum = 0.0f;
    for (unsigned int i = tid; i < vocab_size; i += 256u) {
        local_sum += expf(logits[offset + i] - max_val);
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) row_sum = partial[0];
    __syncthreads();
    float sum_exp = row_sum;

    if (tid == 0u) {
        losses[row] = -(logits[offset + targets[row]] - max_val - logf(sum_exp));
    }
    __syncthreads();

    for (unsigned int i = tid; i < vocab_size; i += 256u) {
        logits[offset + i] = expf(logits[offset + i] - max_val) / sum_exp;
    }
}
