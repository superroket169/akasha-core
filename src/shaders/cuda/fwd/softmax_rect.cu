extern "C" __global__ void softmax_rect_kernel(float* x, const unsigned int* meta) {
    unsigned int num_rows = meta[0];
    unsigned int width = meta[1];
    float scale = __uint_as_float(meta[2]);
    
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows) return;
    unsigned int offset = row * width;

    float max_val = -1000000.0f;
    for (unsigned int i = 0; i < width; i++) {
        float val = x[offset + i] * scale;
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < width; i++) {
        float e = expf(x[offset + i] * scale - max_val);
        x[offset + i] = e;
        sum_exp += e;
    }

    for (unsigned int i = 0; i < width; i++) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}
