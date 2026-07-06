#![allow(dead_code)]

use wilupgu::Shader;
use wilupgu::backends::cuda::{CudaBackend, CudaBinding};

fn shader_key(shader: &'static Shader) -> usize {
    shader as *const Shader as usize
}

pub(crate) const EMBEDDING: &str = r#"
extern "C" __global__ void embedding_kernel(
    const unsigned int* tokens, const float* weight, float* output,
    unsigned int vocab_size, unsigned int embed_dim, unsigned int seq_len
) {
    unsigned int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = blockIdx.y;
    if (token_idx >= seq_len || dim_idx >= embed_dim) return;

    unsigned int token_id = tokens[token_idx];
    if (token_id < vocab_size) {
        unsigned int weight_idx = token_id * embed_dim + dim_idx;
        unsigned int out_idx = token_idx * embed_dim + dim_idx;
        output[out_idx] = weight[weight_idx];
    }
}
"#;

pub(crate) const EMBEDDING_BWD: &str = r#"
extern "C" __global__ void embedding_bwd_kernel(
    const unsigned int* tokens, const float* grad_output, float* grad_table,
    unsigned int vocab_size, unsigned int embed_dim, unsigned int seq_len
) {
    unsigned int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = blockIdx.y;
    if (token_idx >= seq_len || dim_idx >= embed_dim) return;

    unsigned int token_id = tokens[token_idx];
    if (token_id >= vocab_size) return;

    unsigned int target_idx = token_id * embed_dim + dim_idx;
    float grad_val = grad_output[token_idx * embed_dim + dim_idx];
    atomicAdd(&grad_table[target_idx], grad_val);
}
"#;

pub(crate) const SILU: &str = r#"
extern "C" __global__ void silu_kernel(float* x, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        x[idx] = val / (1.0f + expf(-val));
    }
}
"#;

pub(crate) const SILU_BWD: &str = r#"
extern "C" __global__ void silu_bwd_kernel(const float* x, const float* dY, float* dX, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        float grad_silu = sig + val * sig * (1.0f - sig);
        dX[idx] = dY[idx] * grad_silu;
    }
}
"#;

pub(crate) const ROPE: &str = r#"
extern "C" __global__ void rope_kernel(float* vec, unsigned int seq_len, unsigned int dim, unsigned int head_dim) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
        float x0 = vec[offset];
        float x1 = vec[offset + 1u];

        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)token_idx * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        vec[offset]      = x0 * v_cos - x1 * v_sin;
        vec[offset + 1u] = x0 * v_sin + x1 * v_cos;
    }
}
"#;

pub(crate) const ROPE_BWD: &str = r#"
extern "C" __global__ void rope_bwd_kernel(float* d_vec, unsigned int seq_len, unsigned int dim, unsigned int head_dim) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
        float dx0 = d_vec[offset];
        float dx1 = d_vec[offset + 1u];

        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)token_idx * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        d_vec[offset]      = dx0 * v_cos + dx1 * v_sin;
        d_vec[offset + 1u] = -dx0 * v_sin + dx1 * v_cos;
    }
}
"#;

pub(crate) const ROPE_QK: &str = r#"
extern "C" __global__ void rope_qk_kernel(
    float* q, float* k, unsigned int seq_len, unsigned int dim, unsigned int head_dim,
    unsigned int row_offset
) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    unsigned int row = row_offset + token_idx;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = row * dim + h * head_dim + dim_idx;
        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)token_idx * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        float q0 = q[offset];
        float q1 = q[offset + 1u];
        q[offset]      = q0 * v_cos - q1 * v_sin;
        q[offset + 1u] = q0 * v_sin + q1 * v_cos;

        float k0 = k[offset];
        float k1 = k[offset + 1u];
        k[offset]      = k0 * v_cos - k1 * v_sin;
        k[offset + 1u] = k0 * v_sin + k1 * v_cos;
    }
}
"#;

pub(crate) const ROPE_BWD_QK: &str = r#"
extern "C" __global__ void rope_bwd_qk_kernel(
    float* d_q, float* d_k, unsigned int seq_len, unsigned int dim, unsigned int head_dim,
    unsigned int row_offset
) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    unsigned int row = row_offset + token_idx;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = row * dim + h * head_dim + dim_idx;
        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)token_idx * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        float dq0 = d_q[offset];
        float dq1 = d_q[offset + 1u];
        d_q[offset]      = dq0 * v_cos + dq1 * v_sin;
        d_q[offset + 1u] = -dq0 * v_sin + dq1 * v_cos;

        float dk0 = d_k[offset];
        float dk1 = d_k[offset + 1u];
        d_k[offset]      = dk0 * v_cos + dk1 * v_sin;
        d_k[offset + 1u] = -dk0 * v_sin + dk1 * v_cos;
    }
}
"#;

pub(crate) const ROPE_OFFSET: &str = r#"
extern "C" __global__ void rope_offset_kernel(
    float* vec, unsigned int seq_len, unsigned int dim, unsigned int head_dim, unsigned int pos_offset
) {
    unsigned int dim_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    unsigned int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (token_idx >= seq_len || dim_idx >= head_dim) return;

    unsigned int num_heads = dim / head_dim;
    unsigned int abs_pos = token_idx + pos_offset;
    for (unsigned int h = 0; h < num_heads; h++) {
        unsigned int offset = token_idx * dim + h * head_dim + dim_idx;
        float x0 = vec[offset];
        float x1 = vec[offset + 1u];

        float freq = 1.0f / powf(10000.0f, (float)dim_idx / (float)head_dim);
        float v_angle = (float)abs_pos * freq;
        float v_cos = cosf(v_angle);
        float v_sin = sinf(v_angle);

        vec[offset]      = x0 * v_cos - x1 * v_sin;
        vec[offset + 1u] = x0 * v_sin + x1 * v_cos;
    }
}
"#;

pub(crate) const HEAD_GATHER: &str = r#"
extern "C" __global__ void head_gather_kernel(
    const float* src, float* dst,
    unsigned int seq_len, unsigned int full_dim, unsigned int head_dim, unsigned int head_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= head_dim) return;
    unsigned int src_idx = row * full_dim + head_offset + col;
    unsigned int dst_idx = row * head_dim + col;
    dst[dst_idx] = src[src_idx];
}
"#;

pub(crate) const HEAD_SCATTER: &str = r#"
extern "C" __global__ void head_scatter_kernel(
    const float* src, float* dst,
    unsigned int seq_len, unsigned int full_dim, unsigned int head_dim, unsigned int head_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= head_dim) return;
    unsigned int src_idx = row * head_dim + col;
    unsigned int dst_idx = row * full_dim + head_offset + col;
    dst[dst_idx] = src[src_idx];
}
"#;

pub(crate) const QKV_SPLIT: &str = r#"
extern "C" __global__ void qkv_split_kernel(
    const float* src, float* q, float* k, float* v,
    unsigned int seq_len, unsigned int full_dim, unsigned int head_dim, unsigned int head_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= head_dim) return;
    (void)head_offset;

    unsigned int width = head_dim;
    unsigned int src_row = row * full_dim;
    unsigned int dst_idx = row * width + col;

    q[dst_idx] = src[src_row + col];
    k[dst_idx] = src[src_row + width + col];
    v[dst_idx] = src[src_row + 2u * width + col];
}
"#;

pub(crate) const QKV_SCATTER: &str = r#"
extern "C" __global__ void qkv_scatter_kernel(
    const float* q, const float* k, const float* v, float* dst,
    unsigned int seq_len, unsigned int full_dim, unsigned int head_dim, unsigned int head_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq_len || col >= head_dim) return;
    (void)head_offset;

    unsigned int width = head_dim;
    unsigned int src_idx = row * width + col;
    unsigned int dst_row = row * full_dim;

    dst[dst_row + col] = q[src_idx];
    dst[dst_row + width + col] = k[src_idx];
    dst[dst_row + 2u * width + col] = v[src_idx];
}
"#;

pub(crate) const SOFTMAX: &str = r#"
extern "C" __global__ void softmax_kernel(float* x, unsigned int seq_len) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * seq_len;

    float max_val = -1000000.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float val = x[offset + i];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float e = expf(x[offset + i] - max_val);
        x[offset + i] = e;
        sum_exp += e;
    }

    for (unsigned int i = 0; i < seq_len; i++) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}
"#;

pub(crate) const SOFTMAX_BWD: &str = r#"
extern "C" __global__ void softmax_bwd_kernel(
    const float* Y, const float* dY, float* dX, unsigned int seq_len, float scale
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * seq_len;

    float sum_ydy = 0.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        sum_ydy += Y[offset + i] * dY[offset + i];
    }

    for (unsigned int i = 0; i < seq_len; i++) {
        dX[offset + i] = Y[offset + i] * (dY[offset + i] - sum_ydy) * scale;
    }
}
"#;

pub(crate) const SOFTMAX_RECT: &str = r#"
extern "C" __global__ void softmax_rect_kernel(float* x, unsigned int num_rows, unsigned int width, float scale) {
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
"#;

pub(crate) const CAUSAL_SOFTMAX: &str = r#"
extern "C" __global__ void causal_softmax_kernel(float* x, unsigned int seq_len, float scale) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * seq_len;

    float max_val = -1000000.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float val = (i > row) ? -1000000000.0f : x[offset + i] * scale;
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < seq_len; i++) {
        float val = (i > row) ? -1000000000.0f : x[offset + i] * scale;
        float e = expf(val - max_val);
        x[offset + i] = e;
        sum_exp += e;
    }

    for (unsigned int i = 0; i < seq_len; i++) {
        x[offset + i] = x[offset + i] / sum_exp;
    }
}
"#;

pub(crate) const RMSNORM: &str = r#"
extern "C" __global__ void rmsnorm_kernel(
    const float* x, const float* weight, float* output,
    unsigned int seq_len, unsigned int size, float eps
) {
    __shared__ float partial[256];
    unsigned int row = blockIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * size;
    unsigned int tid = threadIdx.x;

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        float val = x[offset + i];
        local_ss += val * val;
    }
    partial[tid] = local_ss;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    float rsqrt_v = 1.0f / sqrtf((partial[0] / (float)size) + eps);

    for (unsigned int i = tid; i < size; i += 256u) {
        output[offset + i] = x[offset + i] * rsqrt_v * weight[i];
    }
}
"#;

pub(crate) const RMSNORM_BWD: &str = r#"
extern "C" __global__ void rmsnorm_bwd_kernel(
    const float* dY, const float* X, const float* Weight,
    float* dX, float* rsqrt_cache,
    unsigned int seq_len, unsigned int size, float eps
) {
    __shared__ float partial[256];
    unsigned int row = blockIdx.x;
    if (row >= seq_len) return;
    unsigned int offset = row * size;
    unsigned int tid = threadIdx.x;

    float local_ss = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        local_ss += X[offset + i] * X[offset + i];
    }
    partial[tid] = local_ss;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float ss = partial[0];
    __syncthreads();

    float rsqrt_v = 1.0f / sqrtf((ss / (float)size) + eps);
    if (tid == 0u) rsqrt_cache[row] = rsqrt_v;

    float local_sum_grad = 0.0f;
    for (unsigned int i = tid; i < size; i += 256u) {
        float norm_x = X[offset + i] * rsqrt_v;
        float dy_w = dY[offset + i] * Weight[i];
        local_sum_grad += dy_w * norm_x;
    }
    partial[tid] = local_sum_grad;
    __syncthreads();
    for (unsigned int stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    float sum_grad = partial[0];
    __syncthreads();

    for (unsigned int i = tid; i < size; i += 256u) {
        float norm_x = X[offset + i] * rsqrt_v;
        float dy_w = dY[offset + i] * Weight[i];
        dX[offset + i] = rsqrt_v * (dy_w - (norm_x * sum_grad / (float)size));
    }
}
"#;

pub(crate) const RMSNORM_WEIGHT_BWD: &str = r#"
extern "C" __global__ void rmsnorm_weight_bwd_kernel(
    const float* dY, const float* X, const float* rsqrt_cache, float* dWeight,
    unsigned int seq_len, unsigned int size
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float acc = 0.0f;
    for (unsigned int row = 0; row < seq_len; row++) {
        unsigned int offset = row * size;
        float norm_x = X[offset + i] * rsqrt_cache[row];
        acc += dY[offset + i] * norm_x;
    }
    dWeight[i] = dWeight[i] + acc;
}
"#;

pub(crate) const CROSS_ENTROPY: &str = r#"
extern "C" __global__ void cross_entropy_kernel(
    const float* logits, const unsigned int* targets, float* probs, float* losses,
    unsigned int vocab_size, unsigned int num_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    unsigned int offset = row * vocab_size;
    unsigned int target_id = targets[row];

    float max_val = -3.4028235e38f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        max_val = fmaxf(max_val, logits[offset + i]);
    }

    float sum_exp = 0.0f;
    for (unsigned int i = 0; i < vocab_size; i++) {
        float e = expf(logits[offset + i] - max_val);
        probs[offset + i] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    for (unsigned int i = 0; i < vocab_size; i++) {
        probs[offset + i] = probs[offset + i] * inv_sum;
    }

    float log_sum_exp = logf(sum_exp);
    losses[row] = -(logits[offset + target_id] - max_val - log_sum_exp);
}
"#;

pub(crate) const CROSS_ENTROPY_BWD: &str = r#"
extern "C" __global__ void cross_entropy_bwd_kernel(
    const float* probs, const unsigned int* targets, const float* d_losses, float* d_logits,
    unsigned int vocab_size, unsigned int num_rows
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    unsigned int offset = row * vocab_size;
    unsigned int target_id = targets[row];
    float grad_scale = d_losses[row];

    for (unsigned int i = 0; i < vocab_size; i++) {
        float indicator = (i == target_id) ? 1.0f : 0.0f;
        d_logits[offset + i] = (probs[offset + i] - indicator) * grad_scale;
    }
}
"#;

pub(crate) const CACHE_WRITE: &str = r#"
extern "C" __global__ void cache_write_kernel(
    const float* src, float* dst,
    unsigned int row_count, unsigned int width, unsigned int dst_row_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= row_count || col >= width) return;
    unsigned int src_idx = row * width + col;
    unsigned int dst_idx = (dst_row_offset + row) * width + col;
    dst[dst_idx] = src[src_idx];
}
"#;

pub(crate) const FLASH_ATTENTION: &str = r#"
extern "C" __global__ void flash_attention_kernel(
    const float* q, const float* k, const float* v, float* out, float* l_cache,
    unsigned int seq_len, unsigned int dim, unsigned int head_dim, float scale,
    unsigned int row_offset
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int num_heads = dim / head_dim;
    if (row >= seq_len || head >= num_heads) return;

    unsigned int head_off = head * head_dim;
    unsigned int q_off = (row_offset + row) * dim + head_off;

    float acc[128];
    for (unsigned int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    float row_max = -1000000000.0f;
    float row_sum = 0.0f;

    for (unsigned int j = 0; j <= row; j++) {
        unsigned int kv_off = (row_offset + j) * dim + head_off;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q[q_off + d] * k[kv_off + d];
        }
        score *= scale;

        float new_max = fmaxf(row_max, score);
        float correction = expf(row_max - new_max);
        float p = expf(score - new_max);

        row_sum = row_sum * correction + p;
        for (unsigned int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction + p * v[kv_off + d];
        }
        row_max = new_max;
    }

    unsigned int out_off = (row_offset + row) * dim + head_off;
    for (unsigned int d = 0; d < head_dim; d++) {
        out[out_off + d] = acc[d] / row_sum;
    }

    l_cache[row * num_heads + head] = row_max + logf(row_sum);
}
"#;

pub(crate) const FLASH_ATTENTION_BWD_DQ: &str = r#"
extern "C" __global__ void flash_attention_bwd_dq_kernel(
    const float* q, const float* k, const float* v, const float* o, const float* d_o,
    const float* l_cache, float* d_q,
    unsigned int seq_len, unsigned int dim, unsigned int head_dim, float scale,
    unsigned int row_offset
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int num_heads = dim / head_dim;
    if (row >= seq_len || head >= num_heads) return;

    unsigned int head_off = head * head_dim;
    unsigned int q_off = (row_offset + row) * dim + head_off;
    unsigned int o_off = (row_offset + row) * dim + head_off;
    float l_i = l_cache[row * num_heads + head];

    float d_i = 0.0f;
    for (unsigned int d = 0; d < head_dim; d++) {
        d_i += d_o[o_off + d] * o[o_off + d];
    }

    float dq_acc[128];
    for (unsigned int d = 0; d < head_dim; d++) dq_acc[d] = 0.0f;

    for (unsigned int j = 0; j <= row; j++) {
        unsigned int kv_off = (row_offset + j) * dim + head_off;
        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q[q_off + d] * k[kv_off + d];
        }
        score *= scale;
        float p = expf(score - l_i);

        float dp = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            dp += d_o[o_off + d] * v[kv_off + d];
        }
        float d_s = p * (dp - d_i);

        for (unsigned int d = 0; d < head_dim; d++) {
            dq_acc[d] += d_s * k[kv_off + d];
        }
    }

    unsigned int dq_off = (row_offset + row) * dim + head_off;
    for (unsigned int d = 0; d < head_dim; d++) {
        d_q[dq_off + d] = dq_acc[d] * scale;
    }
}
"#;

pub(crate) const FLASH_ATTENTION_BWD_DKDV: &str = r#"
extern "C" __global__ void flash_attention_bwd_dkdv_kernel(
    const float* q, const float* k, const float* v, const float* o, const float* d_o,
    const float* l_cache, float* d_k, float* d_v,
    unsigned int seq_len, unsigned int dim, unsigned int head_dim, float scale,
    unsigned int row_offset
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y;
    unsigned int num_heads = dim / head_dim;
    if (col >= seq_len || head >= num_heads) return;

    unsigned int head_off = head * head_dim;
    unsigned int kv_off = (row_offset + col) * dim + head_off;

    float dk_acc[128];
    float dv_acc[128];
    for (unsigned int d = 0; d < head_dim; d++) { dk_acc[d] = 0.0f; dv_acc[d] = 0.0f; }

    for (unsigned int i = col; i < seq_len; i++) {
        unsigned int qo_off = (row_offset + i) * dim + head_off;
        float l_i = l_cache[i * num_heads + head];

        float score = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += q[qo_off + d] * k[kv_off + d];
        }
        score *= scale;
        float p = expf(score - l_i);

        float d_i = 0.0f;
        float dp = 0.0f;
        for (unsigned int d = 0; d < head_dim; d++) {
            d_i += d_o[qo_off + d] * o[qo_off + d];
            dp += d_o[qo_off + d] * v[kv_off + d];
        }
        float d_s = p * (dp - d_i);

        for (unsigned int d = 0; d < head_dim; d++) {
            dv_acc[d] += p * d_o[qo_off + d];
            dk_acc[d] += d_s * q[qo_off + d];
        }
    }

    for (unsigned int d = 0; d < head_dim; d++) {
        d_k[kv_off + d] = dk_acc[d] * scale;
        d_v[kv_off + d] = dv_acc[d];
    }
}
"#;

// ==========================================================================
//                        Custom-shape dispatches
// ==========================================================================

pub(crate) fn embedding(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    wg: [u32; 3],
) {
    b.launch_embedding(bindings, shader_key(s), EMBEDDING, "embedding_kernel", wg)
}

pub(crate) fn embedding_bwd(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    wg: [u32; 3],
) {
    b.launch_embedding(
        bindings,
        shader_key(s),
        EMBEDDING_BWD,
        "embedding_bwd_kernel",
        wg,
    )
}

pub(crate) fn rope(s: &'static Shader, b: &CudaBackend, bindings: &[CudaBinding], _wg: [u32; 3]) {
    b.launch_rope(bindings, shader_key(s), ROPE, "rope_kernel")
}

pub(crate) fn rope_bwd(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rope(bindings, shader_key(s), ROPE_BWD, "rope_bwd_kernel")
}

pub(crate) fn rope_qk(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rope_qk(bindings, shader_key(s), ROPE_QK, "rope_qk_kernel")
}

pub(crate) fn rope_bwd_qk(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rope_qk(bindings, shader_key(s), ROPE_BWD_QK, "rope_bwd_qk_kernel")
}

pub(crate) fn rope_offset(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rope_offset(bindings, shader_key(s), ROPE_OFFSET, "rope_offset_kernel")
}

pub(crate) fn softmax(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_softmax(bindings, shader_key(s), SOFTMAX, "softmax_kernel")
}

pub(crate) fn softmax_bwd(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_softmax_bwd(bindings, shader_key(s), SOFTMAX_BWD, "softmax_bwd_kernel")
}

pub(crate) fn softmax_rect(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_softmax_rect(bindings, shader_key(s), SOFTMAX_RECT, "softmax_rect_kernel")
}

pub(crate) fn causal_softmax(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_causal_softmax(
        bindings,
        shader_key(s),
        CAUSAL_SOFTMAX,
        "causal_softmax_kernel",
    )
}

pub(crate) fn rmsnorm(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rmsnorm(bindings, shader_key(s), RMSNORM, "rmsnorm_kernel")
}

pub(crate) fn rmsnorm_bwd(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rmsnorm_bwd(bindings, shader_key(s), RMSNORM_BWD, "rmsnorm_bwd_kernel")
}

pub(crate) fn rmsnorm_weight_bwd(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_rmsnorm_weight_bwd(
        bindings,
        shader_key(s),
        RMSNORM_WEIGHT_BWD,
        "rmsnorm_weight_bwd_kernel",
    )
}

pub(crate) fn cross_entropy(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_cross_entropy(
        bindings,
        shader_key(s),
        CROSS_ENTROPY,
        "cross_entropy_kernel",
    )
}

pub(crate) fn cross_entropy_bwd(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_cross_entropy_bwd(
        bindings,
        shader_key(s),
        CROSS_ENTROPY_BWD,
        "cross_entropy_bwd_kernel",
    )
}

pub(crate) fn head_gather(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_head_move(bindings, shader_key(s), HEAD_GATHER, "head_gather_kernel")
}

pub(crate) fn head_scatter(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_head_move(bindings, shader_key(s), HEAD_SCATTER, "head_scatter_kernel")
}

pub(crate) fn cache_write(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_cache_write(bindings, shader_key(s), CACHE_WRITE, "cache_write_kernel")
}

pub(crate) fn qkv_split(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_qkv_split(bindings, shader_key(s), QKV_SPLIT, "qkv_split_kernel")
}

pub(crate) fn qkv_scatter(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_qkv_scatter(bindings, shader_key(s), QKV_SCATTER, "qkv_scatter_kernel")
}

pub(crate) fn flash_attention(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_flash_attention(
        bindings,
        shader_key(s),
        FLASH_ATTENTION,
        "flash_attention_kernel",
    )
}

pub(crate) fn flash_attention_bwd_dq(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_flash_attention_bwd_dq(
        bindings,
        shader_key(s),
        FLASH_ATTENTION_BWD_DQ,
        "flash_attention_bwd_dq_kernel",
    )
}

pub(crate) fn flash_attention_bwd_dkdv(
    s: &'static Shader,
    b: &CudaBackend,
    bindings: &[CudaBinding],
    _wg: [u32; 3],
) {
    b.launch_flash_attention_bwd_dkdv(
        bindings,
        shader_key(s),
        FLASH_ATTENTION_BWD_DKDV,
        "flash_attention_bwd_dkdv_kernel",
    )
}
