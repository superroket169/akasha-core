use wilupgu::CpuBinding;

fn find(bindings: &[CpuBinding], slot: u32) -> &CpuBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn read_f32(b: &CpuBinding) -> Vec<f32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, f32>(&g)
}

fn read_u32(b: &CpuBinding) -> Vec<u32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::pod_collect_to_vec::<u8, u32>(&g)
}

fn write_f32(b: &CpuBinding, data: &[f32]) {
    let mut g = b.buffer.lock().unwrap();
    g.copy_from_slice(bytemuck::cast_slice(data));
}

pub(crate) fn embedding(bindings: &[CpuBinding]) {
    let tokens = read_u32(find(bindings, 0));
    let weight = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (vocab_size, embed_dim, seq_len) = (meta[0], meta[1] as usize, meta[2] as usize);

    let mut out = vec![0.0f32; seq_len * embed_dim];
    for t in 0..seq_len {
        let token_id = tokens[t];
        if token_id < vocab_size {
            let w_off = token_id as usize * embed_dim;
            let o_off = t * embed_dim;
            out[o_off..o_off + embed_dim].copy_from_slice(&weight[w_off..w_off + embed_dim]);
        }
    }
    write_f32(find(bindings, 2), &out);
}

pub(crate) fn embedding_bwd(bindings: &[CpuBinding]) {
    let tokens = read_u32(find(bindings, 0));
    let grad_output = read_f32(find(bindings, 1));
    let mut grad_table = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 3));
    let (vocab_size, embed_dim, seq_len) = (meta[0], meta[1] as usize, meta[2] as usize);

    for t in 0..seq_len {
        let token_id = tokens[t];
        if token_id < vocab_size {
            let w_off = token_id as usize * embed_dim;
            let g_off = t * embed_dim;
            for d in 0..embed_dim {
                grad_table[w_off + d] += grad_output[g_off + d];
            }
        }
    }
    write_f32(find(bindings, 2), &grad_table);
}

pub(crate) fn silu(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
    write_f32(find(bindings, 0), &x);
}

pub(crate) fn silu_out(bindings: &[CpuBinding]) {
    let x = read_f32(find(bindings, 0));
    let y: Vec<f32> = x.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    write_f32(find(bindings, 1), &y);
}

pub(crate) fn add(bindings: &[CpuBinding]) {
    let a = read_f32(find(bindings, 0));
    let b = read_f32(find(bindings, 1));
    let out: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    write_f32(find(bindings, 2), &out);
}

pub(crate) fn rope(bindings: &[CpuBinding]) {
    let mut vec_ = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (seq_len, dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let offset = token_idx * dim + h * head_dim + dim_idx;
                let x0 = vec_[offset];
                let x1 = vec_[offset + 1];

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = token_idx as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                vec_[offset] = x0 * v_cos - x1 * v_sin;
                vec_[offset + 1] = x0 * v_sin + x1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &vec_);
}

pub(crate) fn rope_offset(bindings: &[CpuBinding]) {
    let mut vec_ = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (seq_len, dim, head_dim, pos_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let abs_pos = token_idx + pos_offset;
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let offset = token_idx * dim + h * head_dim + dim_idx;
                let x0 = vec_[offset];
                let x1 = vec_[offset + 1];

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = abs_pos as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                vec_[offset] = x0 * v_cos - x1 * v_sin;
                vec_[offset + 1] = x0 * v_sin + x1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &vec_);
}

pub(crate) fn attn_qk_cached(bindings: &[CpuBinding]) {
    let q = read_f32(find(bindings, 0));
    let k_cache = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (attn_len, dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let num_heads = dim / head_dim;

    let mut scores = read_f32(find(bindings, 2));
    for h in 0..num_heads {
        let q_off = h * head_dim;
        for j in 0..attn_len {
            let k_off = j * dim + q_off;
            scores[h * attn_len + j] = (0..head_dim)
                .map(|c| q[q_off + c] * k_cache[k_off + c])
                .sum();
        }
    }
    write_f32(find(bindings, 2), &scores);
}

pub(crate) fn attn_av_cached(bindings: &[CpuBinding]) {
    let scores = read_f32(find(bindings, 0));
    let v_cache = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (attn_len, dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    let mut out = read_f32(find(bindings, 2));
    for d in 0..dim {
        let s_off = (d / head_dim) * attn_len;
        out[d] = (0..attn_len)
            .map(|j| scores[s_off + j] * v_cache[j * dim + d])
            .sum();
    }
    write_f32(find(bindings, 2), &out);
}

pub(crate) fn softmax_rect(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (num_rows, width) = (meta[0] as usize, meta[1] as usize);
    let scale = f32::from_bits(meta[2]);

    for row in 0..num_rows {
        let off = row * width;
        let max_val = x[off..off + width]
            .iter()
            .map(|v| v * scale)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for i in 0..width {
            let e = (x[off + i] * scale - max_val).exp();
            x[off + i] = e;
            sum_exp += e;
        }
        for i in 0..width {
            x[off + i] /= sum_exp;
        }
    }
    write_f32(find(bindings, 0), &x);
}

pub(crate) fn rmsnorm(bindings: &[CpuBinding]) {
    let x = read_f32(find(bindings, 0));
    let weight = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (seq_len, size) = (meta[0] as usize, meta[1] as usize);
    let eps = f32::from_bits(meta[2]);

    let mut out = vec![0.0f32; seq_len * size];
    for row in 0..seq_len {
        let off = row * size;
        let ss: f32 = x[off..off + size].iter().map(|v| v * v).sum();
        let rsqrt = 1.0 / ((ss / size as f32) + eps).sqrt();
        for i in 0..size {
            out[off + i] = x[off + i] * rsqrt * weight[i];
        }
    }
    write_f32(find(bindings, 2), &out);
}

pub(crate) fn cache_write(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let mut dst = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (row_count, width, dst_row_offset) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    for row in 0..row_count {
        let src_off = row * width;
        let dst_off = (dst_row_offset + row) * width;
        dst[dst_off..dst_off + width].copy_from_slice(&src[src_off..src_off + width]);
    }
    write_f32(find(bindings, 1), &dst);
}

pub(crate) fn head_gather(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 2));
    let (seq_len, full_dim, head_dim, head_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );

    let mut dst = read_f32(find(bindings, 1));
    for row in 0..seq_len {
        let src_off = row * full_dim + head_offset;
        let dst_off = row * head_dim;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
    write_f32(find(bindings, 1), &dst);
}

pub(crate) fn head_scatter(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let mut dst = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (seq_len, full_dim, head_dim, head_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );

    for row in 0..seq_len {
        let src_off = row * head_dim;
        let dst_off = row * full_dim + head_offset;
        dst[dst_off..dst_off + head_dim].copy_from_slice(&src[src_off..src_off + head_dim]);
    }
    write_f32(find(bindings, 1), &dst);
}

// In place: binding 0 goes in as logits, comes out as softmax probs.
pub(crate) fn cross_entropy(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let targets = read_u32(find(bindings, 1));
    let meta = read_u32(find(bindings, 3));
    let (vocab_size, num_rows) = (meta[0] as usize, meta[1] as usize);

    let mut losses = vec![0.0f32; num_rows];
    for row in 0..num_rows {
        let off = row * vocab_size;
        let target_id = targets[row] as usize;

        let max_val = x[off..off + vocab_size]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        for i in 0..vocab_size {
            sum_exp += (x[off + i] - max_val).exp();
        }
        losses[row] = -(x[off + target_id] - max_val - sum_exp.ln());

        for i in 0..vocab_size {
            x[off + i] = (x[off + i] - max_val).exp() / sum_exp;
        }
    }

    write_f32(find(bindings, 0), &x);
    write_f32(find(bindings, 2), &losses);
}

// In place: binding 0 goes in as probs, comes out as grad_logits.
pub(crate) fn cross_entropy_bwd(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let targets = read_u32(find(bindings, 1));
    let d_losses = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 3));
    let (vocab_size, num_rows) = (meta[0] as usize, meta[1] as usize);

    for row in 0..num_rows {
        let off = row * vocab_size;
        let target_id = targets[row] as usize;
        for i in 0..vocab_size {
            let indicator = if i == target_id { 1.0 } else { 0.0 };
            x[off + i] = (x[off + i] - indicator) * d_losses[row];
        }
    }
    write_f32(find(bindings, 0), &x);
}

pub(crate) fn silu_bwd(bindings: &[CpuBinding]) {
    let x = read_f32(find(bindings, 0));
    let d_y = read_f32(find(bindings, 1));
    let d_x: Vec<f32> = x
        .iter()
        .zip(d_y.iter())
        .map(|(&v, &dy)| {
            let sig = 1.0 / (1.0 + (-v).exp());
            let grad_silu = sig + v * sig * (1.0 - sig);
            dy * grad_silu
        })
        .collect();
    write_f32(find(bindings, 2), &d_x);
}

pub(crate) fn rope_bwd(bindings: &[CpuBinding]) {
    let mut d_vec = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let (seq_len, dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let offset = token_idx * dim + h * head_dim + dim_idx;
                let dx0 = d_vec[offset];
                let dx1 = d_vec[offset + 1];

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = token_idx as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                d_vec[offset] = dx0 * v_cos + dx1 * v_sin;
                d_vec[offset + 1] = -dx0 * v_sin + dx1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &d_vec);
}

pub(crate) fn rope_qk(bindings: &[CpuBinding]) {
    let mut q = read_f32(find(bindings, 0));
    let mut k = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (seq_len, dim, head_dim, row_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let row = row_offset + token_idx;
                let offset = row * dim + h * head_dim + dim_idx;

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = token_idx as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                let q0 = q[offset];
                let q1 = q[offset + 1];
                q[offset] = q0 * v_cos - q1 * v_sin;
                q[offset + 1] = q0 * v_sin + q1 * v_cos;

                let k0 = k[offset];
                let k1 = k[offset + 1];
                k[offset] = k0 * v_cos - k1 * v_sin;
                k[offset + 1] = k0 * v_sin + k1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &q);
    write_f32(find(bindings, 1), &k);
}

pub(crate) fn rope_bwd_qk(bindings: &[CpuBinding]) {
    let mut d_q = read_f32(find(bindings, 0));
    let mut d_k = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (seq_len, dim, head_dim, row_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[3] as usize,
    );
    let num_heads = dim / head_dim;

    for token_idx in 0..seq_len {
        let mut dim_idx = 0usize;
        while dim_idx < head_dim {
            for h in 0..num_heads {
                let row = row_offset + token_idx;
                let offset = row * dim + h * head_dim + dim_idx;

                let freq = 1.0 / 10000f32.powf(dim_idx as f32 / head_dim as f32);
                let angle = token_idx as f32 * freq;
                let (v_sin, v_cos) = angle.sin_cos();

                let dq0 = d_q[offset];
                let dq1 = d_q[offset + 1];
                d_q[offset] = dq0 * v_cos + dq1 * v_sin;
                d_q[offset + 1] = -dq0 * v_sin + dq1 * v_cos;

                let dk0 = d_k[offset];
                let dk1 = d_k[offset + 1];
                d_k[offset] = dk0 * v_cos + dk1 * v_sin;
                d_k[offset + 1] = -dk0 * v_sin + dk1 * v_cos;
            }
            dim_idx += 2;
        }
    }
    write_f32(find(bindings, 0), &d_q);
    write_f32(find(bindings, 1), &d_k);
}

pub(crate) fn rmsnorm_bwd(bindings: &[CpuBinding]) {
    let d_y = read_f32(find(bindings, 0));
    let x = read_f32(find(bindings, 1));
    let weight = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 5));
    let (seq_len, size) = (meta[0] as usize, meta[1] as usize);
    let eps = f32::from_bits(meta[2]);

    let mut d_x = vec![0.0f32; seq_len * size];
    let mut rsqrt_cache = vec![0.0f32; seq_len];

    for row in 0..seq_len {
        let off = row * size;
        let ss: f32 = x[off..off + size].iter().map(|v| v * v).sum();
        let rsqrt = 1.0 / ((ss / size as f32) + eps).sqrt();
        rsqrt_cache[row] = rsqrt;

        let sum_grad: f32 = (0..size)
            .map(|i| {
                let norm_x = x[off + i] * rsqrt;
                d_y[off + i] * weight[i] * norm_x
            })
            .sum();

        for i in 0..size {
            let norm_x = x[off + i] * rsqrt;
            let dy_w = d_y[off + i] * weight[i];
            d_x[off + i] = rsqrt * (dy_w - (norm_x * sum_grad / size as f32));
        }
    }

    write_f32(find(bindings, 3), &d_x);
    write_f32(find(bindings, 4), &rsqrt_cache);
}

pub(crate) fn rmsnorm_weight_bwd(bindings: &[CpuBinding]) {
    let d_y = read_f32(find(bindings, 0));
    let x = read_f32(find(bindings, 1));
    let rsqrt_cache = read_f32(find(bindings, 2));
    let mut d_weight = read_f32(find(bindings, 3));
    let meta = read_u32(find(bindings, 4));
    let (seq_len, size) = (meta[0] as usize, meta[1] as usize);

    for i in 0..size {
        let mut acc = 0.0f32;
        for row in 0..seq_len {
            let off = row * size;
            let norm_x = x[off + i] * rsqrt_cache[row];
            acc += d_y[off + i] * norm_x;
        }
        d_weight[i] += acc;
    }

    write_f32(find(bindings, 3), &d_weight);
}

pub(crate) fn qkv_split(bindings: &[CpuBinding]) {
    let src = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 4));
    let (seq_len, full_dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    let mut q = vec![0.0f32; seq_len * head_dim];
    let mut k = vec![0.0f32; seq_len * head_dim];
    let mut v = vec![0.0f32; seq_len * head_dim];

    for row in 0..seq_len {
        let src_row = row * full_dim;
        let dst = row * head_dim;
        q[dst..dst + head_dim].copy_from_slice(&src[src_row..src_row + head_dim]);
        k[dst..dst + head_dim].copy_from_slice(&src[src_row + head_dim..src_row + 2 * head_dim]);
        v[dst..dst + head_dim]
            .copy_from_slice(&src[src_row + 2 * head_dim..src_row + 3 * head_dim]);
    }

    write_f32(find(bindings, 1), &q);
    write_f32(find(bindings, 2), &k);
    write_f32(find(bindings, 3), &v);
}

pub(crate) fn qkv_scatter(bindings: &[CpuBinding]) {
    let q = read_f32(find(bindings, 0));
    let k = read_f32(find(bindings, 1));
    let v = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 4));
    let (seq_len, full_dim, head_dim) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);

    let mut dst = vec![0.0f32; seq_len * full_dim];
    for row in 0..seq_len {
        let src = row * head_dim;
        let dst_row = row * full_dim;
        dst[dst_row..dst_row + head_dim].copy_from_slice(&q[src..src + head_dim]);
        dst[dst_row + head_dim..dst_row + 2 * head_dim].copy_from_slice(&k[src..src + head_dim]);
        dst[dst_row + 2 * head_dim..dst_row + 3 * head_dim]
            .copy_from_slice(&v[src..src + head_dim]);
    }

    write_f32(find(bindings, 3), &dst);
}

pub(crate) fn flash_attention(bindings: &[CpuBinding]) {
    let q = read_f32(find(bindings, 0));
    let k = read_f32(find(bindings, 1));
    let v = read_f32(find(bindings, 2));
    let mut out = read_f32(find(bindings, 3));
    let mut l_cache = read_f32(find(bindings, 4));
    let meta = read_u32(find(bindings, 5));
    let (seq_len, dim, head_dim, row_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[4] as usize,
    );
    let scale = f32::from_bits(meta[3]);
    let num_heads = dim / head_dim;

    for row in 0..seq_len {
        for head in 0..num_heads {
            let head_off = head * head_dim;
            let q_off = (row_offset + row) * dim + head_off;

            let mut acc = vec![0.0f32; head_dim];
            let mut row_max = f32::NEG_INFINITY;
            let mut row_sum = 0.0f32;

            for j in 0..=row {
                let kv_off = (row_offset + j) * dim + head_off;
                let score: f32 = (0..head_dim).map(|d| q[q_off + d] * k[kv_off + d]).sum::<f32>()
                    * scale;

                let new_max = row_max.max(score);
                let correction = (row_max - new_max).exp();
                let p = (score - new_max).exp();

                row_sum = row_sum * correction + p;
                for d in 0..head_dim {
                    acc[d] = acc[d] * correction + p * v[kv_off + d];
                }
                row_max = new_max;
            }

            let out_off = (row_offset + row) * dim + head_off;
            for d in 0..head_dim {
                out[out_off + d] = acc[d] / row_sum;
            }
            l_cache[row * num_heads + head] = row_max + row_sum.ln();
        }
    }

    write_f32(find(bindings, 3), &out);
    write_f32(find(bindings, 4), &l_cache);
}

pub(crate) fn flash_attention_bwd_d(bindings: &[CpuBinding]) {
    let d_o = read_f32(find(bindings, 0));
    let o = read_f32(find(bindings, 1));
    let mut d_sum = read_f32(find(bindings, 2));
    let meta = read_u32(find(bindings, 3));
    let (seq_len, dim, head_dim, row_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[4] as usize,
    );
    let num_heads = dim / head_dim;

    for row in 0..seq_len {
        for head in 0..num_heads {
            let off = (row_offset + row) * dim + head * head_dim;
            let d_i: f32 = (0..head_dim).map(|d| d_o[off + d] * o[off + d]).sum();
            d_sum[row * num_heads + head] = d_i;
        }
    }

    write_f32(find(bindings, 2), &d_sum);
}

pub(crate) fn flash_attention_bwd_dq(bindings: &[CpuBinding]) {
    let q = read_f32(find(bindings, 0));
    let k = read_f32(find(bindings, 1));
    let v = read_f32(find(bindings, 2));
    let d_sum = read_f32(find(bindings, 3));
    let d_o = read_f32(find(bindings, 4));
    let l_cache = read_f32(find(bindings, 5));
    let mut d_q = read_f32(find(bindings, 6));
    let meta = read_u32(find(bindings, 7));
    let (seq_len, dim, head_dim, row_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[4] as usize,
    );
    let scale = f32::from_bits(meta[3]);
    let num_heads = dim / head_dim;

    for row in 0..seq_len {
        for head in 0..num_heads {
            let head_off = head * head_dim;
            let q_off = (row_offset + row) * dim + head_off;
            let l_i = l_cache[row * num_heads + head];
            let d_i = d_sum[row * num_heads + head];

            let mut dq_acc = vec![0.0f32; head_dim];
            for j in 0..=row {
                let kv_off = (row_offset + j) * dim + head_off;
                let score: f32 = (0..head_dim).map(|d| q[q_off + d] * k[kv_off + d]).sum::<f32>()
                    * scale;
                let p = (score - l_i).exp();

                let dp: f32 = (0..head_dim).map(|d| d_o[q_off + d] * v[kv_off + d]).sum();
                let d_s = p * (dp - d_i);

                for d in 0..head_dim {
                    dq_acc[d] += d_s * k[kv_off + d];
                }
            }

            let dq_off = (row_offset + row) * dim + head_off;
            for d in 0..head_dim {
                d_q[dq_off + d] = dq_acc[d] * scale;
            }
        }
    }

    write_f32(find(bindings, 6), &d_q);
}

pub(crate) fn flash_attention_bwd_dkdv(bindings: &[CpuBinding]) {
    let q = read_f32(find(bindings, 0));
    let k = read_f32(find(bindings, 1));
    let v = read_f32(find(bindings, 2));
    let d_sum = read_f32(find(bindings, 3));
    let d_o = read_f32(find(bindings, 4));
    let l_cache = read_f32(find(bindings, 5));
    let mut d_k = read_f32(find(bindings, 6));
    let mut d_v = read_f32(find(bindings, 7));
    let meta = read_u32(find(bindings, 8));
    let (seq_len, dim, head_dim, row_offset) = (
        meta[0] as usize,
        meta[1] as usize,
        meta[2] as usize,
        meta[4] as usize,
    );
    let scale = f32::from_bits(meta[3]);
    let num_heads = dim / head_dim;

    for col in 0..seq_len {
        for head in 0..num_heads {
            let head_off = head * head_dim;
            let kv_off = (row_offset + col) * dim + head_off;

            let mut dk_acc = vec![0.0f32; head_dim];
            let mut dv_acc = vec![0.0f32; head_dim];

            for i in col..seq_len {
                let qo_off = (row_offset + i) * dim + head_off;
                let l_i = l_cache[i * num_heads + head];

                let score: f32 = (0..head_dim)
                    .map(|d| q[qo_off + d] * k[kv_off + d])
                    .sum::<f32>()
                    * scale;
                let p = (score - l_i).exp();
                let d_i = d_sum[i * num_heads + head];

                let dp: f32 = (0..head_dim).map(|d| d_o[qo_off + d] * v[kv_off + d]).sum();
                let d_s = p * (dp - d_i);

                for d in 0..head_dim {
                    dv_acc[d] += p * d_o[qo_off + d];
                    dk_acc[d] += d_s * q[qo_off + d];
                }
            }

            for d in 0..head_dim {
                d_k[kv_off + d] = dk_acc[d] * scale;
                d_v[kv_off + d] = dv_acc[d];
            }
        }
    }

    write_f32(find(bindings, 6), &d_k);
    write_f32(find(bindings, 7), &d_v);
}

pub(crate) fn grad_sumsq(bindings: &[CpuBinding]) {
    let grad = read_f32(find(bindings, 0));
    let mut partials = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let (len, out_offset) = (meta[0] as usize, meta[1] as usize);

    // Mirrors emit.rs::grad_sumsq_wgs -- must produce exactly this many
    // partials, since the caller sized the `partials` buffer for it.
    let num_wg = (((len as u32 + 255) / 256).clamp(1, 256)) as usize;
    let chunk = (len + num_wg - 1) / num_wg.max(1);

    for w in 0..num_wg {
        let start = (w * chunk).min(len);
        let end = ((w + 1) * chunk).min(len);
        let sumsq: f32 = grad[start..end].iter().map(|v| v * v).sum();
        partials[out_offset + w] = sumsq;
    }

    write_f32(find(bindings, 1), &partials);
}

pub(crate) fn grad_norm_scale(bindings: &[CpuBinding]) {
    let partials = read_f32(find(bindings, 0));
    let mut scale = read_f32(find(bindings, 1));
    let meta = read_u32(find(bindings, 2));
    let num_partials = meta[0] as usize;
    let max_norm = f32::from_bits(meta[1]);

    let sumsq: f32 = partials[..num_partials].iter().sum();
    let norm = sumsq.sqrt();
    scale[0] = if norm > max_norm {
        max_norm / (norm + 1e-6)
    } else {
        1.0
    };

    write_f32(find(bindings, 1), &scale);
}

pub(crate) fn grad_scale(bindings: &[CpuBinding]) {
    let mut grad = read_f32(find(bindings, 0));
    let scale = read_f32(find(bindings, 1));
    for g in grad.iter_mut() {
        *g *= scale[0];
    }
    write_f32(find(bindings, 0), &grad);
}
