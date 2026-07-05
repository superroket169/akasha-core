use wilupgu::CpuBinding;

fn find(bindings: &[CpuBinding], slot: u32) -> &CpuBinding {
    bindings
        .iter()
        .find(|b| b.slot == slot)
        .expect("missing binding slot")
}

fn read_f32(b: &CpuBinding) -> Vec<f32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::cast_slice::<u8, f32>(&g).to_vec()
}

fn read_u32(b: &CpuBinding) -> Vec<u32> {
    let g = b.buffer.lock().unwrap();
    bytemuck::cast_slice::<u8, u32>(&g).to_vec()
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

pub(crate) fn softmax(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let seq_len = meta[0] as usize;

    for row in 0..seq_len {
        let off = row * seq_len;
        let max_val = x[off..off + seq_len]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for i in 0..seq_len {
            let e = (x[off + i] - max_val).exp();
            x[off + i] = e;
            sum_exp += e;
        }
        for i in 0..seq_len {
            x[off + i] /= sum_exp;
        }
    }
    write_f32(find(bindings, 0), &x);
}

pub(crate) fn causal_softmax(bindings: &[CpuBinding]) {
    let mut x = read_f32(find(bindings, 0));
    let meta = read_u32(find(bindings, 1));
    let seq_len = meta[0] as usize;
    let scale = f32::from_bits(meta[1]);

    fn masked(x: &[f32], off: usize, row: usize, i: usize, scale: f32) -> f32 {
        if i > row {
            -1_000_000_000.0
        } else {
            x[off + i] * scale
        }
    }

    for row in 0..seq_len {
        let off = row * seq_len;
        let max_val = (0..seq_len)
            .map(|i| masked(&x, off, row, i, scale))
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for i in 0..seq_len {
            let e = (masked(&x, off, row, i, scale) - max_val).exp();
            x[off + i] = e;
            sum_exp += e;
        }
        for i in 0..seq_len {
            x[off + i] /= sum_exp;
        }
    }
    write_f32(find(bindings, 0), &x);
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

pub(crate) fn cross_entropy(bindings: &[CpuBinding]) {
    let logits = read_f32(find(bindings, 0));
    let targets = read_u32(find(bindings, 1));
    let meta = read_u32(find(bindings, 4));
    let (vocab_size, num_rows) = (meta[0] as usize, meta[1] as usize);

    let mut probs = vec![0.0f32; num_rows * vocab_size];
    let mut losses = vec![0.0f32; num_rows];

    for row in 0..num_rows {
        let off = row * vocab_size;
        let target_id = targets[row] as usize;

        let max_val = logits[off..off + vocab_size]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        for i in 0..vocab_size {
            let e = (logits[off + i] - max_val).exp();
            probs[off + i] = e;
            sum_exp += e;
        }
        for i in 0..vocab_size {
            probs[off + i] /= sum_exp;
        }

        let log_sum_exp = sum_exp.ln();
        losses[row] = -(logits[off + target_id] - max_val - log_sum_exp);
    }

    write_f32(find(bindings, 2), &probs);
    write_f32(find(bindings, 3), &losses);
}
