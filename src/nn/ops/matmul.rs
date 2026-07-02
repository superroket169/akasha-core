use super::meta::{KernelMeta, MatMulMeta};
use std::sync::Arc;
use wilupgu::{Backend, Binding, ComputeGraph, Tensor, TensorMode};

fn grid_nm(shape: MatMulMeta) -> [u32; 3] {
    [(shape.n + 15) / 16, (shape.m + 15) / 16, 1]
}

/// `C[m,n] = A[m,k] @ B[k,n]`
pub(crate) fn matmul_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "MatMul",
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_with(graph, a, b, c, shape, &meta);
}

/// `C[m,n] = A[m,k] @ B[n,k]^T`
pub(crate) fn matmul_trp_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "MatMulTrp",
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul_trp<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_trp_with(graph, a, b, c, shape, &meta);
}

/// `C[m,n] += A[m,k] @ B[k,n]` (fused residual, `c` is InOut)
pub(crate) fn matmul_add_with<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
    meta: &Arc<Tensor<B>>,
) {
    graph.add_node(
        "MatMulAdd",
        &[
            Binding::new(0, &a.buffer, TensorMode::Input),
            Binding::new(1, &b.buffer, TensorMode::Input),
            Binding::new(2, &c.buffer, TensorMode::InOut),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        grid_nm(shape),
    );
}

pub(crate) fn matmul_add<B: Backend>(
    graph: &mut ComputeGraph<B>,
    a: &Arc<Tensor<B>>,
    b: &Arc<Tensor<B>>,
    c: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&a.ctx);
    matmul_add_with(graph, a, b, c, shape, &meta);
}

/// `dW[k,n] += A[m,k]^T @ dY[m,n]` -- accumulates, zero `grad_weight` first.
pub(crate) fn matmul_weight_bwd<B: Backend>(
    graph: &mut ComputeGraph<B>,
    input: &Arc<Tensor<B>>,
    grad_output: &Arc<Tensor<B>>,
    grad_weight: &Arc<Tensor<B>>,
    shape: MatMulMeta,
) {
    let meta = shape.upload(&input.ctx);
    graph.add_node(
        "MatMulWeightBwd",
        &[
            Binding::new(0, &input.buffer, TensorMode::Input),
            Binding::new(1, &grad_output.buffer, TensorMode::Input),
            Binding::new(2, &grad_weight.buffer, TensorMode::Output),
            Binding::new(3, &meta.buffer, TensorMode::Meta),
        ],
        [(shape.n + 15) / 16, (shape.k + 15) / 16, 1],
    );
}
