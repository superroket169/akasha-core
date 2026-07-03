use crate::Real;
use crate::config::ModelConfig;
use rand::Rng;
use std::sync::Arc;
use wilupgu::{Backend, Tensor};

fn xavier_std(fan_in: u32) -> Real {
    1.0 / (fan_in as Real).sqrt()
}

static INIT_RNG: std::sync::OnceLock<std::sync::Mutex<rand::rngs::StdRng>> =
    std::sync::OnceLock::new();
fn init_rng() -> std::sync::MutexGuard<'static, rand::rngs::StdRng> {
    INIT_RNG
        .get_or_init(|| std::sync::Mutex::new(rand::SeedableRng::seed_from_u64(42)))
        .lock()
        .unwrap()
}

fn random_normal_vec(len: usize, mean: Real, std: Real) -> Vec<Real> {
    let mut rng = init_rng();
    (0..len)
        .map(|_| {
            let u1: f32 = rng.gen_range(1e-9_f32..1.0);
            let u2: f32 = rng.gen_range(0.0_f32..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + z0 * std
        })
        .collect()
}

pub struct BlockWeights<B: Backend> {
    pub norm_1: Arc<Tensor<B>>,
    pub qkv_proj: Arc<Tensor<B>>, // fused [dim, 3*dim]
    pub out_proj: Arc<Tensor<B>>,
    pub norm_2: Arc<Tensor<B>>,
    pub ffn_up: Arc<Tensor<B>>,
    pub ffn_down: Arc<Tensor<B>>,
}

pub struct ModelWeights<B: Backend> {
    pub cfg: ModelConfig,
    pub embedding: Arc<Tensor<B>>, // [vocab_size, dim]
    pub blocks: Vec<BlockWeights<B>>,
    pub final_norm: Arc<Tensor<B>>,
    pub lm_head: Arc<Tensor<B>>, // untied from embedding
}

fn interleave_qkv(dim: u32, q_w: &[Real], k_w: &[Real], v_w: &[Real]) -> Vec<Real> {
    let dim = dim as usize;
    let mut combined = Vec::with_capacity(dim * 3 * dim);
    for r in 0..dim {
        combined.extend_from_slice(&q_w[r * dim..(r + 1) * dim]);
        combined.extend_from_slice(&k_w[r * dim..(r + 1) * dim]);
        combined.extend_from_slice(&v_w[r * dim..(r + 1) * dim]);
    }
    combined
}

impl<B: Backend> ModelWeights<B> {
    pub fn random(ctx: Arc<B>, cfg: &ModelConfig) -> Self {
        let ModelConfig {
            vocab_size,
            dim,
            num_layers,
            ffn_hidden,
            ..
        } = *cfg;
        let t = |data: &[Real]| Arc::new(Tensor::init_from_cpu(ctx.clone(), data));

        let embedding = t(&random_normal_vec(
            (vocab_size * dim) as usize,
            0.0,
            xavier_std(dim),
        ));

        let proj_std = xavier_std(dim);
        let blocks = (0..num_layers)
            .map(|_| {
                let norm_1 = t(&random_normal_vec(dim as usize, 1.0, 0.02));
                let q_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
                let k_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
                let v_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
                let out_w = random_normal_vec((dim * dim) as usize, 0.0, proj_std);
                BlockWeights {
                    norm_1,
                    qkv_proj: t(&interleave_qkv(dim, &q_w, &k_w, &v_w)),
                    out_proj: t(&out_w),
                    norm_2: t(&random_normal_vec(dim as usize, 1.0, 0.02)),
                    ffn_up: t(&random_normal_vec(
                        (dim * ffn_hidden) as usize,
                        0.0,
                        xavier_std(dim),
                    )),
                    ffn_down: t(&random_normal_vec(
                        (ffn_hidden * dim) as usize,
                        0.0,
                        xavier_std(ffn_hidden),
                    )),
                }
            })
            .collect();

        let final_norm = t(&random_normal_vec(dim as usize, 1.0, 0.02));
        let lm_head = t(&random_normal_vec(
            (dim * vocab_size) as usize,
            0.0,
            xavier_std(dim),
        ));

        Self {
            cfg: *cfg,
            embedding,
            blocks,
            final_norm,
            lm_head,
        }
    }

    pub fn zeros(ctx: Arc<B>, cfg: &ModelConfig) -> Self {
        let ModelConfig {
            vocab_size,
            dim,
            num_layers,
            ffn_hidden,
            ..
        } = *cfg;
        let z = |n: u32| {
            Arc::new(Tensor::init_from_cpu(
                ctx.clone(),
                &vec![0.0 as Real; n as usize],
            ))
        };

        Self {
            cfg: *cfg,
            embedding: z(vocab_size * dim),
            blocks: (0..num_layers)
                .map(|_| BlockWeights {
                    norm_1: z(dim),
                    qkv_proj: z(dim * dim * 3),
                    out_proj: z(dim * dim),
                    norm_2: z(dim),
                    ffn_up: z(dim * ffn_hidden),
                    ffn_down: z(ffn_hidden * dim),
                })
                .collect(),
            final_norm: z(dim),
            lm_head: z(dim * vocab_size),
        }
    }

    pub fn params(&self) -> Vec<Arc<Tensor<B>>> {
        let mut params = vec![self.embedding.clone()];
        for b in &self.blocks {
            params.push(b.norm_1.clone());
            params.push(b.qkv_proj.clone());
            params.push(b.out_proj.clone());
            params.push(b.norm_2.clone());
            params.push(b.ffn_up.clone());
            params.push(b.ffn_down.clone());
        }
        params.push(self.final_norm.clone());
        params.push(self.lm_head.clone());
        params
    }
}
