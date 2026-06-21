use akasha_core::Real;
use akasha_core::nn::akasha_model::AkashaModel;
use filuplex::context::{Context, GpuPref};
use filuplex::ops::GpuBuffer;
use std::sync::Arc;

fn main() {
    println!("Smoke test starting");

    let ctx = Arc::new(Context::new(GpuPref::Default));

    let vocab_size = 50304;
    let dim = 1024;
    let seq_len = 128;
    let num_layers = 12;

    println!(
        "Model Config: {} Layer, {} Dim, {} SeqLen",
        num_layers, dim, seq_len
    );

    let mut tokens_cpu = vec![0.0 as Real; seq_len as usize];
    for i in 0..seq_len {
        tokens_cpu[i as usize] = ((i * 13) % vocab_size) as Real;
    }

    let input_tokens = GpuBuffer::from_cpu(&tokens_cpu, &ctx);

    println!("Akasha going to gpu (Pipeline Starting)...");

    let model = AkashaModel::new(
        ctx.clone(),
        vocab_size,
        dim,
        seq_len,
        num_layers,
        &input_tokens,
    );

    println!("Forward Pass is started");

    let start_time = std::time::Instant::now();
    model.forward(ctx.clone());
    println!("Forward Pass Complated! Time: {:?}", start_time.elapsed());

    let out_size = (seq_len * vocab_size) as usize;
    let cpu_out = vec![0.0 as Real; out_size];

    println!("datas from GPU goes to CPU...");

    let cpu_out = model.lm_head.out_buffer.to_cpu(&ctx);

    println!("Test Completad, thats are values:");
    println!("{:#?}", &cpu_out[0..10]);
}
