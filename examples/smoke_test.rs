use akasha_core::nn::akasha_model::AkashaModel;
use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

fn main() {
    println!("[SMOKE TEST] Başlıyor...");

    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));

    let vocab_size = 50304;
    let dim = 1024;
    let seq_len = 128;
    let num_layers = 1;

    println!(
        "Model Config: {} Katman, {} Boyut, {} SeqLen, {} Vocab",
        num_layers, dim, seq_len, vocab_size
    );

    let mut tokens_cpu = vec![0u32; seq_len as usize];
    for i in 0..seq_len {
        tokens_cpu[i as usize] = ((i * 13) % vocab_size) as u32;
    }

    let t_input_tokens = Arc::new(Tensor::init_from_cpu(ctx.clone(), &tokens_cpu));

    println!("Akasha going to vram...");
    let model = AkashaModel::new(
        ctx.clone(),
        vocab_size,
        dim,
        seq_len,
        num_layers,
        &t_input_tokens,
    );

    println!("Forward Pass...");
    let start_time = std::time::Instant::now();

    model.forward();

    println!("Forward Pass Tamamlandı! Süre: {:?}", start_time.elapsed());

    println!("Results comes from vram...");
    let cpu_out: Vec<f32> = model.lm_head.out_buffer.to_cpu();

    println!("[TEST BAŞARILI] İlk 10 Değer:");
    println!("{:#?}", &cpu_out[0..10]);
}
