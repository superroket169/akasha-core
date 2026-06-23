use std::sync::Arc;
use wilupgu::context::WgpuContext;
use wilupgu::tensor::Tensor;

fn main() {
    println!("[ROUND TRIP TEST] Başlıyor...");

    let ctx = Arc::new(pollster::block_on(async { WgpuContext::new().await }));

    let test_size = 10;
    let mut cpu_data_in = vec![0.0f32; test_size];
    for i in 0..test_size {
        cpu_data_in[i] = (i + 1) as f32;
    }

    println!(
        "Gönderilen ilk değer: {}, son değer: {}",
        cpu_data_in[0],
        cpu_data_in[test_size - 1]
    );

    let tensor = Tensor::init_from_cpu(ctx.clone(), &cpu_data_in);
    println!("Veri VRAM'e yazıldı. Şimdi shader OLMADAN doğrudan geri çekiliyor...");

    let cpu_data_out: Vec<f32> = tensor.to_cpu();

    println!(
        "[TEST SONUCU] Çekilen ilk değer: {}, Son değer: {}",
        cpu_data_out[0],
        cpu_data_out[test_size - 1]
    );

    if cpu_data_out[0] == 1.0 && cpu_data_out[test_size - 1] == 10.0 {
        println!("BAŞARILI! Staging -> VRAM -> Staging hattı KUSURSUZ çalışıyor.");
        println!("SONUÇ: Hata %100 Compute Shader'ın içinde veya forward'daki queue.submit'te.");
    } else {
        println!("HATA! Veri VRAM'den bozuk geldi");
    }
}
