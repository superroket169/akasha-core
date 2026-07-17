# Gerçek Batch Size

## Neden

Şu an production `train_step`'in `batch_size` parametresi host tarafında
sıralı bir Rust `for` döngüsü -- `accumulation_steps` ile mimari olarak
birebir aynı şey, hiçbir gerçek paralellik yoktu. `ComputeGraph` hep TEK bir
sequence (`seq_len` uzunluğunda) için kuruluydu.

Hedef: batch_size'ı gerçek bir tensor boyutu yapmak -- B sequence'ı TEK bir
compute-graph execution'ında birlikte işlemek. Motivasyon: 4050'nin VRAM'i
şu an ~1/4'ünden bile az kullanılıyor, GPU büyük ihtimalle batch=1'de
doymuyor (FFN matmul'leri baskın maliyet, attention'dan ~6x pahalı).

**Dürüst beklenti:** yön doğru, ama gerçek ölçülen hızlanma tam Bx değil,
muhtemelen bant genişliği/sabit-overhead sınırları yüzünden biraz altında
olabilir. Yön kesin doğru, sayı tahmini.

## Gerçekleşen tasarım (ilk taslaktan DAHA BASİT çıktı)

İlk taslakta RoPE/FlashAttention için "3D grid + modulo pozisyon" planlanmıştı.
Kernel kaynağını (`cache_write.wgsl`'nin zaten kullandığı `dst_row_offset`
deseni) okuduktan sonra çok daha küçük bir değişiklikle aynı sonuca ulaşıldı:

- **Trivial kategori** (matmul ailesi, RMSNorm, embedding, head_gather/
  scatter, qkv_split/scatter, silu, add, zero, cross_entropy): SIFIR kernel
  değişikliği. Bu op'ların "seq_len" parametresi zaten sadece "satır sayısı"
  -- çağıran taraf `rows = batch_size * seq_len` geçiyor, kernel hiç
  değişmedi.
- **RoPE (rope_qk/rope_bwd_qk) ve FlashAttention (fwd/bwd_dq/bwd_dkdv):**
  Meta struct'a tek bir `row_offset: u32` alanı eklendi (struct'ın SONUNA --
  byte-layout'u legacy/test-only tüketicilerle (rope/rope_bwd, l_cache
  indexleme) uyumlu tutmak için). Kernel'in causal döngü sınırı VE pozisyon/
  açı hesabı yerel kalıyor (`0..=row`, `f32(token_idx)*freq`) -- SADECE
  q/k/v/out/grad adresleme hesaplarına `+ row_offset` eklendi. `l_cache`
  paylaşılmıyor (her `b` için ayrı, küçük, private scratch tensor), o yüzden
  onun indexlemesi hiç değişmedi.
  Sonuç: her batch item'ı için kernel AYNI (seq_len boyutlu) dispatch ile,
  ama artık B kere host loop yerine TEK static graph içinde B node olarak
  inşa ediliyor (construction-time'da bir kere, training step başına değil).
  Causal pencere hiçbir zaman `seq_len` dışına çıkmıyor -> batch item'lar
  arası "sızıntı" yapısal olarak imkansız (ayrı bir modulo/bounds mantığına
  gerek yok).

Bu tasarım `flash_attention`/`rope_qk`'nin emit.rs imzalarını DEĞİŞTİRMEDİ
(zaten `shape: FlashAttnMeta`/`RopeMeta` by-value alıyorlardı) -- sadece
çağıran taraf (`SelfAttention::new`, `TransformerBlock::new`) artık `b in
0..batch_size` döngüsüyle B kere çağırıyor, her seferinde farklı
`row_offset`.

## Uygulanan değişiklikler

1. `src/config.rs`: `ModelConfig.batch_size: u32` (default 1 via `::new()`)
   + `.with_batch_size(n)` builder. Opt-in, sıfır davranış değişikliği.
2. `src/nn/ops/meta.rs`: `RopeMeta`/`FlashAttnMeta`'ya `row_offset: u32`
   (sonda, byte-uyumlu).
3. WGSL: `rope_qk.wgsl`, `rope_bwd_qk.wgsl`, `flash_attention.wgsl`,
   `flash_attention_bwd_dq.wgsl`, `flash_attention_bwd_dkdv.wgsl` --
   Meta struct + adresleme güncellendi. Legacy `rope.wgsl`/`rope_bwd.wgsl`
   (test-only 2-call path) DOKUNULMADI (trailing byte'ları güvenle yok
   sayıyorlar).
4. `src/shaders/cuda.rs`: aynı 5 kernel'in CUDA-C string'lerine `row_offset`
   trailing param eklendi (WGSL ile birebir aynı desen).
5. `wilupgu/src/backends/cuda.rs`: `launch_rope_qk`, `launch_flash_attention`,
   `launch_flash_attention_bwd_dq`, `launch_flash_attention_bwd_dkdv`
   macro çağrılarına `row_offset` meta alanı + launch arg eklendi (sondaki
   pozisyonda, `read_meta!` sadece deklare edileni okuyup gerisini yok
   sayıyor -- legacy `launch_rope` dokunulmadı).
6. `src/nn/layers.rs`:
   - `SelfAttention::new` artık `batch_size: u32` alıyor, forward/backward
     graph'lara B ayrı flash_attention/flash_attention_bwd node'u ekliyor
     (her biri kendi `row_offset`, kendi küçük `l_cache`'i ile).
   - `TransformerBlock::new`: `rows = batch_size * seq_len` hesaplanıp
     trivial op'ların HEPSİNE (`RMSNorm`, `Linear` x4, `Add` x2, `SiLU`,
     `HeadMoveMeta::qkv_slice` x2) `seq_len` yerine `rows` geçiliyor.
     RoPE forward/backward graph'ları `for b in 0..batch_size` ile B node
     inşa ediyor.
7. `src/nn/train.rs`: `Trainer::new` aynı `rows` mantığını `Embedding`,
   `RMSNorm` (final_norm), `Linear` (lm_head), `CrossEntropy` için
   uyguluyor. **`train_step`'in kendisi HİÇ DEĞİŞMEDİ** (bkz. aşağıdaki
   "Nasıl kullanılır" bölümü -- gerek kalmadı).

## Nasıl kullanılır (mevcut haliyle, main.rs'e HENÜZ dokunulmadı)

`Trainer::train_step(input_tokens, target_tokens, batch_size, step,
accumulation_steps)` -- BURADAKİ `batch_size` parametresi `cfg.batch_size`
İLE AYNI ŞEY DEĞİL, host-loop tekrar sayısı (bkz. fonksiyonun içindeki
`for i in 0..batch_size` -- bu hiç değişmedi). `cfg.batch_size = B` ile
kurulmuş bir `Trainer`'ı gerçek batching ile kullanmak için:

- `input_tokens` buffer'ını `B * seq_len` eleman olarak allocate et (tüm B
  sequence art arda).
- `train_step(...)`'i kendi `batch_size` argümanına **1** vererek çağır
  (host loop'un TEK dönmesini, tüm `rows` satırını tek seferde kopyalayıp
  tek `execute_captured()` çağırmasını sağlar).
- `accumulation_steps` aynen kullanılmaya devam eder (donanımın
  kaldıramadığı effective-batch büyümesi için).

Bu sayede `train_step`'e dokunmadan (production'ı hiç riske atmadan) yeni
kapasite kullanılabilir hale geldi -- main.rs'i gerçekten `cfg.batch_size>1`
ile çalıştırmaya geçirmek (buffer boyutlarını güncellemek, VRAM'e göre B
seçmek) hâlâ ayrı, bilinçli bir adım, henüz YAPILMADI.

## Doğrulama

`src/nn/train.rs::batching_validation::real_batching_matches_sequential_accumulation`
(WgpuBackend, CUDA gerekmez, bu makinede PASS): aynı ağırlıklar + aynı 3
sequence'lık token verisiyle, (a) `batch_size=1` Trainer'ı 3 kere sıralı
çağırıp gradyanları accumulate etmek (bugünkü production yolu) ile (b)
`batch_size=3` Trainer'ı TEK seferde çalıştırmak -- loss, son batch item'ın
forward çıktısı, VE tüm `trainable_params()` gradyanları 1e-3 tolerans
içinde birebir eşleşiyor. `cargo test --lib` (default/vulkan feature) ile
tüm 6 test (bu yeni test dahil) yerel olarak PASS.

**CUDA tarafı (adım 4-5 -- `cuda.rs` string'leri + `wilupgu` launch
macro'ları) bu makinede test EDİLEMEDİ (CUDA donanımı yok).** Aynen bu
oturumdaki f16/bf16 çalışması gibi -- WGSL ile birebir aynı, basit,
mekanik bir değişiklik olduğu için düşük risk olarak değerlendirildi, ama
gerçek doğrulama arkadaşın RTX 4050'sinde bir round-trip gerektiriyor.
`tester_script.bat`'a bu -- `cargo test --features cuda batching -- ...`
gibi bir adım eklenip friend'e gönderilebilir (henüz eklenmedi).

## Sıradaki adımlar

1. ✅ (2026-07-17) CUDA-side doğrulama: `batching_validation` test gövdesi
   backend-jenerik `batching_parity<B>()` fonksiyonuna çıkarıldı; wgpu testi
   onu çağırıyor, yeni `real_batching_matches_sequential_accumulation_cuda`
   (cfg cuda) aynı gövdeyi GERÇEK CudaBackend'de koşuyor — akasha
   tester_script.bat'ın 2. adımı (`cargo test --lib --features cuda`) bunu
   otomatik kapsıyor. Arkadaşın makinesinde koşması bekleniyor.
2. ✅ (2026-07-17) `main.rs` gerçek batch'e geçti:
   `akasha_hall_1().with_batch_size(BATCH_SIZE)`, input_tokens `B*seq_len`
   allocate, `random_batch(BATCH_SIZE)` çıktısı `train_step(..., 1, ...)`
   ile besleniyor. config.rs: BATCH_SIZE=4 (4050'de VRAM'e göre kalibre
   edilecek; logits tek başına B×~103MB), ACCUMULATION_STEPS=16 → effective
   batch 64 DEĞİŞMEDİ. Sacred model_final.bin hâlâ dokunulmadı; devam eden
   koşular checkpoint formatı/şeması değişmediği için etkilenmez.
3. `rmsnorm_weight_bwd.wgsl`'nin iç döngü sınırının `seq_len`'e değil zaten
   genel satır sayısına bağlı olduğu (yani `rows` ile de doğru çalıştığı)
   -- yukarıdaki test bunu dolaylı olarak zaten doğruladı (RMSNorm
   backward'ları test'te 2 katmanlı bir modelde çalıştı ve gradyanlar
   eşleşti), ayrı bir kontrole gerek kalmadı.

Kalan (F1'in bilinçli DIŞINDA bırakılan kısmı): prefill/decode tarafında
batch — pretraining için gereksiz, Big Refactor'un Block yüzeyi bunu zaten
yeniden şekillendirecek.
