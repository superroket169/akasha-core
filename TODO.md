# TODO

## Mamba/GDN — heterojen stack

Transformer-only mimari sabit; aşağıdakiler heterojen (`[Attn, Attn, Mamba, ...]`)
stack'e geçişin adımları, hiçbiri başlanmadı.

- **Config-driven heterojen stack** — `Vec<BlockSpec>` + checkpoint header V4
  uzantısı (V3 homojen: dim/heads/layers/ffn). İlk günden tasarlanmalı ki
  üçüncü format migrasyonu yaşanmasın.
- **Gated DeltaNet** — ilk attention-dışı blok, Mamba-2'den ÖNCE: chunk'lı
  matmul'lara ayrışır (conv1d'siz), decode sabit-state, soyutlamayı en ucuz
  attention-dışı blokla doğrular.
- **Mamba-2 (SSD formülasyonu)** — sequential selective scan yerine
  chunk'lı matmul'lara ayrışır, mevcut matmul/cuBLAS altyapısına oturur.
  Inference-first: fwd/decode önce yazılıp hazır bir checkpoint import'uyla
  (ör. state-spaces/mamba-130m) doğrulanır, bwd (en zor kernel) sonra.
- **Jamba/hibrit = config satırı** — yukarıdaki ikisi bittiğinde yeni model
  serisi buradan başlar.
- **GQA** — heterojen stack'ten bağımsız, ucuz (KV head sayısını düşürmek
  KV cache + decode bant trafiğini `heads/kv_heads` kat küçültür). **Karar
  gerektirir**: mimari değişikliği → sıfırdan init, eğitilmiş weight'lerden
  continued pretraining ile bağdaşmaz. Sonraki koşu sıfırdansa gir, continued
  ise girme.
- **`Architecture` trait'inin gelecek ihtiyaç listesi** (Mamba/GDN günü
  kontrol listesi — hepsini implemente etme, ama hiçbirini imkânsız kılma):
  - Token-mixer / channel-mixer ayrımı (MoE = FFN koltuğunun alternatifi).
  - MoE ihtiyaçları: device'ta router top-k, expert weight'lerinin
    `params()` sırasına girişi, aux-loss hook'u (load-balancing).
  - Pozisyon bilgisi RoPE gibi attention BLOĞUNA ait olmalı, global
    pipeline'a değil (Mamba/GLA pozisyonsuz).
  - Init'in stack bağlamı — E2'nin 1/√(2L)'si L = blok sayısı, heterojen
    stack'te init bunu parametre almalı.
  - Per-blok dtype — arayüz weight dtype'ını blok başına taşıyabilmeli.
  - `BlockSpec` hiperparamları (head/kv_head, ffn boyutu, SSM state boyutu)
    per-blok, config'te `Vec<BlockSpec>` içinde.
  - `Tape` türü blok impl'ine ait olmalı — attention l_cache ister, GLA
    chunk state'leri ister; trait tek somut `TrainOp` dayatmasın.
  - State handle "cache" değil "adım durumu" — büyüyen (KV) ve sabit
    (SSM/linear-attn) state aynı arayüzü taşımalı.

## Diğer

- **Decode'u cuBLAS'sızlaştırmak** — bkz. [wilupgu/TODO.md](../wilupgu/TODO.md),
  bu repoyu da doğrudan etkiler (GEMV/GEMV_ADD emitter'ları burada).
- **Binding-seviyeli `DynamicMeta` işareti** — `Shader.layout`'a konamaz
  (aynı shader train'de sabit, decode'da dinamik meta alır; işaret
  binding'e ait). Kazançlar: CUDA `build_node` dinamik meta için bayat
  `cached_meta` üretmez; `execute_captured`, `DynamicMeta` içeren graph'ta
  assert'le durur — "decode capture edilemez" kuralı yorumdan koda iner.
- **`train_step`'in her adımda loss okuması** — eski `Trainer::train_step`
  loss'u yalnız `log_every` adımlarında CPU'ya okuyordu; `Model::train_step`
  şimdi her adımda okuyor. Uzun eğitimlerde fark edebilir, ölçülmedi.
- **GPU loss recorder ("loss_counter")** — CE loss zaten GPU'da
  hesaplanıyor; kayıt aralığında bir kez küçük bir kernel losses'ı
  indirgesin ve sabit boyutlu history buffer'ına yazsın — eğri sonda TEK
  dtoh ile iner.
