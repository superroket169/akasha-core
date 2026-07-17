# akasha-core — Mimari

## Ne bu?

Genel tanıtım ve hızlı başlangıç [README.md](README.md)'de. Bu dosya iç
mimariyi anlatır: katman haritası, faz tip sistemi, grad topolojisi, meta
protokolü, KV cache, invariantlar ve fikir kuyruğu.

## Katman haritası

```
config.rs ── ModelConfig + akasha-hall-1.0 (ve gelecek modellerin) sabitleri
   │
   ▼
weights.rs ── ModelWeights / BlockWeights          ← TEK GERÇEK:
   │            (salt weight tensörleri, grad yok)   iki dünya da bunu paylaşır
   │
   ├───────────── TRAIN dünyası ────────────────────────────────────┐
   │  layers.rs   : grad'lı sarmalayıcılar (Linear, RMSNorm,        │
   │                Embedding, SelfAttention, Add, SiLU, CE)        │
   │                — her biri kendi fwd/bwd graph'ını kurar        │
   │  train.rs    : Trainer — hepsini fuse eder (fwd, bwd,          │
   │                zero_grads, zero_transient, clip)               │
   │  optim/      : AdamW (schedule kernel + param başına node)     │
   │                                                                │
   ├───────────── INFERENCE dünyası ────────────────────────────────┤
   │  inference_graphs.rs : build_prefill_layer / build_decode_layer│
   │                        + DecodeScratch (buffer+meta havuzu)    │
   │                        + Cache (KV)                            │
   │  inference.rs        : InferenceSession (prefill/decode/       │
   │                        generate akış kontrolü)                 │
   │  sampling.rs         : host-side token seçimi                  │
   │                                                                │
   ▼                                                                ▼
nn/ops ── meta.rs: typed meta struct'lar (KernelMeta)
          emit.rs: kernel başına TEK emitter fonksiyonu
          GraphBuilder<Phase>: Train / Prefill / Decode tip kapıları
   │
   ▼
shaders/ ── akasha'ya özgü kernel üçüzleri (wgsl + cuda + cpu)
   │
   ▼
wilupgu ── builtin'ler (matmul ailesi, GEMV, ADAMW, ZERO_TENSOR,
           RESIDUAL_ADD...) + ComputeGraph + Backend'ler
```

İki dünya kuralı: `layers.rs` **sadece** Trainer'a aittir (grad buffer'ları taşır),
`inference_graphs.rs` **sadece** inference'a. İkisi de aynı `ModelWeights`'i ve aynı
`emit` katmanını kullanır ama birbirinin dosyasına asla dokunmaz.

## nn/ops: GraphBuilder ve faz tip sistemi

Bu bölüm yalnızca `nn/ops/`'u anlatır (GraphBuilder, Phase trait'leri, emitter'lar).
Graph'lerin *kurulduğu* yerler başka dosyalardır: Train graph'leri `layers.rs` +
`train.rs`'de, Prefill/Decode graph'leri `inference_graphs.rs`'de kurulur
(`inference.rs` sadece akış kontrolüdür). "Prefill" adını görünce `inference.rs`'e
değil, `inference_graphs.rs`'e bak.

### GraphBuilder ne yapar (ve ne yapmaz)

`GraphBuilder<'g, B, P>` çalışma zamanında hiçbir şey yapmaz: içinde graph'a bir
`&mut` ve sıfır byte'lık `PhantomData<P>` vardır. "Etiketleme" bir işlem değildir —
**etiket, tipin kendisidir**:

```rust
let gb = GraphBuilder::prefill(&mut g); // tip: GraphBuilder<'_, B, Prefill>
let gb = GraphBuilder::decode(&mut g);  // tip: GraphBuilder<'_, B, Decode>
```

Üç constructor'ın gövdesi bilerek aynıdır; farkları hangi
`impl GraphBuilder<'g, B, ___>` bloğunda yaşadıkları, yani hangi `P` ile
döndükleridir. Kapı, emitter imzasındaki bound'dadır:

```rust
pub(crate) fn flash_attention<B: Backend, P: FullSeqPhase>(gb: &mut GraphBuilder<'_, B, P>, ...)
```

`GraphBuilder<B, Decode>` ile çağrılırsa derleyici `impl FullSeqPhase for Decode`
arar, bulamaz, **derlemez**. Mekanizmanın tamamı bu tek aramadır. `PhantomData`
sadece Rust'ın "tanımlanan tip parametresi struct'ta bir alanda geçmeli" kuralını
sıfır maliyetle karşılar.

Sıra bilgisiyle ilgisi yoktur: node sırası = emitter'ları çağırma sıran
(`add_node` bir Vec'e append eder). Faz sistemi sıralayıcı değil, kapı görevlisidir.

### Fazlar ve rozetleri

| Faz | FwdPhase | FullSeqPhase (kare causal attn / full-seq RoPE) | CachedPhase (KV cache okur/yazar) |
|---|---|---|---|
| Train | ✓ | ✓ | — |
| Prefill | ✓ | ✓ | ✓ |
| Decode | ✓ | — | ✓ |

Tam üyelik listeleri:

- `FwdPhase`     = { Train, Prefill, Decode }
- `FullSeqPhase` = { Train, Prefill }
- `CachedPhase`  = { Prefill, Decode }
- Trait'siz, somut tipe kilitli emitter'lar da vardır: bwd/loss/clip ailesi
  doğrudan `GraphBuilder<B, Train>`, cached-attention ailesi doğrudan
  `GraphBuilder<B, Decode>` ister.

Prefill'in iki rozet taşıması bilinçlidir: prompt'u Train gibi işler (tüm satırlar,
kare causal attention) *ve* ürettiği K/V'yi cache'e yazar. Katman i+1'in K/V'si
katman i'nin tam çıktısına bağlı olduğundan prefill zorunlu olarak tam bir forward
pass'tir; cache yazımı her katmanın yan etkisidir.

### Emitter kataloğu

Kapıya göre gruplu tam liste (emit.rs'in haritası):

| Kapı | Girebilenler | Emitter'lar |
|---|---|---|
| `P: FwdPhase` | üç faz da | matmul, matmul_trp, matmul_add, rmsnorm, embedding, head_gather, qkv_split, qkv_scatter, silu, silu_out, residual_add, add_out, zero |
| `P: FullSeqPhase` | Train, Prefill | rope, rope_qk, flash_attention |
| `P: CachedPhase` | Prefill, Decode | cache_write |
| somut `Train` | yalnız Train | matmul_weight_bwd, rmsnorm_bwd, embedding_bwd, rope_bwd_qk, flash_attention_bwd, silu_bwd, add_inplace_bwd, cross_entropy, cross_entropy_bwd, grad_sumsq, grad_norm_scale, grad_scale |
| somut `Decode` | yalnız Decode | rope_offset_with, attn_qk_cached_with, attn_av_cached_with, softmax_rect_with |

Katalog notları:

- `_with` eki = sabit meta uploadlamak yerine, caller'ın sahip olduğu ve adımlar
  arasında `write_to` ile güncellenen meta buffer'ını alan varyant (ayrıntı:
  "Meta protokolü"). FwdPhase satırındakilerin çoğunun `_with` ikizi vardır;
  decode yolu hep `_with` kullanır.
- `matmul(_add)_with`, m=1'de otomatik GEMV(_ADD) builtin'ine yönlenir (H6).
- `rope_bwd` ve `head_scatter` yalnız `#[cfg(test)]` yaşar: fused ikizlerinin
  (rope_bwd_qk, qkv_split/qkv_scatter) doğrulama referanslarıdır.

### Gösterim amaçlı mini yollar

Tam pipeline değil, faz başına hangi kapılardan geçildiğinin özeti:

```
Train  : embedding → [FwdPhase zinciri + rope_qk + flash_attention] × 12 katman
         → cross_entropy   |   bwd: Train-only emitter'lar ters sırada, tek fused graph
Prefill: embedding → [FwdPhase zinciri + rope + flash_attention + cache_write] × 12 katman
         → yalnız son satırın logits'i
Decode : embedding_with → [_with zinciri + rope_offset + cache_write
         + attn_qk_cached / softmax_rect / attn_av_cached] × 12 katman
         → logits → host'ta sample
```

Yaşam döngüleri: Train graph'leri bir kez kurulur, metaları sabittir →
`execute_captured` kullanabilirler. Prefill graph'ı her prompt'ta sıfırdan
kurulur. Decode graph'ı ilk decode_step'te kurulur, cache değişene kadar saklanır;
adım başına yalnız 4 dinamik meta güncellenir (rope pos, cache offset, attn_len,
softmax width). cuBLAS metaları capture'da donduğu için decode capture edilemez.
Mikro-batch döngüsü (zero_transient → fwd → bwd, cycle sonunda clip + AdamW)
"Eğitim döngüsü" bölümünün konusudur.

## Grad topolojisi ve zero'lama sözleşmesi

### layers.rs: iki dünyanın kavşağı

`layers.rs` weight SAHİBİ değildir — her sarmalayıcı weights.rs'teki tensöre Arc
klonuyla tutunur — ama eğitime özgü her şeyin sahibidir: grad buffer'ları, ara
çıktı buffer'ları, fwd/bwd graph'leri. Bir `Linear` üç parçanın evliliğidir:
weight (weights.rs'ten ödünç) + grad (burada doğar) + emit çağrıları (ops'tan).
weights.rs kernel bilmez; layers.rs weight init bilmez.

**(weight, grad) eşleşmesi bir SIRA sözleşmesi taşır**: `collect_trainable_params`
çiftleri `ModelWeights::params()` sırasıyla üretir (embedding → blok başına
norm1/qkv/out/norm2/ffn_up/ffn_down → final_norm → lm_head). AdamW momentleri
bu sırayla yaratılır, checkpoint dosyası bu sırayla yazılır/okunur, V1 grad
restore bu sıraya zip'lenir. Sırayı değiştirmek = checkpoint ve moment karışması.

### Üç grad sınıfı

| Sınıf | Örnekler | Yazım | Kim sıfırlar | Neden |
|---|---|---|---|---|
| **Persistent** (weight grad'ları) | grad_weight, grad_table, norm grad'ları | `+=` (dB+=, dW+=, atomik embedding scatter) | `zero_grads` — accumulation cycle BAŞINDA | mikro-batch'ler arası birikmeleri tasarımın kendisi |
| **Transient** (residual fan-in) | blok başına add_1/add_2 grad_a/grad_b (`transient_grads()`), edges[] dahil | `+=` (birden çok kaynak aynı buffer'a ekler) | `zero_transient` — HER mikro-batch'te | `+=` hedefi taze başlamazsa önceki örneğin grad'ı sızar |
| **Overwrite** (ara grad'lar) | g_* buffer'ları: matmul_trp grad_input'ları, silu/rmsnorm/flash bwd çıktıları, CE bwd (in-place) | `=` (Output: tamamen ezilir) | kimse | her bwd geçişinde baştan yazılır; pool çöpü zararsız |

Fan-in noktaları (transient'lerin `+=` almasının sebebi): residual kavşağında
grad iki koldan gelir — ör. `g_add2_a` hem add_2 bwd'den hem barrier üzerinden
`norm_2.grad_input`'tan ekleme alır; blok grad_input'u (`edges[i]`) hem add_1
bwd'den hem norm_1 kolundan.

### edges[] zinciri

Trainer `num_layers + 1` adet dim-boyutlu buffer yaratır:
`edges[i]` = blok i'nin grad_input'u = kendinden önceki katmanın grad_output'u.
Embedding bwd `edges[0]`'ı okur. `edges[num_layers]` son bloğun grad_output'udur
ve final_norm bwd tarafından Output olarak ezilir — bu yüzden yalnız
`edges[0..num_layers]` transient (zero'lanan) listededir, sonuncusu değil.

Not: fwd ve bwd'de aynı matematiği yapan iki ayrı `+=` kernel'i vardır
(RESIDUAL_ADD / BWD_ADD_INPLACE) — bilinçli ayrım; bwd'nin kendi Shader statiği
fused backward graph içinde ayrı düğüm kimliği taşır.

### Dataline: weight → layer → shader hattı

Her weight tensörünün hangi sarmalayıcıdan geçip hangi kernellere bağlandığı.
Blok deseni 12 kez tekrarlanır:

| weights.rs (blok) | layers.rs | fwd kernel | bwd kernelleri | grad (sınıf) |
|---|---|---|---|---|
| norm_1 | RMSNorm | RMSNORM | RMSNORM_BWD (dX) + RMSNORM_WEIGHT_BWD (dW `+=`) | grad_weight (Persistent) |
| qkv_proj | Linear | MATMUL | MATMUL_WEIGHT_BWD (dW `+=`) + MATMUL_TRP (dX) | grad_weight (Persistent) |
| out_proj | Linear | MATMUL | aynı çift | grad_weight (Persistent) |
| norm_2 | RMSNorm | RMSNORM | RMSNORM_BWD + RMSNORM_WEIGHT_BWD | grad_weight (Persistent) |
| ffn_up | Linear | MATMUL | MATMUL_WEIGHT_BWD + MATMUL_TRP | grad_weight (Persistent) |
| ffn_down | Linear | MATMUL | MATMUL_WEIGHT_BWD + MATMUL_TRP | grad_weight (Persistent) |

Üst seviye:

| weights.rs | layers.rs | fwd kernel | bwd kernelleri | grad (sınıf) |
|---|---|---|---|---|
| embedding | Embedding | EMBEDDING | EMBEDDING_BWD (atomik `+=`) | grad_table (Persistent) |
| final_norm | RMSNorm | RMSNORM | RMSNORM_BWD + RMSNORM_WEIGHT_BWD | grad_weight (Persistent) |
| lm_head | Linear | MATMUL | MATMUL_WEIGHT_BWD + MATMUL_TRP | grad_weight (Persistent) |

Weight'siz katmanlar (weights.rs sütunu boş — sadece akışı şekillendirirler):

| layers.rs | fwd kernel(ler) | bwd kernel(ler) | grad buffer'ları (sınıf) |
|---|---|---|---|
| SelfAttention | FLASH_ATTENTION ×batch | FLASH_BWD_DQ + FLASH_BWD_DKDV ×batch | g_attn_q/k/v (Overwrite), l_cache (scratch) |
| SiLU | SILU_OUT | SILU_BWD | g_silu_in (Overwrite) |
| Add ×2 | ADD | RESIDUAL_ADD ×2 (`+=` fan-in) | grad_a/grad_b (Transient) |
| qkv taşıma | QKV_SPLIT | QKV_SCATTER + ROPE_BWD_QK | g_attn_qkv (Overwrite) |
| blok barrier'ları | — | BWD_ADD_INPLACE ×2 (`+=` fan-in) | hedefleri Transient |
| CrossEntropy | CROSS_ENTROPY (in-place) | CROSS_ENTROPY_BWD (in-place) | logits buffer'ının kendisi (bkz. VRAM aliasing) |

## Meta protokolü

Meta = kernel parametrelerini taşıyan küçük bir tensör; `TensorMode::Meta` ile
bağlanır. Tipli struct'lar `ops/meta.rs`'te yaşar; `KernelMeta` trait'i iki şey
verir: `upload(ctx)` (yeni sabit meta yarat) ve `write_to(tensor)` (mevcut
metayı yerinde güncelle).

İki kullanım türü:

|  | Sabit meta | Dinamik meta |
|---|---|---|
| API | `foo(gb, ..., shape)` | `foo_with(gb, ..., shape, &meta)` |
| Sahibi | kimse — emit içinde uploadlanır, buffer'ı node'un Arc'ı yaşatır | caller (DecodeScratch alanları) |
| Güncelleme | asla | adım başına `write_to` (`update_for_step`) |
| Kullanıcı | tüm train graph'leri, prefill | decode'un 4 metası: rope pos, cache offset, attn_len, softmax width |

Kim ne zaman okur (backend asimetrisi):

- **wgpu**: kernel buffer'ı execute anında okur → güncelleme her zaman görünür.
- **CUDA generic**: meta device pointer olarak gider → yine canlı.
- **CUDA cuBLAS** (matmul ailesi): boyutlar host'ta lazım → capture yokken her
  dispatch'te dtoh, capture'da build anında donmuş `cached_meta`. **Kural bu
  yüzden var: içinde matmul olan bir graph dinamik meta taşıyorsa
  `execute_captured` KULLANAMAZ** — decode'un capture edilememesinin tek sebebi.

shape/grid ikiliği (kolay unutulan kural): `shape` parametresi grid'i **build**
anında boyutlandırır, meta ise kernel'i **run** anında sınırlar. Dinamik durumda
grid maksimuma göre kurulur (decode attention grid'i max_context_len'e göre),
canlı işi meta bounds eder — fazla thread'ler guard'la erken döner.

## KV cache

Yapı (`inference_graphs.rs::Cache`):

- Katman başına K ve V ayrı birer tensör, düz `[max_context_len, dim]`
  (satır = mutlak pozisyon; head h, satır içinde `h*head_dim ..< (h+1)*head_dim`
  sütunları). Attention kernelleri cache'i stride'lı okur (H6) — per-head kopya yok.
- VRAM: 2 × num_layers × max_ctx × dim × 4B (hall-1.0 @ 512 ctx ≈ 38 MB).

Kontratlar:

- Cache **RoPE'lanmış key** tutar: rope, cache_write'tan ÖNCE uygulanır
  (prefill'de `rope`, decode'da `rope_offset`). V roped değildir. Okuyan kernel
  pozisyon bilgisine ihtiyaç duymaz.
- `cur_len`'in tek sahibi InferenceSession'dır: prefill başarılı execute
  SONRASINDA `cur_len = prompt_len` yazar, decode_step her adımda `pos + 1`.
  Kerneller cur_len'i hiç görmez — canlı uzunluk onlara attn_len dinamik
  metasıyla gider.
- `replace_cache` / `take_cache` decode graph'ını None'lar: graph node'ları
  ESKİ cache buffer'larına Arc tutar; cache değişince graph yeniden kurulmak
  zorundadır. Cache'i session'lar arası taşımak bu yüzden ucuzdur ama graph
  rebuild bedeli vardır.
- prefill boş cache ister (`CacheNotEmpty`), context dolunca decode `ContextFull`
  döner. `Cache::reset` yalnız cur_len'i sıfırlar, buffer içeriğine dokunmaz —
  bayat satırlar zararsızdır çünkü attn_len bound'u sayesinde asla okunmazlar.

## VRAM aliasing kararları

Başrol — tek `[rows, vocab]` buffer (V1; probs + grad_logits alloc'larını,
~2×200MB, sildi):

```
lm_head matmul çıktısı (logits)
  → CE fwd  AYNI buffer'ı yerinde probs'a çevirir
  → CE bwd  AYNI buffer'ı yerinde grad_logits'e çevirir
  → lm_head bwd bunu grad_output olarak okur
```

- `Linear::new` out_buffer'ı dışarıdan alır; lm_head'e `&logits` HEM out_buffer
  HEM grad_output olarak verilir — alias bilinçlidir.
- **Sonuç 1**: CE fwd'yi bwd koşmadan iki kez çalıştırmak buffer'ı bozar
  (probs'un softmax'ı alınır). diagnose'un logit okuyan check'leri bu yüzden
  CE'siz `forward()` kullanır. Bu bölüm okunmadan bu davranış "bug" sanılır.
- **Sonuç 2**: fused graph'lerde sıra yük taşır — fwd graph CE fwd ile bitmek,
  bwd graph CE bwd ile başlamak zorundadır.

Küçük paylaşımlar:

- `edges[]`: bloklar arası grad hattı — `edges[i]` hem block i'nin grad_input'u
  hem bir önceki katmanın grad_output'udur; embedding bwd `edges[0]`'ı okur.
- `rsqrt_cache`: rmsnorm_bwd yazar, rmsnorm_weight_bwd aynı graph içinde okur —
  iki node arasında scratch aliası (fwd'de hesaplanmaz, bwd yeniden türetir).

## Eğitim döngüsü ve checkpoint

Accumulation cycle anatomisi (`train_step`):

```
step % accum == 0        → zero_grads (persistent weight grad'ları)
her mikro-batch penceresi:
    token + target htod → zero_transient → fused fwd (captured)
    [step % READ_LOSS == 0 ise losses dtoh, ~2KB]
    → fused bwd (captured)
(step+1) % accum == 0    → clip (GRAD_SUMSQ → GRAD_NORM_SCALE → GRAD_SCALE)
                         → optimizer.step()
```

- Grad ölçeği: `d_losses` her çağrıda `1/(seq·batch·accum)` yazılır — effective
  batch normalizasyonu CE bwd'nin içinde olur, sonradan ayrıca bölme yoktur.
- **Sıra yük taşır**: AdamW graph'ında ADAMW_SCHEDULE node'u parametre
  node'larından ÖNCE gelir. step=0'da bias_correction = 1−β⁰ = 0 → sıfıra
  bölme → NaN; schedule önce koştuğu için AdamW hiçbir zaman step=0 görmez.
- Host trafiği (steady state): pencere başına token upload + READ_LOSS'ta bir
  losses dtoh. Grad'lar GPU'dan hiç inmez (clip H2'den beri device'ta).
  READ_LOSS bugün = LOG_EVERY = 50 (lib.rs alias'ı) — okuma ve yazdırma
  sıklığı ayrılabilir ama şu an çakışıktır.

Checkpoint:

- **V3 = TEK format** (B5) = `AKV3` magic + mimari başlığı (vocab/dim/heads/
  layers/ffn; yüklerken eşleşmezse hata) + `train_step` (loop sayacı) +
  `schedule_step` (AdamW cycle sayacı — ikisi FARKLI sayaçlardır, accum > 1'de
  ayrışırlar) + weight'ler `weights.params()` sırasında + AdamW (m, v)
  momentleri aynı sırada (**sıra format sözleşmesidir**). `moments` boş =
  weights-only dosya (migre v1/v2): yüklenince optimizer soğuk, schedule 0'dan.
- Legacy v1/v2 okuyucuları YALNIZ `bin/migrate_checkpoint_v3.rs`'te yaşar;
  kütüphane v3 dışında hiçbir şeyi okumaz/yazmaz. Migre dosya momentsiz +
  train_step 0 yazılır: eğitilmiş weight'ler üzerinde taze schedule —
  continued pretraining'in başlangıç durumu tam olarak bu.
- main.rs resume sırası: en yeni `model_step_*.bin` (step dosyanın İÇİNDEN
  okunur, dosya adı yalnız migre dosyalar için fallback) → yoksa
  `checkpoints/model_final.v3.bin` → yoksa sıfırdan. Final kayıt
  `model_final.v3.bin`'e gider; `model_final.bin` adı v1 anı dosyasıdır,
  asla yazılmaz.
- `checkpoint::save` tensör-tensör streaming yazar (B13 fix): byte düzeni
  bincode'un V3Body çıktısıyla birebir aynıdır (fixint LE, elle sıralanmış) —
  host tepesi tek tensör (~150MB); `load` bincode parse'ında kaldığı için
  roundtrip testi format sözleşmesini bekçiler.
- Format sözleşmesinin bekçisi: train.rs `v3_save_load_roundtrip` testi
  (weight + moment + iki sayaç, taze trainer'a bit-exact dönüş).

## Invariantlar

Assert'e (henüz) dökülememiş kurallar. Bugfix turunda her düzeltmenin
invariantı buraya bir satır olarak eklenir.

- **`weights.params()` sırası format sözleşmesidir** — checkpoint dosyası,
  AdamW momentleri ve V1 grad restore bu sıraya zip'lenir (bkz. Grad topolojisi).
- **Pozisyonlar batch elemanı başına 0'dan başlar** (row_offset tasarımı; RoPE
  açısı yerel `token_idx`'ten). `batching_validation` bunu kanıtlar.
- **`train_step`'in `batch_size` ARGÜMANI ≠ `cfg.batch_size`.** Argüman
  host-loop tekrar sayısıdır; gerçek batching için `cfg.batch_size = B` kur ve
  argümanı 1 ver (ayrıntı: BATCHING_PLAN.md). İkisini birden >1 vermek
  tanımsızdır — train_step bunu assert'le reddeder; Trainer::new de
  input_tokens'ın rows token tuttuğunu assert eder (B6).
- **RMSNorm eps'in tek kaynağı `cfg.norm_eps`'tir** (B14 fix) — train ve
  inference aynı config alanını okur; yine de 1e-5'ten oynatmak eğitilmiş
  checkpoint'in numeriğinden sapmaktır, model başına sabit tut.
- **`l_cache` düzeni `[row, head]`, row_offset'siz** — her flash çağrısının
  kendi küçük scratch'i; fwd ile bwd aynı buffer'ı paylaşır.
- **ADAMW_SCHEDULE, AdamW node'larından önce koşar** — AdamW step=0 görürse
  bias_correction sıfırlanır → NaN.
- **CE fwd'den sonra bwd koşmadan CE fwd tekrar koşulamaz** (V1 in-place;
  bkz. VRAM aliasing).
- **Cache roped K tutar; `cur_len`'i yalnız InferenceSession ilerletir;**
  cache değişimi = decode graph rebuild (node'lar cache buffer'larına Arc'lı).
- **Dinamik metalı + matmul'lu graph capture edilemez** (Meta protokolü).
- **EOS = 50256 hardcode** (`generate`) — GPT-2 vocab'ına bağlıdır; tokenizer
  değişirse burası da değişmeli.
- **Tokenizer offline-first**: yerel `tokenizer.json` / `AKASHA_TOKENIZER`
  varsa ağa çıkılmaz; ilk indirme yerel kopya bırakır.
- **`checkpoints/model_final.bin` (v1) dokunulmazdır** — tek eğitilmiş model;
  migrasyonlar kopya üzerinde yapılır (`model_final.bin.v2.bin`).
- **Weight decay yalnız matmul weight'lerine uygulanır** — norm gain'leri ve
  embedding tablosu decay'siz (E1; bayrak `collect_trainable_params`'ta,
  AdamW node'u ona göre decay'li/decay'siz ConstCfg'ye bağlanır).
- **embedding_bwd atomik CAS kullanır** — token tekrarı yarışının çözümü ve
  2026-06'daki "WGSL'de atomics yok" kararının kayıtlı istisnası (eski wilupgu
  README'sindeki aksi beyan kaldırıldı).

## Test haritası

| Test | Neyi koruyor |
|---|---|
| train.rs `full_chain_gradcheck` | GERÇEK fused fwd/bwd zincirinde analitik grad == sayısal grad (norm/ffn/qkv örneklem indeksleri) |
| train.rs `fused_ops_integration` | loss sonlu + grad embedding'e kadar akıyor (smoke) |
| train.rs `batching_validation` | batch=N tek geçiş == N ardışık accumulation — row_offset tasarımının kanıtı |
| train.rs `grad_clip_validation` | GPU clip zinciri == host formülü (kırpan ve kırpmayan iki rejim) |
| emit.rs `flash_attention_validation` | flash fwd+bwd == bağımsız düz-Rust CPU referansı (sınır-dışı boyutlar dahil) |
| emit.rs `kernel_fusion_validation` | rope_qk == 2×rope; qkv_split/scatter == head_gather/scatter zinciri (fused == unfused) |
| emit.rs `decode_kernel_validation` | 3-dispatch cached attention == CPU ref (grid max'a göre, meta canlıyı sınırlar); m=1 GEMV yönlendirmesi |
| emit.rs `elementwise_grid_validation` | elementwise kerneller 65535-workgroup (16.7M eleman) 1D sınırının ÜZERİNDE doğru — 2D-linearize grid'in bekçisi (B16) |
| optim/adamw.rs | device schedule == host `cosine_lr`; weight'ler doğru yöne hareket ediyor |
| sampling.rs | greedy / top-k / top-p özellikleri |
| wilupgu `tests/` | backend parity (wgpu↔cuda↔CPU-ref), graph zinciri grid'i (T8), CUDA meta semantiği (**cfg(cuda) — yalnız nvidia makinede derlenir/koşar**), mode-mismatch paniği |
| `bin/diagnose.rs` (test değil, elle koşulan binary) | 10 check: param sayısı, grad akışı, gradcheck'ler, accumulation, CE kapalı-form, KV-cache == naif decode (CHECK 9, bit-exact), tok/s kıyası — "eğitim bozuk görünüyor" günlerinin ilk durağı. Durumu: ~900 satır, elden geçmesi Fikir kuyruğunda |

Kurallar:

- Her zaman `cargo test -- --test-threads=1` — paralel testler eşzamanlı
  WgpuBackend yüzünden segfault eder.
- Yeni op eklerken: emit.rs'teki ilgili validation modülüne CPU-referanslı
  satır ekle; fused kernel yazıyorsan unfused'ı `#[cfg(test)]` referans olarak
  yaşat (rope_bwd / head_scatter deseni).

Bilinen boşluklar: birkaç bwd kernel'inin CPU impl'i yok → evrensel parity
matrisi henüz koşamıyor; test yardımcıları (rand_vec, max_abs_diff) dosyalar
arası kopya — ikisi de "Fikir kuyruğu → test mimarisi"nin konusu.

---

## Fikir kuyruğu (mimari)

REFACTOR.md planlanmış/numaralı işleri tutar; burası mimariye dair fikir ve
istekler. Olgunlaşan madde oradan bir K/H/V/T numarası alıp taşınır.

- **Shader kataloğu tablosu**: shader başına tek satır — wgsl konumu (builtin ise
  sabiti) / varsa eski wgsl karşılaştırma versiyonu / cuda konumu (builtin ise
  sabiti) / varsa eski cuda referansı / cpu impl (yoksa boşluk kendini gösterir) /
  meta struct'ı (ops/meta.rs) / emitter'ı (emit.rs). ~35 satır. Amaç: bir kernel'in
  tüm parçalarını ve hangi fazlarda yaşadığını tek bakışta görmek; CPU-impl
  boşluklarını görünür kılmak.

- **CUDA shader'larını .cu dosyalarına ayırmak**: bugün hepsi `shaders/cuda.rs`
  içinde tek string yığını; wgsl'lerdeki dosya-başına-kernel düzenine geçir
  (`include_str!("cuda/foo.cu")` — NVRTC zaten string alır, davranış değişmez;
  syntax highlighting + diff okunabilirliği bedavaya gelir).

- **Decode'u cuBLAS'sızlaştırmak (GEMV'yi CUDA Generic'e taşımak)** — muhtemelen
  en yüksek getirili tekil iş: `gemv.wgsl` / `gemv_add.wgsl`'in CUDA C çevirisi
  yazılıp GEMV/GEMV_ADD `CudaShape::Custom`(cuBLAS)'tan `Generic`'e geçer.
  Decode'daki TÜM matmul'lar m=1 → GEMV olduğundan decode graph'ında hiç cuBLAS
  kalmaz. Sonuçlar: (1) dispatch-başı meta dtoh'u (B9) kökünden yok olur;
  (2) capture yasağının tek sebebi cuBLAS host-side boyutlarıydı → **decode
  graph'ı capture edilebilir olur** (token başına 100+ dispatch yerine tek
  graph launch; dinamik metalar device-pointer'dan replay'de güncel okunur);
  (3) `Custom` shape yalnız eğitim matmul'larının istisnasına küçülür.
  GEMV bellek-bant sınırlı olduğu için cuBLAS'a hız kaybı yok denecek kadar az;
  eğitim (m>1) ve prefill cuBLAS'ta kalır — orada tensor core (TF32/bf16) farkı
  yapısal: skaler CUDA C tensor core'a hiç dokunamaz, erişim cuBLAS / CUTLASS /
  elle mma-PTX ister. Eldeki 80-vs-31 step/dk ölçümü pooling/flash ÖNCESİ
  dönemden — **ilk adım: güncel kodda CUDA↔Vulkan'ı yeniden ölçmek**, kararlar
  o sayıyla verilecek. Not: f16/bf16 altyapısı (`Dtype`, `Gemm<T>`) zaten
  cuBLAS'ın tensor-core yoluna akar — mixed-precision eğitimin zemini orası.

- **Dinamik metayı tipe bağlamak** (Binding düzeyinde `DynamicMeta` işareti):
  Shader.layout'a KONAMAZ — aynı shader train'de sabit, decode'da dinamik meta
  alır; işaret binding'e aittir (layout `Meta` der, caller `Meta`/`DynamicMeta`
  ile bağlar). Kazançlar: (1) CUDA build_node dinamik meta için bayat cached_meta
  üretmez; (2) execute_captured, DynamicMeta içeren graph'ta assert'le durur —
  "decode capture edilemez" kuralı yorumdan koda iner. **Etkileşim notu:**
  yukarıdaki decode-cuBLAS'sızlaştırma yapılırsa bu maddenin perf gerekçesi
  (cuBLAS cached_meta hızlı yolu) düşer; kalan değeri dokümantasyon + guard —
  önceliği ona göre düşür.

- **Test mimarisi ("testleri adam etmek")**: bugün testler src dosyalarının
  dibinde `#[cfg(test)]` modülleri (train.rs'in ~yarısı test) ve `rand_vec` /
  `max_abs_diff` gibi yardımcılar 4+ dosyada kopya-yapıştır. Plan:
  (1) ortak test-util modülü — kopya yardımcılar tek yere;
  (2) inline test modüllerini tests/ ağacına veya ayrı mod dosyalarına taşı;
  (3) her kernel için mekanik CPU-referans parity harness'ı (CPU impl boşlukları
  kapanınca tam matris koşar);
  (4) Output kontrat canary'si: Output etiketli buffer'ları dispatch öncesi
  NaN/çöple doldurup çıktıyı doğrulayan tek jenerik test — T1 sınıfı etiket
  buglarını otomatik yakalar.
  (5) diagnose.rs elden geçirilmesi: ~900 satır, env-flag'li, elle koşulan
  check yığını — hâlâ değerli olan check'ler (gradcheck'ler, KV==naif,
  kapalı-form CE) gerçek test modüllerine taşınır, refactor'lardan sağ çıkmış
  ölü check'ler silinir, kalan şey küçük bir "sağlık taraması" binary'si olur.
  İlke: test edilen yer değil, test edilmeyen yer patlar.

- **GPU'nun eğitimde CPU'dan bağımsızlığının doğrulanması**: steady-state train
  loop'un kasıtlı trafiği dışında (pencere başına token htod + READ_LOSS'ta bir
  ~2KB losses dtoh) hiçbir gizli host senkronu/kopyası olmadığını kanıtla —
  ör. copy_to_cpu/synchronize çağrılarını sayan bir debug sayaç veya
  wgpu/nsight trace. Aşağıdaki loss recorder ile birlikte döngü trafiğini tam
  sıfıra indirir.

- **GPU loss recorder ("loss_counter")**: CE loss'u zaten GPU'da hesaplıyor;
  kayıt aralığında bir kez küçük bir kernel losses'ı indirgesin (Σ/seq_len) ve
  bir kez alloc edilmiş sabit boyutlu history buffer'ına yazsın
  (500k step / 50 = 10k float ≈ 40KB — eğri sonda TEK dtoh ile iner).
  Binding taslağı: losses (Input) · history (tek slota kısmi yazım — sözleşme
  tartışılır: kısmi yazım aslında InOut'tur, Accumulate "+=" demektir; her
  durumda caller-zeroed init şart) · interval + kapasite (sabit Meta) ·
  adım sayacı — **Meta OLAMAZ: Meta read-only bağlanır**; kendini sayan değer
  InOut state buffer'ı ister (ADAMW_SCHEDULE'ın ScheduleState deseni). Hatta
  ayrı sayaca hiç gerek olmayabilir: optimizer'ın `state.step`'i zaten
  device'ta sayıyor — Input olarak okunup `index = step / interval`
  türetilebilir.

- **Kernel fusion adayları** (eski FUSION_TODO.md buraya taşındı; unfused
  kerneller her durumda doğrulama referansı olarak kalır):
  (1) *AdamW foreach* — bugün weight tensörü başına bir dispatch (~75);
  tek/az dispatch'e indir — wilupgu'nun işi (`builtin::ADAMW`).
  (2) *Linear+SiLU epilogue* — ffn_up → silu arasındaki pre-activation'ı
  kimse okumuyor; blocker: matmul cuBLAS builtin'i, epilogue hook'u yok —
  bu yol için elle fused kernel gerekir (flash attention tradeoff'unun aynısı).
  (3) *Add+RMSNorm epilogue* — aynı blocker (`residual_add` builtin).
  (4) *rmsnorm_bwd + rmsnorm_weight_bwd* — park edildi: iki kernel farklı
  eksende paralelleşir (satır-başına vs özellik-başına reduction), fusion
  dWeight için atomik ister; mevcut model boyutunda değer/karmaşıklık oranı
  düşük.

---

## Big Refactor: Block trait ve heterojen stack

**Statü: TASARIM — ertelendi (2026-07-17).** Ön koşul: F1 (gerçek batch) +
bf16 entegrasyonu bitip optimize sürüm eğitim için arkadaşın makinesine
paslanacak; refactor o ~10 günlük eğitim koşarken tasarlanır/yapılır.

### Motivasyon

1. **layers.rs ile inference_graphs.rs aynı matematiği iki kez tarif ediyor.**
   Gerçek fark üç kalem: (a) bwd + grad buffer'ları, (b) attention'ın iki hali
   (full-seq flash vs cached 3-dispatch), (c) train'in bwd için aktivasyon
   saklaması. (a) ve (b) faz tip sistemi tarafından zaten kodlanmış — eksik
   olan tek şey üstteki blok katmanı.
2. **Mamba/Jamba vizyonu**: config.rs'ten katman listesi yazarak
   (`[Attn, Attn, Mamba, ...]`) heterojen model kurabilmek — hardcode'suz.
   Attention alternatifi her mimari aynı arayüze oturur; Jamba bir config
   satırı olur ve yeni model serisi oradan başlar.

### Trait yüzeyi (taslak)

```rust
trait Block {
    /// BİR kez yazılır; prefill, decode ve train-fwd aynı fonksiyondan çıkar.
    fn fwd<P: FwdPhase>(&self, gb: &mut GraphBuilder<P>, x: ..., tape: &mut P::Tape) -> Tensor;
    /// Yalnız Train; aktivasyonları tape'ten okur.
    fn bwd(&self, gb: &mut GraphBuilder<Train>, dx: ..., tape: &TrainTape) -> Tensor;
    // + weight/grad kaydı (GradClass ile)
    // + cached-phase state handle'ı (KV cache / SSM state) + dinamik meta kaydı
}
```

- **Tape** — faza bağlı associated type (`P::Tape`). `TrainTape` bwd'nin
  ihtiyaçlarını saklar (residual'lar, rms, pre-activation, l_cache);
  inference fazlarının Tape'i boş ZST — sıfır maliyet. "Etiket tipin
  kendisidir" ilkesinin bir kat üstü.
- **State handle** — cached fazların adım-durumu blok türüne aittir:
  attention'da KV cache, Mamba'da SSM state (sabit boyut, büyümez).
  `update_for_step` elle yazılmış write_to listesi yerine blokların
  kaydettiği dinamik metaları dolaşır. *(Eski kuyruk maddesi
  "GraphBuilder\<Decode\> meta kaydı" buraya emildi.)*
- **Weight/grad kaydı** — blok, weight'lerini (decay bayrağıyla) ve
  grad'larını sınıfıyla (Persistent/Transient/Overwrite) kaydeder.
  `params()` sırası "stack sırası" olur; checkpoint düzeni ve
  zero_grads/zero_transient graph'leri registry'den TÜRETİLİR. Bonus assert:
  "Accumulate ile bağlanan her buffer bir zero listesinde kayıtlı olmalı".
  *(Eski kuyruk maddesi "GradClass registry" buraya emildi.)*
- lm_head + CE + V1 aliasing stack DIŞINDA kalır — top-level ve train'e özgü.

### Checkpoint etkisi

V3 header'ı homojen (dim/heads/layers/ffn). Heterojen stack katman-spec
listesi ister → V4 (ya da geriye uyumlu V3 uzantısı). **Refactor'un 1.
gününde tasarlanır** ki üçüncü format migrasyonu yaşanmasın.

### Mamba planı

- Hedef **Mamba-2 (SSD formülasyonu)**: Mamba-1'in sequential selective
  scan'i yerine chunk'lı matmul'lara ayrışır — mevcut matmul/cuBLAS
  altyapısına oturur, bwd'si de aynı yapıdadır.
- **Inference-first**: fwd/decode kernelleri önce yazılır ve hazır bir
  checkpoint import'uyla (ör. state-spaces/mamba-130m) bilinen-doğru çıktıya
  karşı doğrulanır; bwd (projenin bugüne kadarki en zor kernel'i) ondan sonra.
- Mamba decode'u sabit bellek + sabit hesap/token — KV cache büyümez;
  iGPU/düşük-donanım kimliğiyle birebir örtüşür.

### Aday blok listesi (token-mixer koltuğu)

Modern çerçeve: her blok = **token-mixer** (attention'ın koltuğu) +
**channel-mixer** (FFN'in koltuğu). Adaylar token-mixer koltuğu için;
hepsi aynı `Block` arayüzüne oturur.

| Aday | Not |
|---|---|
| Attention (mevcut) | GQA / sliding-window / MLA ayrı blok DEĞİL, bu bloğun config varyantları |
| **Linear attention (GLA / Gated DeltaNet)** | Muhtemelen en kolay ikinci blok — chunk'lı matmul'lara ayrışır (Mamba-2 kernel'leriyle akraba ama conv1d'siz), decode sabit-state. Qwen3-Next hibriti Gated DeltaNet kullanır. **Mamba-2'den ÖNCE gelmesi mantıklı: soyutlamayı en ucuz attention-dışı blokla doğrula.** Dikkat: "tek kernel + emit" değil — train fwd (chunk) + bwd + decode state-update üçlüsü ister; yine de projenin yazdığı flash bwd'den zor değil |
| Mamba-2 (SSD) | Yukarıdaki plan |
| RWKV(v7) / Griffin | Uzak takip listesi — arayüz bunları da taşır, aceleye gerek yok |

**GQA notu (Big Refactor'dan BAĞIMSIZ, ucuz):** KV head sayısını düşürmek
KV cache'i ve decode bant trafiğini `heads/kv_heads` kat küçültür — iGPU
derdine birebir. Maliyet: config + kernellerde `head → kv_head = head/group`
eşlemesi (flash fwd/bwd, cached attention, qkv split). **KARAR GEREKTİRİR:**
GQA mimari değişikliğidir → sıfırdan init; hall 1.0 weight'lerinden continued
pretraining ile BAĞDAŞMAZ. Sonraki koşu sıfırdansa gir, continued ise girme.

### Block'un gelecek ihtiyaç listesi (tasarım günü kontrol listesi)

Refactor günü trait yüzeyi çizilirken bunlar masada olmalı — hepsini
İMPLEMENTE etme, ama hiçbirini imkânsız kılma:

- **Token-mixer / channel-mixer ayrımı** — `Block` iki koltuğu ayrı görmeli;
  MoE = FFN koltuğunun alternatifi (Mixtral deseni).
- **MoE'nin isteyecekleri**: device'ta router top-k, expert weight'lerinin
  `params()` sırasına girişi, ve **aux loss hook'u** (load-balancing loss
  ana loss'a eklenir) — trait'te "bloğun loss'a katkısı" kapısı baştan
  düşünülmeli, sonradan eklemesi acı verir.
- **Pozisyon bilgisinin sahipliği**: RoPE attention BLOĞUNA aittir, global
  pipeline'a değil — Mamba/GLA pozisyonsuz. `rope pos` dinamik metası da
  bloğun kendi state/meta kaydına iner.
- **Init'in stack bağlamı**: E2'nin 1/√(2L)'si L = "residual'a yazan blok
  sayısı" — heterojen stack'te blok init'i stack uzunluğunu parametre almalı.
- **Per-blok dtype**: bf16 entegrasyonu Block'tan önce geleceği için arayüz
  weight dtype'ını blok başına taşıyabilmeli (mixed-precision zemini).
- **BlockSpec hiperparamları**: head/kv_head sayısı, ffn boyutu, SSM state
  boyutu — hepsi per-blok, config'te `Vec<BlockSpec>` içinde.
- **Tape türü blok impl'ine aittir** — her blok türünün bwd ihtiyacı farklı
  (attention: l_cache; GLA: chunk state'leri); trait tek somut TrainTape
  dayatmasın.
- **State handle "cache" değil "adım durumu"** — büyüyen (KV) ve sabit
  (SSM/linear-attn) state'leri aynı arayüz taşımalı; `cur_len` KV'ye özgü
  bir detay olarak blok içine iner.

### Adım sırası

1. **Block çıkarma refactor'u, YALNIZ transformer'la** — davranış bit-exact
   aynı, bekçiler hazır (KV==naif CHECK 9, full_chain_gradcheck,
   v3_save_load_roundtrip). layers.rs/inference_graphs.rs ikiliği burada
   ölür. Mamba hiç gelmese bile tek başına kâr.
2. Config-driven stack (`Vec<BlockSpec>`) + checkpoint header uzantısı.
3. İlk attention-dışı blok — aday sırası: **Gated DeltaNet → Mamba-2**
   (GDN soyutlamayı en ucuz doğrular; Mamba-2'de import doğrulaması
   [state-spaces/mamba-130m] kullanılır: önce fwd/decode, sonra bwd).
4. Jamba/hibrit = config satırı. Yeni model serisi buradan başlar.
