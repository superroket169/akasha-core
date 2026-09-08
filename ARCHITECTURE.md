# sequexa-core — Mimari

## Ne bu?

Genel tanıtım ve hızlı başlangıç [README.md](README.md)'de. Bu dosya iç
mimariyi anlatır: katman haritası, faz tip sistemi, Tape/Op sistemi, grad
topolojisi, meta protokolü, KV cache, invariantlar. Bekleyen işler ve
fikirler için [TODO.md](TODO.md) / [IDEAS.md](IDEAS.md) /
[wilupgu/TODO.md](../wilupgu/TODO.md).

## Katman haritası

```
config.rs ── ModelConfig + BlockKind (mimari seçimi: bugün tek variant,
   │          Transformer — AnyOptimizer/AnyGradClip'le aynı closed-enum kalıbı)
   ▼
weights.rs ── ModelWeights / BlockWeights          ← TEK GERÇEK:
   │            (salt weight tensörleri, grad yok)   train de chat da bunu paylaşır
   │            Clone = ucuz (Arc paylaşımı, tensör kopyalamaz)
   ▼
arch.rs ── Architecture<B> trait: train_specs/prefill_specs/decode_specs
blocks/transformer.rs ── impl Architecture<B> for Transformer (tek implementor)
   │        arch.rs'teki build_block/build_prefill_forward/build_decode_forward
   │        BlockKind'a göre match'leyip ilgili Architecture impl'ini çağırır
   ▼
model.rs ── Model<B>: weights + Option<TrainState> + Option<ChatState>
   │          for_training / for_chat kurar (Tape'leri arch.rs::build_* ile inşa
   │          eder), sonra: train_step / zero_grad / optimizer_step / eval_loss /
   │          save_checkpoint / load_checkpoint / params() / generate() /
   │          prefill_logits / decode_step_logits / reset_cache / max_context_len
   ▼
chat_session.rs ── ChatSession<B>: Model'i saran streaming ön-yüz
   │                (feed_prompt / next / reset / is_finished) — generate()'in
   │                blocking tek-çağrı halinden farklı olarak token-token akış,
   │                iptal, manuel cache kontrolü ister.
   ▼
tape.rs ── Tape<B, Node>, NodeSpec, Forward<B,P>/Backward<B>/Advance trait'leri,
   │        node! makrosu (compakt NodeSpec literal)
ops/{full_seq,cached}.rs ── somut op struct'ları (LinearOp, RmsNormOp, SiluOp,
   │        AddOp, EmbeddingOp, AttentionOp, RopeQkOp, QkvSplitOp — full_seq.rs;
   │        CacheWriteOp, CachedAttentionOp, RopeOffsetOp, HeadGatherOp — cached.rs)
dispatch/{forward,backward,advance,from}.rs ── TrainOp/PrefillOp/DecodeOp enum'larının
   │        Forward/Backward/Advance/From dispatch'i (makro üretimli, hepsi <50 satır)
   ▼
kernels/ ── meta.rs: typed meta struct'lar (KernelMeta)
             emit.rs: kernel başına TEK emitter fonksiyonu
             GraphBuilder<Phase>: Train / Prefill / Decode tip kapıları
   │
   ▼
shaders/ ── wgsl/ + cuda/ (ikisi de aynı fwd/bwd + üç top-level dosya düzeninde)
   │
   ▼
wilupgu ── builtin'ler (matmul ailesi, GEMV, ADAMW, ZERO_TENSOR,
           RESIDUAL_ADD...) + ComputeGraph + Backend'ler
```

Ayrıca, `Model`'e paralel duran iki şey:

- `diagnostic.rs` — `DiagnosticCheck` trait (`name`/`run`/`log`) +
  `DiagnosticSuite` (elle koşulan check'leri sırayla çalıştırıp özetleyen
  ince bir runner, `AnyOptimizer` tarzı closed-enum yerine trait+struct
  kullanır çünkü check'ler birbirinden bağımsız kaynak kurup/serbest
  bırakıyor).
- `bin/diagnose_kernels.rs` (CHECK 3/4/7 — ham kernel-vs-CPU-referans,
  `layers.rs`'in standalone `RMSNorm`/`CrossEntropy` sarmalayıcılarını
  kullanır) ve `bin/diagnose_model.rs` (CHECK 1/2/5/8/9/10 — `Model`
  üzerinden gerçek-ölçek sağlık taraması).

**`layers.rs`'in bugünkü rolü**: artık Trainer'a ait değil (Trainer yok) —
sadece `diagnose_kernels.rs`'in izole kernel testleri için duran, kendi
grad buffer'ını taşıyan standalone sarmalayıcılar (`Linear`, `RMSNorm`,
`CrossEntropy`, ...). `Model`/`Tape` bunlara hiç dokunmaz, `ModelWeights`'i
doğrudan okur.

**Tek dünya kuralı (eskisinin yerine)**: train/prefill/decode artık ayrı
dosya çiftleri değil — TEK Tape/Op sistemi + faz tip kapıları. Üç faz
arasındaki gerçek fark yalnız üç kalem: (a) attention algoritması (flash
vs 3-dispatch cached), (b) KV cache okuma/yazma (`CachedPhase`), (c)
backward'ın var olup olmaması (yalnız Train). Bunların hepsi `TrainOp`/
`PrefillOp`/`DecodeOp` enum'larında ve faz marker trait'lerinde zaten kodlu.

## nn/kernels: GraphBuilder ve faz tip sistemi

Bu bölüm yalnızca `nn/kernels/`'ı anlatır (GraphBuilder, Phase trait'leri,
emitter'lar). Graph'lerin *kurulduğu* yer artık tek bir yer: `arch.rs`'teki
`build_block`/`build_prefill_forward`/`build_decode_forward`, gerçek
düğüm-şeması ise `blocks/transformer.rs`'teki `Architecture` impl'inde
(Mamba/GDN geldiğinde kendi `blocks/*.rs` dosyalarında).

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
Train  : embedding → [FwdPhase zinciri + rope_qk + flash_attention] × N blok
         → cross_entropy   |   bwd: Train-only emitter'lar ters sırada, tek fused graph
Prefill: embedding → [FwdPhase zinciri + rope + flash_attention + cache_write] × N blok
         → yalnız son satırın logits'i
Decode : embedding_with → [_with zinciri + rope_offset + cache_write
         + attn_qk_cached / softmax_rect / attn_av_cached] × N blok
         → logits → host'ta sample
```

Yaşam döngüleri: Train graph'leri (`fwd_graph`+`bwd_graph`) `Model::for_training`
içinde bir kez kurulur, metaları sabittir → `execute_captured` kullanabilirler.
Prefill graph'ı her `prefill_logits` çağrısında sıfırdan kurulur. Decode graph'ı
`Model::for_chat`'te kurulur, `ChatState` yaşadığı sürece saklanır; adım başına
yalnız `Tape::advance(pos)` ile birkaç dinamik meta güncellenir (rope pos, cache
offset, attn_len, softmax width — `Advance` trait'i, `dispatch/advance.rs`). cuBLAS
metaları capture'da donduğu için decode capture edilemez.

## Tape/Op sistemi

`Tape<B, Node>` tek struct, üç bağlamda kullanılıyor: `Tape<B, TrainOp<B>>`
(train, `head`/her blok/`tail` için ayrı ayrı), `Tape<B, PrefillOp<B>>`,
`Tape<B, DecodeOp<B>>`. `Node: Backward<B>` yalnız `TrainOp` için sağlanıyor
(`.backward()` metodu `impl<B, Node: Backward<B>> Tape<B, Node>` şartlı bloğunda
yaşıyor — Prefill/Decode Tape'lerinde derleme zamanında yoklar).

```rust
pub(crate) struct NodeSpec<Node> {
    pub(crate) name: &'static str,
    pub(crate) inputs: &'static [(&'static str, usize)],
    pub(crate) op: Node,
}
```

Bir blok, `Vec<NodeSpec<TrainOp<B>>>` üretip `Tape::extend` ile tek seferde
inşa edilir — girdi referansları isim üzerinden (`"input"`, `"n1"`, `"qkv"`...)
çözülür, `HashMap<&'static str, NodeId>`'de tutulur. `node!` makrosu
(`tape.rs`, `pub(crate) use node;` ile path-import edilebilir) literal'ı
kısaltır:

```rust
node!("n1" <- &[("input", 0)], TrainOp::RmsNorm(RmsNormOp::new(&bw.norm_1, norm_shape)))
```

**Backward, graph-inşa-zamanında ters sırada gezip build eder, runtime'da
tekrar çağrılmaz.** `Tape::backward()` bir kez çalışır (`Model::for_training`
içinde), `bwd_graph`'ı doldurur; sonrasında yalnız `bwd_graph.execute_captured()`
tekrarlanır. Bu, eski sistemdeki "her mikro-batch'te elle `zero_transient`
çağır" ihtiyacını **ortadan kaldırdı**: fan-in noktalarında (bir node'un
grad'ı birden çok tüketiciden geliyorsa) `Tape::backward()` inşa sırasında
otomatik bir `add_inplace_bwd` node'u ekliyor — bu node'un iki girdisi de
AYNI execute içinde daha önceki node'larca taze üretiliyor, önceki
execute'tan kalan hiçbir şey yok. Yalnız **weight grad'ları** (Persistent
sınıf, aşağıda) execute'lar arası gerçekten birikir — `zero_grad()` bunun
için var, cycle başında bir kez çağrılır.

`LinearOp`/`RmsNormOp`/`SiluOp`/`AddOp`/`EmbeddingOp` her üç fazda da BİREBİR
aynı kod (`Forward<B, P: FwdPhase>` genel yazılmış); attention/rope/qkv-çıkarma
faza göre ayrılır (Train+Prefill flash/fused kullanır — ikisi de
`FullSeqPhase`; Decode cached/unfused kullanır, ayrı struct'lar —
`CachedAttentionOp`/`RopeOffsetOp`/`HeadGatherOp`/`CacheWriteOp`,
`ops/cached.rs`).

`TrainOp`/`PrefillOp`/`DecodeOp` **kapalı enum'lar**, `AnyOptimizer`/
`AnyGradClip` ile aynı kalıp — dispatch `match`, `dyn` yok. Fark: bu üçünün
Forward/Backward/Advance/From impl'leri elle değil `dispatch/`'teki
makrolarla üretiliyor (`impl_forward_dispatch!`/`impl_backward_dispatch!`/
`impl_from_op!`) çünkü match kolları mekanik tekrar; `Advance` elle yazılı
kalıyor çünkü gerçekten tekrar etmiyor (yalnız `CacheWriteOp`/`RopeOffsetOp`/
`CachedAttentionOp`/`DecodeOp` ilgileniyor, `TrainOp`/`PrefillOp`'ta hiç yok).

**Mimari genişleme noktası burası, `Architecture` trait'i değil**: Mamba/GDN
geldiğinde `TrainOp`/`PrefillOp`/`DecodeOp`'a yeni variant'lar eklenir (ör.
`TrainOp::MambaScan(...)`), `blocks/mamba.rs` bunları üreten yeni bir
`Architecture` impl'i olur, `config::BlockKind`'a yeni variant girer, üç
`build_*` fonksiyonundaki `match kind` birer kol alır. `Model<B>` generic
parametre almaz — `Vec<BuiltBlock<B>>` hâlâ homojen `TrainOp<B>` taşır,
heterojenlik yalnız HANGİ variant'ların o `Vec`'te göründüğünde yaşar.

## Grad topolojisi ve zero'lama sözleşmesi

### Üç grad sınıfı

| Sınıf | Örnekler | Yazım | Kim sıfırlar | Neden |
|---|---|---|---|---|
| **Persistent** (weight grad'ları) | `LinearOp`/`RmsNormOp`/`EmbeddingOp`'un `grad_weight`/`grad_table`'ı | `+=` (dB+=, dW+=, atomik embedding scatter) | `Model::zero_grad()` — accumulation cycle BAŞINDA | mikro-batch'ler arası birikmeleri tasarımın kendisi |
| **Fan-in** (çok-tüketicili node'lar) | residual kavşağındaki `add1`/`add2` girdi grad'ları, blok girişleri | `Tape::backward()` inşa sırasında otomatik `add_inplace_bwd` node'u | kimse (her execute'ta o node'un İKİ girdisi de aynı execute içinde taze) | fan-in birleşimi build-time'da sabitlenmiş bir graph node'u, runtime state değil |
| **Overwrite** (ara grad'lar) | tek-tüketicili node'ların `grad_in` buffer'ları (matmul_trp çıktıları, silu/rmsnorm/flash bwd) | `=` (Output: tamamen ezilir) | kimse | her execute'ta baştan yazılır; pool çöpü zararsız |

**`Model::params()` sırası bir SIRA sözleşmesi taşır**: `head.params() →
her blok'un `tape.params()` → `tail.params()`, ve blok içi sıra `node!`
push sırasıyla (norm_1, qkv, out, norm_2, up, down) — bu da
`ModelWeights::params()`'in sırasıyla birebir aynı (embedding → blok →
final_norm → lm_head). AdamW momentleri, checkpoint dosyası bu sırayla
yazılır/okunur. Sırayı değiştirmek = checkpoint ve moment karışması.

### Dataline: weight → op → shader hattı

Her weight tensörünün hangi op struct'tan geçip hangi kernellere bağlandığı.
Blok deseni her katmanda tekrarlanır:

| weights.rs (blok) | ops/full_seq.rs | fwd kernel | bwd kernelleri | grad (sınıf) |
|---|---|---|---|---|
| norm_1 | RmsNormOp | RMSNORM | RMSNORM_BWD (dX) + RMSNORM_WEIGHT_BWD (dW `+=`) | grad_weight (Persistent) |
| qkv_proj | LinearOp | MATMUL | MATMUL_WEIGHT_BWD (dW `+=`) + MATMUL_TRP (dX) | grad_weight (Persistent) |
| out_proj | LinearOp | MATMUL | aynı çift | grad_weight (Persistent) |
| norm_2 | RmsNormOp | RMSNORM | RMSNORM_BWD + RMSNORM_WEIGHT_BWD | grad_weight (Persistent) |
| ffn_up | LinearOp | MATMUL | MATMUL_WEIGHT_BWD + MATMUL_TRP | grad_weight (Persistent) |
| ffn_down | LinearOp | MATMUL | MATMUL_WEIGHT_BWD + MATMUL_TRP | grad_weight (Persistent) |

Üst seviye:

| weights.rs | ops/full_seq.rs | fwd kernel | bwd kernelleri | grad (sınıf) |
|---|---|---|---|---|
| embedding | EmbeddingOp | EMBEDDING | EMBEDDING_BWD (atomik `+=`) | grad_table (Persistent) — **decay=false** |
| final_norm | RmsNormOp | RMSNORM | RMSNORM_BWD + RMSNORM_WEIGHT_BWD | grad_weight (Persistent) — **decay=false** |
| lm_head | LinearOp | MATMUL | MATMUL_WEIGHT_BWD + MATMUL_TRP | grad_weight (Persistent) — decay=true |

Weight'siz op'lar (weights.rs sütunu boş — sadece akışı şekillendirirler):

| ops/full_seq.rs | fwd kernel(ler) | bwd kernel(ler) | grad buffer'ları (sınıf) |
|---|---|---|---|
| AttentionOp | FLASH_ATTENTION ×batch | FLASH_BWD_DQ + FLASH_BWD_DKDV ×batch | Overwrite |
| SiluOp | SILU_OUT | SILU_BWD | Overwrite |
| AddOp ×2 | ADD | RESIDUAL_ADD ×2 (`+=` fan-in) | Fan-in |
| QkvSplitOp | QKV_SPLIT | QKV_SCATTER + ROPE_BWD_QK | Overwrite |
| CrossEntropyOp (loss.rs) | CROSS_ENTROPY (in-place) | CROSS_ENTROPY_BWD (in-place) | logits buffer'ının kendisi (bkz. VRAM aliasing) |

## Meta protokolü

Meta = kernel parametrelerini taşıyan küçük bir tensör; `TensorMode::Meta` ile
bağlanır. Tipli struct'lar `kernels/meta.rs`'te yaşar; `KernelMeta` trait'i iki şey
verir: `upload(ctx)` (yeni sabit meta yarat) ve `write_to(tensor)` (mevcut
metayı yerinde güncelle).

İki kullanım türü:

|  | Sabit meta | Dinamik meta |
|---|---|---|
| API | `foo(gb, ..., shape)` | `foo_with(gb, ..., shape, &meta)` |
| Sahibi | kimse — emit içinde uploadlanır, buffer'ı node'un Arc'ı yaşatır | caller (op struct'ının kendi alanı, ör. `CachedAttentionOp::attn_meta`) |
| Güncelleme | asla | adım başına `write_to` (`Advance::advance`) |
| Kullanıcı | tüm train graph'leri, prefill | decode'un dinamik metaları: rope pos, cache offset, attn_len, softmax width |

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

Yapı (`Model`'in `ChatState`'i, `model.rs`):

- Katman başına K ve V ayrı birer tensör, düz `[max_context_len, dim]`
  (satır = mutlak pozisyon; head h, satır içinde `h*head_dim ..< (h+1)*head_dim`
  sütunları). Attention kernelleri cache'i stride'lı okur (H6) — per-head kopya yok.
- VRAM: 2 × num_layers × max_ctx × dim × 4B.

Kontratlar:

- Cache **RoPE'lanmış key** tutar: rope, cache_write'tan ÖNCE uygulanır
  (prefill'de `rope`, decode'da `rope_offset`). V roped değildir. Okuyan kernel
  pozisyon bilgisine ihtiyaç duymaz.
- **`pos`'un sahibi çağrıdır, `Model` değil.** `Model::prefill_logits`/
  `decode_step_logits` (ve bunları saran `ChatSession`) `pos`'u parametre
  olarak alır/tutar — eski `InferenceSession`'ın kendi içinde ilerlettiği
  `cur_len` alanı yerine, artık `Tape::advance(pos)` ile her decode adımında
  açıkça geçilir. Kerneller `pos`'u hiç görmez — canlı uzunluk onlara
  `attn_len` dinamik metasıyla gider (`Advance for CachedAttentionOp`:
  `attn_len = step + 1`, yani cache `[0, step]` aralığında geçerli, kendi
  pozisyonu dahil).
- `Model::for_chat` her çağrıda TAZE bir cache (`zeros()`) kurar;
  `Model::reset_cache()` var olan cache'i sıfırlar (ChatSession'ın `reset()`'i
  bunu çağırır). Prefill boş cache ister — `ChatSession::feed_prompt`
  `self.pos != 0` ise `ModelError::CacheNotEmpty` döner; context dolunca
  decode `ModelError::ContextFull` döner (`Model::decode_step_logits`).

## VRAM aliasing kararları

Başrol — tek `[rows, vocab]` buffer (probs + grad_logits için ayrı alloc yok):

```
lm_head matmul çıktısı (logits)
  → CE fwd  AYNI buffer'ı yerinde probs'a çevirir
  → CE bwd  AYNI buffer'ı yerinde grad_logits'e çevirir
  → lm_head bwd bunu grad_output olarak okur
```

- `LinearOp::new` out_buffer'ı kendi ayırır; `loss.rs`'teki `CrossEntropyOp`
  `tail.output(logits_id)` ile AYNI Arc'ı hem "logits'i oku" hem "yerinde
  probs/grad_logits'e çevir" için kullanır — alias bilinçlidir.
- **Sonuç**: CE fwd'yi bwd koşmadan iki kez çalıştırmak buffer'ı bozar
  (probs'un softmax'ı alınır). `Model::eval_loss` bu yüzden yalnız
  `fwd_graph.execute_captured()` çalıştırır, `bwd_graph`'a hiç dokunmaz —
  ama AYNI `t.fwd_graph`'ı `train_step` de kullandığından, `eval_loss`
  çağrıları arasına gerçek `train_step` serpiştirmek güvenlidir (bir
  sonraki `train_step` kendi fwd'ini baştan çalıştırıp eval'in bıraktığı
  probs'u ezer), ama `eval_loss`'u accumulation cycle'ın ORTASINA (zero_grad
  ile optimizer_step arasına) sokmak YANLIŞTIR — o an fwd_graph training
  verisiyle tekrar koşmadan bwd_graph'a girilmez.

Küçük paylaşımlar:

- `rsqrt_cache`: `RmsNormOp`'un kendi alanı, forward'da yazılır bwd'de
  okunur — fwd/bwd arasında tek-node'luk scratch aliası (backward yeniden
  türetmez, forward'ın hesapladığını kullanır).

## Eğitim döngüsü ve checkpoint

Accumulation cycle artık `Model::train_step` İÇİNDE değil, **çağıranın**
(main.rs'in `run_training`'i) sorumluluğu — eski `Trainer::train_step`'in
`step`/`accumulation_steps` argümanlarıyla içeride yaptığı şeyi elle yapıyor:

```rust
if step % accumulation_steps == 0 { model.zero_grad(); }
let loss = model.train_step(&inputs, &targets);   // fwd (captured) + bwd (captured)
if (step + 1) % accumulation_steps == 0 { model.optimizer_step(); } // clip + AdamW
```

- Grad ölçeği: `loss.set_grad_scale(1.0 / (rows * accumulation_steps))` —
  `Model::for_training`'de bir kez ayarlanır, effective batch normalizasyonu
  CE bwd'nin içinde olur.
- **Sıra yük taşır**: AdamW graph'ında ADAMW_SCHEDULE node'u parametre
  node'larından ÖNCE gelir. step=0'da bias_correction = 1−β⁰ = 0 → sıfıra
  bölme → NaN; schedule önce koştuğu için AdamW hiçbir zaman step=0 görmez.
- `Model::train_step` HER çağrıda loss'u CPU'ya okur (eski `Trainer`'ın
  `log_every`'de bir okuma optimizasyonu yeni sistemde yok — bilinen, henüz
  ele alınmamış bir performans notu, "Fikir kuyruğu"na bakın).
- **Eval harness** (main.rs): `data/eval.txt` varsa başlangıçta baseline +
  her EVAL_EVERY step'te held-out loss/perplexity — `Model::eval_loss`
  yalnız forward çalıştırır (grad'lara, optimizer'a, accumulation cycle'ına
  dokunmaz). Sonuç konsola + `checkpoints/eval_log.txt`'ye (step, loss, ppl).
  Dosya eğitim verisiyle örtüşmemeli — yoksa sayı ezberi ölçer.

Checkpoint:

- **V3 = TEK format** = `AKV3` magic + mimari başlığı (vocab/dim/heads/
  layers/ffn; yüklerken eşleşmezse hata) + `train_step` (loop sayacı) +
  `schedule_step` (AdamW cycle sayacı) + weight'ler `weights.params()`
  sırasında + AdamW (m, v) momentleri aynı sırada (**sıra format
  sözleşmesidir**). `moments` boş = weights-only dosya (migre v1/v2):
  yüklenince optimizer soğuk, schedule 0'dan.
- `checkpoint.rs`'in `save`/`load` fonksiyonları `ModelWeights<B>` + opsiyonel
  `&[(Arc<Tensor<B>>, Arc<Tensor<B>>)]` moment listesi üzerinde çalışır —
  sistem-agnostik (`Model` ve eski `Trainer` AYNI format sözleşmesini
  paylaşırdı, bu yüzden checkpoint dosyaları arasında hiç format geçişi
  gerekmedi). `Model::save_checkpoint`/`load_checkpoint` bunun ince
  sarmalayıcıları — `t.optimizer.moments()`/`current_schedule()`/
  `load_state()` (`AnyOptimizer`) üzerinden.
- Legacy v1/v2 okuyucuları YALNIZ `bin/migrate_checkpoint_v3.rs`'te yaşar;
  kütüphane v3 dışında hiçbir şeyi okumaz/yazmaz.
- main.rs resume sırası: en yeni `model_step_*.bin` (step dosyanın İÇİNDEN
  okunur, dosya adı yalnız migre dosyalar için fallback) → yoksa
  `checkpoints/model_final.v3.bin` → yoksa sıfırdan.

## Invariantlar

Assert'e (henüz) dökülememiş kurallar. Bugfix turunda her düzeltmenin
invariantı buraya bir satır olarak eklenir.

- **`Model::params()` / `ModelWeights::params()` sırası format sözleşmesidir**
  — checkpoint dosyası ve AdamW momentleri bu sıraya zip'lenir (bkz. Grad
  topolojisi).
- **Pozisyonlar batch elemanı başına 0'dan başlar** (row_offset tasarımı; RoPE
  açısı yerel `token_idx`'ten). `batching_validation` (model_tests.rs) bunu
  kanıtlar.
- **RMSNorm eps'in tek kaynağı `cfg.norm_eps`'tir** — train ve chat aynı
  config alanını okur; yine de 1e-5'ten oynatmak eğitilmiş checkpoint'in
  numeriğinden sapmaktır, model başına sabit tut.
- **ADAMW_SCHEDULE, AdamW node'larından önce koşar** — AdamW step=0 görürse
  bias_correction sıfırlanır → NaN.
- **CE fwd'den sonra bwd koşmadan CE fwd tekrar koşulamaz** (in-place; bkz.
  VRAM aliasing) — `eval_loss`'u accumulation cycle ortasına sokma.
- **Cache roped K tutar; `pos`'u yalnız çağıran ilerletir** (`ChatSession`
  veya elle `Model::decode_step_logits`) — cache buffer'ları `Model::for_chat`
  ile birlikte doğar, `reset_cache()` içeriği sıfırlar ama graph'ı yeniden
  kurmaz (aynı Arc'lara işaret etmeye devam eder).
- **Dinamik metalı + matmul'lu graph capture edilemez** (Meta protokolü) —
  decode graph'ının hiç `execute_captured` KULLANMAMASININ sebebi budur.
- **`cfg.eos_token = vocab_size - 1`** (`ModelConfig::new`) — artık hardcode
  değil, GPT-2 BPE `<|endoftext|>` kuralına göre config'ten türetilir.
- **Tokenizer offline-first**: yerel `tokenizer.json` / `SEQUEXA_TOKENIZER`
  varsa ağa çıkılmaz; ilk indirme yerel kopya bırakır.
- **`checkpoints/model_final.bin` (v1) dokunulmazdır** — tek eğitilmiş model;
  migrasyonlar kopya üzerinde yapılır.
- **Weight decay yalnız matmul weight'lerine uygulanır** — `EmbeddingOp`/
  `RmsNormOp::param()` sabit `decay=false` döner, `LinearOp::param()`
  kurucusuna geçilen `decay` bool'unu iletir (tüm `LinearOp`'lar `true` ile
  kuruluyor). `AdamW` bu bayrağı param başına gerçekten uyguluyor (bkz.
  optim/adamw.rs) — eski Trainer'ın tek-grup tasarımının (norm/embedding'i
  de decay'liyordu) düzeltildiği yer.
- **embedding_bwd atomik CAS kullanır** — token tekrarı yarışının çözümü.
- **`head_dim` her zaman 64 olmalı** — flash attention WGSL'i buna hardcode
  (`assert_flash_head_dim`, `kernels/emit.rs`); yeni bir mimari/config
  denerken önce bunu kontrol et (`diagnose_model`'in CHECK 8'i bunu bir kez
  yanlış yakaladı, bkz. Test haritası).

## Test haritası

| Test | Neyi koruyor |
|---|---|
| `tests/model_tests.rs::full_chain_gradcheck` | GERÇEK Model fwd/bwd zincirinde analitik grad == sayısal grad |
| `tests/model_tests.rs::checkpoint_roundtrip` | save/load bit-exact (weight + moment + iki sayaç) |
| `tests/model_tests.rs::flat_weights_roundtrip` | `to_flat_weights`/`set_flat_weights` optimizer state'e dokunmuyor |
| `tests/model_tests.rs::batching_validation` | batch=N tek geçiş == N ardışık accumulation — row_offset tasarımının kanıtı |
| `tests/model_tests.rs::grad_clip_validation` | GPU clip zinciri == host formülü |
| `tests/train_tests.rs`, `tests/kernels_emit_tests.rs`, ... | her kaynak dosyanın kendi testi, `#[path]` ile o dosyanın gerçek modülüne asılı (bkz. `src/tests/`) |
| `tests/kernels_emit_tests.rs::flash_attention_validation` | flash fwd+bwd == bağımsız düz-Rust CPU referansı |
| `tests/kernels_emit_tests.rs::kernel_fusion_validation` | rope_qk == 2×rope; qkv_split/scatter == head_gather/scatter zinciri |
| `tests/kernels_emit_tests.rs::decode_kernel_validation` | 3-dispatch cached attention == CPU ref; m=1 GEMV yönlendirmesi |
| `tests/main_tests.rs` | eval penceresi kesimi: sabit, örtüşmesiz, batch'e hizalı |
| `optim/adamw.rs` (tests/adamw_tests.rs) | device schedule == host `cosine_lr`; weight'ler doğru yöne hareket ediyor |
| `nn/sampling.rs` (tests/sampling_tests.rs) | greedy / top-k / top-p özellikleri |
| `bin/diagnose_kernels.rs` (elle koşulan) | CHECK 3 (HeadGather/Scatter), CHECK 4 (RMSNorm bwd), CHECK 7 (CE kapalı-form) — sistem-agnostik, `layers.rs`'in standalone sarmalayıcılarıyla |
| `bin/diagnose_model.rs` (elle koşulan) | CHECK 1 (param sayısı @ hall_1 ölçek), CHECK 2 (grad akışı @ gerçek derinlik/genişlik), CHECK 5 (accumulation), CHECK 8 (memorization smoke), CHECK 9 (prefill vs decode-cache logit paritesi — sayısal, argmax DEĞİL), CHECK 10 (cache hız kazancı) |

Kurallar:

- Her zaman `cargo test -- --test-threads=1` — paralel testler eşzamanlı
  WgpuBackend yüzünden segfault eder.
- Yeni op eklerken: `kernels_emit_tests.rs`'teki ilgili validation modülüne
  CPU-referanslı satır ekle.
- Yeni bir "X vs Y aynı mı" tarzı check yazarken **sampled/argmax token değil
  ham sayısal değeri karşılaştır** — eğitilmemiş/rastgele ağırlıklarda
  logit'ler tepesiz olur, iki bağımsız kernel yolunun mikroskobik
  floating-point farkı argmax'ı çevirebilir (CHECK 9'un ilk versiyonunun
  düştüğü tuzak tam bu).
