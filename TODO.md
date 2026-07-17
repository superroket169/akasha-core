# TODO

Yalnız yapılacak işler; biten buradan silinir (tarihçe git log'da).
Numaralı bug/optimizasyon listesi: [wilupgu/REFACTOR.md](../wilupgu/REFACTOR.md).
Mimari fikirler ve Big Refactor planı: [ARCHITECTURE.md](ARCHITECTURE.md).

## Koşu öncesi (continued pretraining hazırlığı)

- [ ] `data/eval.txt` oluştur — yeni corpus'tan ~100KB+ dilim, `train.txt`'ye
  GİRMEDEN önce ayrılmalı (örtüşürse perplexity ezberi ölçer). Eval harness
  dosya yoksa kendini kapatıp uyarı basar, koşuyu engellemez.
- [ ] Arkadaşın RTX 4050'sinde tester_script.bat round-trip'i: CUDA batching
  parity + bf16 gemm testleri + release build.
- [ ] İlk gerçek B=4 koşusuyla VRAM kalibrasyonu — OOM olursa BATCH_SIZE'ı
  düşür, effective batch 64 kalacak şekilde ACCUMULATION_STEPS'i artır.

## Koşu ve sonrası

- [ ] **Continued pretraining** (~10 gün) — daha büyük, daha çeşitli dataset;
  ilk koşu ~15M-token WikiText-103 dilimini birkaç kez döndürmüştü (güçlü
  yerel gramer, zayıf uzun-menzil tutarlılık bundan). Eval eğrisi
  `checkpoints/eval_log.txt`'de.
- [ ] Big Refactor (Block trait) — koşu sırasında tasarlanıp yapılır, plan
  ARCHITECTURE.md'nin son bölümünde.
- [ ] "v1'i geçtik mi" kararını eval eğrisine bağla — held-out perplexity artık
  ölçülüyor; hangi eşiğin/plateau davranışının "yeter" sayılacağına karar ver.
- [ ] Chat/instruction fine-tuning — continued pretraining'den SONRA. Full-FT
  (LoRA değil: model küçük, 4050 full-train'i zaten kanıtladı, mevcut
  backward/AdamW yolu sıfır yeni kernel'le yeniden kullanılır). Gerekenler:
  küçük instruction/chat dataset'i (on binlerce örnek yeterli olabilir) +
  chat template/delimiter konvansiyonu.

## Orta vade

- [ ] Custom tokenizer — yeni dataset'e kendi BPE'sini eğitmek (sabit yabancı
  vocab, dataset WebText-benzeri olmaktan çıkınca kötü uyum). Non-trivial:
  BPE-training implementasyonu (ya da güvenilir crate) + vocab boyutu kararı;
  vocab değişirse bu sıfırdan pretraining demektir, mevcut checkpoint'e
  hot-swap edilemez.
- [ ] wilupgu CPU backend'i rayon'la paralelleştir (şu an tek thread,
  ~1s/token) — arkadaş GPU'suz kalındığında gerçekten yük taşıyacaksa.

## Uzun vade / açık sorular

- [ ] Multi-turn chat context (şu an her prompt taze KV-cache ile başlıyor —
  `take_cache()` her `generate()` öncesi; konuşma hafızası yok).
- [ ] Ölçek büyütme (katman/dim) — veri miktarı/kalitesi bariz darboğaz
  olmaktan çıkmadan prematüre.
- [ ] License dosyası — iki repoda da yok.
