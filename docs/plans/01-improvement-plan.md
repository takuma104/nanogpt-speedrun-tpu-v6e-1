# NanoGPT speedrun on TPU v6e-1 — 改善方針

目的: `val_loss <= 3.28` 到達までの `train_time` を TPU v6e-1 (1 chip) で短縮する。
出発点は `third-party/modded-nanogpt-jax`（v6e-8 で約 10 分）。比較対象は RTX 5090 x1 版（`third-party/nanogpt-speedrun-rtx5090x1`、約 1077 s）。

## 結論

- 参考 JAX 実装を v6e-1 でそのまま動かすと、単純計算で学習だけで **約 80 分**（v6e-8 の学習約 611 s × 8）。ただし全バッチを事前に HBM へ載せる実装なので、そのままでは載らない可能性が高い。
- **システム面**: ログから逆算すると学習時間の約半分が素朴な attention（T×T 行列を HBM に実体化）。splash attention 化で −35〜40% の見込み。システム最適化だけでは 40 分前後が上限。
- **レシピ面（最大のレバー）**: JAX 版は 2024 年 11 月頃のレシピ（約 878M tokens、d=1024）。RTX 5090 版の最新レシピは約 339M tokens、d=768/11 層で、総 FLOPs が **約 4〜5 倍** 少ない。20 分を切るにはレシピの移植が必須。

## 1. 現状の把握（`records/sub10m.txt`, v6e-8）

| 区間 | 10 step の時間 | tokens/step |
|---|---|---|
| seq 1024（前半 837 step） | 3.07 s | 524k |
| seq 2048（後半 838 step） | 4.23 s | 524k |

- 同じトークン数で +38%。実行 FLOPs の増加は約 11%（XLA の attention は causal でも T×T を全部計算する）。
- `step(T) = a + b·T` と分解すると、attention 相当は seq1024 で約 38%、seq2048 で約 55%、run 全体で **約 48%**。
- attention 以外は MFU 約 45% 相当（v6e bf16 ピーク 918 TFLOPS/chip 基準）。作者ブログでの全体 MFU は 23%。

### 計算量の比較

| | JAX 版 | RTX 5090 版（最新レシピ） |
|---|---|---|
| 構成 | 12 層, d=1024, 4×256 heads, MLP 4096 | 11 層（attn 10 層）, d=768, 6×128 heads, MLP 3072 + 各種 |
| 学習トークン | 約 878M（1675 step × 524k） | 約 339M（1275 step, 131k→262k→393k tokens/step） |
| 行列積 FLOPs/token（学習） | 約 1.2 G（+ attention） | 約 0.7 G（sampled softmax で更に減） |
| 総 FLOPs | 約 1.2e18 | 約 2.3e17 |

## 2. Phase 1: v6e-1 で動かし、正しく測る（必須）

- **データ**: `load_dataset` が全 1675 step 分を int32 で事前に device_put（x, y 合計約 7 GB）。ホスト側に保持し、数 step 先読みで device_put する方式へ。uint16 で転送しデバイス側で cast。
- **micro-batch**: `micro_batch_size=16` は v6e-8 では 1 chip あたり 2 系列。v6e-1 で 16 系列（32k tokens）にすると f32 logits だけで 6.6 GB + attention 行列。OOM しない値に調整。
- **計測**: train_time を測っていない（ログ時刻は非同期実行でずれ、validation 込み）。validation を除外し、`block_until_ready` で区切るタイマーを追加。validation は v6e-1 で 1 回約 40 s × 14 回なので頻度を下げる。
- **validation の精度**: softcap が bf16 で計算されている（`train.py:765-768`）。logits が 15〜30 付近で刻み 0.125。最初に f32 へ直し、以後の評価定義として固定する。
- **プロファイル**: `jax.profiler` + xprof で op 単位の内訳を取る。`compiled.memory_analysis()` / `cost_analysis()` でメモリ・FLOPs を確認。

## 3. Phase 2: システム面（loss 曲線は基本的に変わらない）

1. **Attention → Pallas splash attention**（causal を block-sparse に）。d_head=256 は v6e の 256×256 MXU と相性が良い。作者は「flash/splash は遅かった・初回で動かなかった」と書いているので、ブロックサイズを詰めたマイクロベンチから始める。後の sliding window・文書マスクの土台にもなる。
2. **lm_head + cross-entropy の融合/分割**: V=50304 の logits を f32 で丸ごと作っている。lm_head は FLOPs の約 25%。分割でメモリが空き micro-batch を大きくできる。
3. **その他**: micro-batch、再計算（`jax.checkpoint`）方針、XLA flags の調整（数%）。Muon の同形状行列のバッチ化は v6e-1 では step 全体比が小さいので後回し。

## 4. Phase 3: 数値精度・optimizer（安いが学習曲線が変わる）

- **全部 bf16**: パラメータ、勾配累積（`train.py:805`）、Adam の m/v、Muon の momentum。warmdown 末期（lr 0.1×）は Muon 更新の相対量が約 0.4% で bf16 の丸め幅と同程度。PyTorch 版は fp32 マスター重み。作者の bf16 化は「matmul が f32 で走っていた」速度バグが原因なので、matmul だけ bf16 に明示 cast すれば fp32 化しても速度はほぼ落ちないはず。
- **Muon のスケール係数**: `train.py:355` の `sqrt(max(1, rows/cols))` は PyTorch の (out, in) 配置前提。JAX 版の MLP 重みは (in, out) なので c_fc（本来 2 → 実際 1）と c_proj（本来 1 → 実際 2）で逆になっている疑い。要検証。
- **QKV の直交化**: 3072×1024 の 1 行列として直交化。最新版は Q/K/V 別（さらに head ペア単位）。
- **バッチサイズ**: 524k tokens/step 固定は 8 台並列時代の名残。1 chip なら通信コストがないので、小さいバッチから始めるスケジュール（最新版 131k → 393k）の方がトークン効率が良いはず。記録の最終 val_loss は 3.2751（約 0.005 の余裕、1 run のみ）。

## 5. Phase 4: レシピの近代化（最大のレバー）

- **TPU と相性が良い**（行列積・要素ごとの演算中心で XLA が自動融合）:
  - 文書マスク、BOS 揃え、sliding window とそのスケジュール（splash の `LocalMask` + `segment_ids`）
  - YaRN、attention を飛ばす層、smear gate、attention gate、XSA、key offset、MTP、U-net / backout
  - Polar Express、NorMuon、cautious WD、Adam を奇数 step のみ、embed/lm_head の tied 初期化 → 分離
- **注意が必要**（gather/scatter・メモリ帯域律速）: value embeddings（作者は大幅な速度低下で断念）、bigram/trigram hash embedding、sampled softmax。RTX Exp 2b と同様に「取り出した行に対する勾配を出し、scan の carry の fp32 累積バッファへ in-place で scatter-add」して、V×d の密な勾配を毎 micro-batch 作らない。v6e には SparseCore もあるが JAX から使うのは重い。
- **後回し**: DC attention（Triton 自作カーネル → Pallas 書き直しが必要）。FP8 は v6e 非対応 → int8（1836 TOPS = bf16 の 2 倍、AQT 等）での学習が後半の候補。

## 6. RTX 5090 との違いで判断が変わる点

- **計算量/帯域比**: v6e 約 570 FLOP/byte（918 TFLOPS / 1.6 TB/s）、RTX 5090 約 117 FLOP/byte。TPU ではメモリ帯域律速の処理の相対コストが約 5 倍。巨大 embedding 表で精度を稼ぐ RTX 路線より「行列を大きくして小技を減らす」方が最適になる可能性。
- **MXU 256×256**: head_dim=128 だと QK^T の縮約次元が MXU の半分しか埋まらない（作者が 4×256 にした理由）。最新レシピの 6×128 を採るかは実験項目。
- **XLA の自動融合**: RTX で Triton を書いた系統（relu² MLP、CE）は XLA がある程度やる。自作カーネルは Pallas。
- **静的 shape**: window/batch/seq のスケジュール変化は AOT で全バリエーションをコンパイル（既存実装の方式で OK）。

## 進め方

1. Phase 1（v6e-1 で動かし、正しく測る）→ ベースライン測定 + プロファイル
2. splash attention
3. 数値面の安い修正（fp32 マスター重み、Muon 係数の検証）
4. レシピ近代化: 文書マスク + sliding window → バッチスケジュール → value embedding（行勾配方式）→ 小技 → sampled softmax / int8

## 参考

- [modded-nanogpt-jax 作者のブログ](https://nor-blog.pages.dev/posts/2025-08-21-modded-nanogpt-jax/)
- [Cloud TPU v6e ドキュメント](https://cloud.google.com/tpu/docs/v6e)
