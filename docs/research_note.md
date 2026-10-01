# NanoGPT speedrun on TPU v6e-1 — 結果ノート

目的: `val_loss <= 3.28` 到達までの `train_time` を短縮する（1x TPU v6e, 32 GB HBM）。方針は [plans/01-improvement-plan.md](plans/01-improvement-plan.md)。

## 現在の最小 train_time

（未測定）

## 環境（2026-10-02 調査）

- GCP プロジェクト `tpuproject-510313`（TPUProject）、Cloud TPU API 有効
- v6e-1 を提供するゾーン: asia-east1-c, asia-northeast1-b, asia-south1-b/c, asia-southeast1-b, europe-west4-a, southamerica-east1-c, southamerica-west1-a, us-central1-a/b/c, us-east1-d, us-east4-a, us-east5-a/b/c, us-south1-a/c, us-west1-c
- クォータ: v6e（on-demand / preemptible とも）既定 16 chip/zone。ただし us-east4-a/b と us-east5-c は 0
- Spot 価格の目安（第三者サイト、変動あり）: us-east4 $0.24/h, asia-southeast1 $0.27/h, us-central1 / us-east1 / us-west1 $0.65/h。on-demand は $2.70/h（米国）
- 2026-10-02 の Spot 空き状況: asia-southeast1-b, us-central1-a/b/c は容量なし、us-east1-d で作成成功（CREATING に約 15 分）→ **約 10 分で preempt（ノードごと削除）**。直後の再試行（us-east1-d, us-central1-a/b/c, us-west1-c, us-south1-a/c, asia-southeast1-b, us-east5-a/b）は全ゾーン容量なし（us-east1-d は CREATING で約 23 分待った末に失敗）

### VM（`ct6e-standard-1t`, runtime `v6e-ubuntu-2404`）

- AMD EPYC 9B14, 44 vCPU, RAM 172 GB, ディスク 96 GB, Ubuntu 24.04, kernel 6.11
- システム Python 3.12 → uv で Python 3.13.15 を使用
- jax / jaxlib 0.11.2, libtpu 0.0.48。`device_kind = "TPU v6 lite"`, 1 core/chip
- HBM `bytes_limit` 33.5 GB
- Transparent hugepages が無効だと JAX が警告 → `echo always > /sys/kernel/mm/transparent_hugepage/enabled`（再作成のたびに必要）
- データ（train 10 chunk + val、2.1 GB）の HF からのダウンロードは 25 s

### マイクロベンチ（`scripts/probe_tpu.py`）

| 項目 | 実測 | 備考 |
|---|---|---|
| bf16 matmul 4096³ | 805 TFLOPS | ピーク 918 の 88% |
| bf16 matmul 8192³ | 785 TFLOPS | |
| int8 matmul 8192³（int32 累積） | 1337 TOPS | bf16 の 1.7 倍 |
| fp8 e4m3 matmul 8192³ | 545 TFLOPS | **bf16 より遅い**（v6e はネイティブ FP8 なし） |
| HBM 要素演算 r+w（1 GiB f32） | 1185 GB/s | 公称 1600 GB/s の 74% |

→ FP8 系の RTX 5090 での改善（FP8 MLP backward、FP8 lm_head）は v6e では使えない。低精度化するなら int8。

## 実験ログ
