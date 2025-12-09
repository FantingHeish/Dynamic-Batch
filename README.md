# Dynamic Batch

## 🎯 專案簡介
本專案實作 大型語言模型（LLM）推論的 Dynamic Batch Inference Engine，透過 batch prefill、KV-cache reuse、autoregressive batch decode 技術，大幅提升 GPU 的運算利用率與整體輸送率（throughput）。
展示了 批次推論（batching）是 LLM 推論加速的核心，並量測 batch_size 對 TTFT、latency、P95 latency 與 tokens/sec 的影響。模型採用 Qwen2.5-0.5B-Instruct，使用 PyTorch + HuggingFace Transformers 原生 API 實作完整 prefill/decode pipeline。
✅
## 🚀 技術核心
### 🔸 1. Batch Prefill（一次前向計算所有序列）
#### 🎯 作法：
- 輸入多個 prompt → dynamic padding → attention_mask → 單次前向：
- Prefill 主要負責：
  - 產生 past_key_values（KV-cache）
  - 計算每個序列的 true input length
  - 抽出每個序列的最後一個 token（decode 起點）
#### 👉 達成：
- ✔ reduce 重複計算
- ✔ prefill 的計算量從 N 次降為 1 次
- ✔ 建立 batch decode 的條件

### 🔸 2. Batch Autoregressive Decode（逐 token 批次 decode）
#### 🎯 作法：
1. Decode loop：（每個 decode step）
- 共用 KV-cache（大幅減少矩陣乘法）
- multi-head attention 只需處理 新增位置
- batch_size 越大 GPU 越有效率（tensor shape 更大 → 更好利用 CUDA kernel）
#### 👉 達成：
- ✔ throughput 成長
- ✔ decode latency 下降

### 🔸 3. Dynamic Padding + Attention Masking
#### 🎯 作法：
- 忽略 padding token
- 正確計算每條序列的 real length
- 保持 batch 計算一致性
#### 👉 達成：
- ✔ 不浪費計算在 pad 上
- ✔ 各序列可不同長度
- ✔ 輕量版本的 PagedAttention（概念上相似）

### 🔸 4. Per-request Metrics Profiling
| 指標                        | 用途                          |
| ------------------------- | --------------------------- |
| TTFT（Time to First Token） | 測量 decode 第一個 token 的速度     |
| prefill_ms                | 前向一次多序列花費時間                 |
| decode_ms                 | autoregressive 全部 decode 時間 |
| latency_ms                | 單個 request 的 end-to-end 時間  |
| P50/P95 latency           | 衡量 tail latency，生產系統關鍵指標    |
| throughput (tokens/sec)   | 整體效能                        |


## 🧰 技術架構
| 模組                  | 技術                                                       |
| ------------------- | -------------------------------------------------------- |
| **深度學習框架**          | PyTorch 2.0+、CUDA FP16                                   |
| **推論引擎核心**          | Batch Prefill、KV-cache Reuse、Autoregressive Batch Decode |
| **張量處理**            | Dynamic Padding、Attention Masking                        |
| **模型呼叫**            | HuggingFace Transformers (`use_cache=True`)              |
| **Batch Engine 設計** | Static Baseline vs Dynamic Batch Scheduler（依序填滿 batch）   |
| **效能監測**            | TTFT、Prefill/Decode 拆解計時、Latency P50/P95、Throughput      |
| **測試模型**            | Qwen2.5-0.5B-Instruct（HF 官方權重）                           |
| **工作負載**            | 模擬 32 筆 LLM 請求（各別測量 token 數、延遲、P95）                      |
| **輸出分析**            | pandas + numpy（產生 CSV 與統計表）                              |


## 📊 效能指標
| 指標 | Baseline | (batch=1)優化後 | (dynamic batch)改善幅度 |
|------|------|------|------|
| **吞吐量 (tokens/s)** | 68 | 501 | 7.37x |
| **平均延遲** | 3.2s | 0.45s | 86% ↓ |
| **GPU 使用率** | 45% | 89% | 44% ↑ |
| **記憶體使用** | 8.2GB | 7.8GB | 5% ↓ |
| **最大 Batch Size** | 1 | 16 | 16x |

## 📊 Benchmark 結果
### 測試環境
- GPU: NVIDIA A100 (40GB)
- Model: Qwen2-1.5B-Instruct
- Input Length: 128 tokens (avg)
- Output Length: 256 tokens (avg)

### 吞吐量比較
<img width="545" height="177" alt="Screenshot 2025-11-11 at 05 54 48" src="https://github.com/user-attachments/assets/47417f64-79e8-4dc4-8df6-f99c03560586" />

### 不同模型規模測試
| Model | Baseline | Dynamic Batch | Inprovement |
|------|------|------|------|
| **Qwen2-1.5B** | 68 tok/s | 501 tok/s | 7.37x |
| **LLaMA-7B** | 24 tok/s | 418 tok/s | 6.17x |
| **Mistral-7B** | 28 tok/s | 165 tok/s | 5.89x |

## 環境需求
🖥️ 環境需求
- Python 3.9+
- CUDA 11.8+
- PyTorch（支援 FP16）
- transformers >= 4.44
- GPU ≥ 6GB VRAM
