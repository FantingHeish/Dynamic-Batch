# Dynamic Batch

## 🎯 專案簡介
本專案針對 LLM 推論的效能瓶頸（GPU idle、prefill 重複計算、decode 低效率），打造一個 研究用 Dynamic Batch Inference Engine，功能包含：
- Dynamic batching（合併多個 request）
- Shared prefill（一次前向計算多序列）
- KV-cache reuse（減少 attention 計算）
- Dynamic padding + attention masking
- Per-request TTFT / Latency profiling
- GPU utilization & memory profiling（NVML）
測試模型：Qwen2.5-0.5B-Instruct
測試硬體：NVIDIA Tesla T4（15GB）

✅
## 🚀 技術核心
### 🔸 1. Batch Prefill（一次前向計算所有序列）
#### 🎯 作法：
- 輸入多個 prompt → dynamic padding → attention_mask → 單次前向：
  - 將多個 prompt 做 dynamic padding → attention_mask
  - 使用單次 model.forward 執行 prefill
  - 產生 past_key_values（KV-cache）
  - 抽出每個序列最後 token 作為 decode 起點
#### 👉 達成：
- ✔ 大幅降低重複計算
- ✔ prefill 計算量由 N 次 → 1 次
- ✔ 建立 batch decode 的基礎

### 🔸 2. Batch Autoregressive Decode（逐 token 批次 decode）
#### 🎯 作法：
1. Decode loop：（每個 decode step）
- 共用 KV-cache（大幅減少矩陣乘法）
- multi-head attention 只需處理新增位置
- batch 越大 → CUDA kernel 越飽和 → GPU 利用率更高
#### 👉 達成：
- ✔ throughput 成長
- ✔ decode latency 下降

### 🔸 3. Dynamic Padding + Attention Masking
#### 🎯 作法：
- 為 batch 中較短的序列自動 padding
- 以 attention_mask 確保模型忽略 pad token
- 保持 batch 計算一致性
#### 👉 達成：
- ✔ 不浪費計算在 pad token
- ✔ 支援不同長度序列
- ✔ 本質為 batch-level padding，不等同於 PagedAttention（但概念上同樣是減少不必要的計算）

### 🔸 4. Per-request Metrics Profiling
| 指標                        | 用途                          |
| ------------------------- | --------------------------- |
| TTFT（Time to First Token） | 測量 decode 第一個 token 的速度     |
| prefill_ms                | 前向一次多序列花費時間                 |
| decode_ms                 | autoregressive 全部 decode 時間 |
| latency_ms                | 單個 request 的 end-to-end 時間  |
| P50/P95 latency           | 衡量 tail latency，生產系統關鍵指標    |
| throughput (tokens/sec)   | GPU 的整體推論效率                   |


## 🧰 技術架構
| 模組              | 技術                                                |
| --------------- | ------------------------------------------------- |
| 深度學習框架          | PyTorch 2.0+、CUDA FP16                            |
| 推論引擎核心          | Batch Prefill、Batch Decode、KV-cache Reuse         |
| 張量處理            | Dynamic Padding、Attention Masking                 |
| 模型              | HuggingFace Transformers (`use_cache=True`)       |
| Batch Engine 設計 | Static Baseline vs Fixed-size Dynamic Batch Fill  |
| 效能監測            | TTFT、Prefill/Decode 時間、Latency P50/P95、Throughput |
| GPU Profiling   | NVML：GPU Utilization / Memory Tracking            |
| 工作負載            | 模擬 32 個 request、每次 decode 64 tokens               |
| 統計              | pandas / numpy（輸出 summary table）                  |



## 📊 效能指標
### 📊 1. Benchmark 結果
本次測試共 32 個請求、每次 decode 64 tokens，對比：
- static_single (batch=1) → baseline
- dynamic_bs=2 / 4 / 8 → 模擬 dynamic batching 行為
- bs=8 對 T4) 是最佳 trade-off，顯示 dynamic batching 的效果與 SM/Memory 結構有關。
| Mode              | Throughput (tokens/s) | Speedup   | P95 Latency | GPU Util Avg | GPU Util Max | Max Mem |
| ----------------- | --------------------- | --------- | ----------- | ------------ | ------------ | ------- |
| **static_single** | **248**               | 1.00×     | **7994 ms** | 21%          | 46%          | 2.46 GB |
| **dynamic_bs=2**  | 302                   | 1.21×     | 4450 ms     | 36%          | 52%          | 2.46 GB |
| **dynamic_bs=4**  | 791                   | 3.19×     | 2588 ms     | 37%          | 45%          | 2.48 GB |
| **dynamic_bs=8**  | **869**               | **3.50×** | **2356 ms** | **37%**      | 43%          | 2.53 GB |


#### 🎯 2. Key Findings
##### ⭐ 1. Throughput 上升：3.5×
Dynamic batching 讓 GPU 得以一次處理更多序列
##### ⭐ 2. Latency 顯著下降：−70.5%
多個請求 共用一次 Prefill，大幅攤平 Self-Attention 的固定成本。Decode 過程也因多個序列併入同一 kernel 而吞吐提升。
##### ⭐ 3. GPU 使用率提升 +15%
Baseline GPU idle 明顯（21%）。Dynamic batching 後 GPU 利用率提升到 36–38%：
##### ⭐ 4. GPU Memory 幾乎不變（+3%）
Dynamic padding + KV-cache reuse 成功控制記憶體。2.46GB → 2.53GB（+2.9%）。
##### 🎉 bs=8 在 T4 GPU 上是最佳 sweet spot（效能 → 記憶體的最佳trade-off）

## 環境需求
🖥️ 環境需求
- Python 3.9+
- CUDA 11.8+
- PyTorch（支援 FP16）
- transformers >= 4.44
- NVIDIA GPU（≥ 6GB VRAM）
- transformers >= 4.44
- GPU ≥ 6GB VRAM
