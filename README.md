# Dynamic Batch

## 🎯 專案簡介
針對大型語言模型（LLM）高吞吐量場景設計的 動態批次推論系統，
透過 CUDA 優化、KV Cache 管理與 GPU 記憶體最佳化，大幅提升推論效率。
適用於服務多個併發使用者的生產環境，如 API 服務、對話機器人平台等。
本專案採用 PyTorch CUDA + Dynamic Batching + Attention Masking 架構，
實現智慧批次調度與記憶體管理，達成 7.37× 吞吐量提升與 86% 延遲降低，
在保持回應品質的前提下，最大化 GPU 資源利用率。

## ✅ 核心功能
✅ 動態批次調度: 即時調整 batch size 適應不同請求長度
✅ CUDA 優化: 自定義 CUDA kernel 加速 attention 與 padding 操作
✅ KV Cache 管理: 高效快取機制減少重複計算
✅ GPU 記憶體優化: Dynamic padding 最小化記憶體浪費
✅ Attention Masking: 精確處理變長序列的 attention
✅ 吞吐量監控: 即時追蹤 tokens/sec、GPU 使用率

## 🧰 技術架構
| 模組 | 技術 |
|------|------|
| **深度學習框架** | PyTorch 2.0+、CUDA 11.8+ |
| **核心技術** | Dynamic Batching、KV Cache、Attention Masking |
| **GPU 優化** | Custom CUDA Kernels、Memory Pooling |
| **推論引擎** | HuggingFace Transformers、Flash Attention |
| **排程策略** | Priority Queue、First-Come-First-Served |
| **測試模型** | Qwen2-1.5B、LLaMA-7B |
| **部署方式** | FastAPI + Uvicorn |

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
- Python 3.9+
- CUDA 11.8+ / 12.1+
- GPU 記憶體 ≥ 8GB (建議 16GB+)
- PyTorch 2.0+ with CUDA support
