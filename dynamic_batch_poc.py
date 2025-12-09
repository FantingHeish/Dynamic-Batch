# ============================================================
# PoC#2: Dynamic Batch Inference for LLM Throughput Optimization
#
# Model: Qwen/Qwen2.5-0.5B-Instruct
# Focus:
#   - Dynamic batch inference engine
#   - KV-cache reuse & attention masking
#   - TTFT / P95 / throughput / latency monitoring
# ============================================================

import os
import time
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

print("=" * 70)
print("PoC#2: Dynamic Batch Inference for LLM Throughput Optimization")
print("=" * 70)

# ------------------------------------------------------------
# PART 0: Optional pip install (for notebook / Colab)
# ------------------------------------------------------------
try:
    import torch  # noqa
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
    import pandas as pd  # noqa
    import numpy as np  # noqa
except ImportError:
    import subprocess
    import sys
    print("\n📦 Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.44.0", "accelerate>=0.33.0",
        "torch", "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy"
    ])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    import numpy as np

# ============================================================
# PART 1: Environment & Model
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"\n✅ Device: {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU: {props.name}, {props.total_memory / 1e9:.2f} GB")

print(f"\n🚀 Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()

print("✅ Model loaded")


# ============================================================
# PART 2: Utility – Sampling & Request Data Structure
# ============================================================

def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> torch.Tensor:
    """
    Simple nucleus sampling for next token.
    logits: [batch, vocab]
    return: [batch, 1] int64 tensor of sampled token ids
    """
    if temperature <= 0.0:
        # greedy
        next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        return next_ids

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # nucleus (top-p) sampling
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # mask tokens beyond top_p
    cutoff = (cum_probs > top_p).float()
    # keep first token above cutoff
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = 0.0
    sorted_probs = sorted_probs * (1.0 - cutoff)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # sample in sorted space
    batch, vocab = sorted_probs.shape
    next_sorted = torch.multinomial(sorted_probs, num_samples=1)  # [batch, 1]

    # map back to original token id
    next_ids = torch.gather(sorted_indices, -1, next_sorted)
    return next_ids


@dataclass
class InferenceRequest:
    """
    Simulated inference request for research benchmark.
    """
    request_id: int
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class RequestMetrics:
    request_id: int
    mode: str             # "static_single" or "dynamic_batch"
    batch_size: int
    total_tokens: int
    ttft_ms: float        # Time to first token
    latency_ms: float     # End-to-end time (prefill + decode)
    prefill_ms: float
    decode_ms: float


# ============================================================
# PART 3: Core Engine – Prefill + KV-cache Decode
# ============================================================

class KVCacheBatchEngine:
    """
    Batch inference engine with:
      - Attention masking for padding
      - KV-cache reuse (prefill once, autoregressive decode)
      - Manual timing for prefill & decode
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def prefill(self, prompts: List[str]) -> Tuple[Dict[str, Any], float]:
        """
        Prefill stage:
          - Tokenize with padding (uses attention_mask)
          - Single forward pass with use_cache=True
          - Returns:
              * state dict (past_key_values, last_input_ids, attention_mask, input_lengths)
              * prefill time in seconds
        """
        t0 = time.time()

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = enc["input_ids"]          # [B, T]
        attention_mask = enc["attention_mask"]  # [B, T]

        # 每個序列實際長度（不含 padding），用來切割 generated tokens
        input_lengths = attention_mask.sum(dim=-1)  # [B]

        # KV-cache prefill
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        past_key_values = out.past_key_values  # KV cache

        # 最後一個非 pad token 作為 decode 起點
        # 因為這裡是右 padding，所以最後一個 token 就是最後一個非 pad token
        last_input_ids = input_ids.gather(
            1,
            (input_lengths - 1).unsqueeze(-1)
        )  # [B, 1]

        t1 = time.time()

        state = {
            "past_key_values": past_key_values,
            "last_input_ids": last_input_ids,
            "attention_mask": attention_mask,
            "input_lengths": input_lengths,
        }
        return state, (t1 - t0)

    @torch.no_grad()
    def decode(
        self,
        state: Dict[str, Any],
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[List[torch.Tensor], float, float]:
        """
        Decode stage with KV-cache reuse:
          - step-by-step autoregressive decoding
          - uses last_input_ids + past_key_values
          - returns:
              * list of generated token id tensors (length B)
              * decode_time_sec
              * ttft_sec (time until first decode step finishes)
        """
        past_key_values = state["past_key_values"]
        last_input_ids = state["last_input_ids"]

        batch_size = last_input_ids.size(0)
        generated = [list() for _ in range(batch_size)]

        t_decode_start = time.time()
        ttft_time = None

        for step in range(max_new_tokens):
            out = self.model(
                input_ids=last_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :]  # [B, vocab]
            past_key_values = out.past_key_values

            next_ids = sample_next_token(
                logits, temperature=temperature, top_p=top_p
            )  # [B, 1]

            # 記錄第一個 decode step 的時間當作 TTFT
            if ttft_time is None:
                ttft_time = time.time()

            # 累積每個 request 的 generated tokens
            for i in range(batch_size):
                generated[i].append(next_ids[i, 0].unsqueeze(0))

            last_input_ids = next_ids

        t_decode_end = time.time()

        # 把 list-of-tokens 轉為 tensor
        gen_tensors = [
            torch.stack(tok_list, dim=0) if len(tok_list) > 0
            else torch.empty(0, dtype=torch.long)
            for tok_list in generated
        ]

        decode_time = t_decode_end - t_decode_start
        if ttft_time is None:
            ttft_sec = decode_time
        else:
            ttft_sec = ttft_time - t_decode_start

        return gen_tensors, decode_time, ttft_sec


# ============================================================
# PART 4: Static Single-Request Baseline
# ============================================================

def run_static_baseline(
    engine: KVCacheBatchEngine,
    requests: List[InferenceRequest]
) -> List[RequestMetrics]:
    """
    Baseline: 每個 request 單獨跑 (batch_size = 1)
      - Prefill + Decode 都是一個一個來
      - 用 KV-cache decode, 但沒有 batch 併發
    """
    metrics: List[RequestMetrics] = []
    print("\n=== Running static single-request baseline ===")

    for req in requests:
        prompts = [req.prompt]
        state, prefill_sec = engine.prefill(prompts)
        gen_tensors, decode_sec, ttft_sec = engine.decode(
            state,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )

        gen_ids = gen_tensors[0]
        total_tokens = int(gen_ids.numel())
        total_sec = prefill_sec + decode_sec

        m = RequestMetrics(
            request_id=req.request_id,
            mode="static_single",
            batch_size=1,
            total_tokens=total_tokens,
            ttft_ms=ttft_sec * 1000.0,
            latency_ms=total_sec * 1000.0,
            prefill_ms=prefill_sec * 1000.0,
            decode_ms=decode_sec * 1000.0,
        )
        metrics.append(m)

        print(
            f"[static] req={req.request_id} "
            f"tokens={total_tokens} "
            f"TTFT={m.ttft_ms:.1f}ms "
            f"latency={m.latency_ms:.1f}ms"
        )

    return metrics


# ============================================================
# PART 5: Dynamic Batch Inference (研究用 Dynamic Engine)
# ============================================================

def run_dynamic_batch(
    engine: KVCacheBatchEngine,
    requests: List[InferenceRequest],
    batch_size: int
) -> List[RequestMetrics]:
    """
    Dynamic batch inference:
      - 將 requests 切成「變動大小」的 batch（最後一個 batch 可能不足）
      - 每個 batch 做一次 prefill + decode（真正 batch prefill/decode）
      - KV-cache 在 batch decode 中被重複使用
      - Attention mask 用於 prefill，避免 padding 參與 attention
    """
    print(f"\n=== Running dynamic batch inference (batch_size={batch_size}) ===")

    metrics: List[RequestMetrics] = []
    n = len(requests)

    # 模擬一個簡單的「scheduler」：依序把 requests 塞滿 batch
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_reqs = requests[start:end]
        prompts = [r.prompt for r in batch_reqs]
        max_new_tokens = max(r.max_new_tokens for r in batch_reqs)
        temperature = batch_reqs[0].temperature
        top_p = batch_reqs[0].top_p

        # Prefill 一次
        state, prefill_sec = engine.prefill(prompts)
        # Batch decode 多個序列
        gen_tensors, decode_sec, ttft_sec = engine.decode(
            state,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        total_sec = prefill_sec + decode_sec

        # 每個 request 自己的 token 數、latency
        for i, req in enumerate(batch_reqs):
            gen_ids = gen_tensors[i]
            total_tokens = int(gen_ids.numel())
            # 此處假設同一個 batch 中每個 request latency 相同（prefill+decode 一起完成）
            m = RequestMetrics(
                request_id=req.request_id,
                mode="dynamic_batch",
                batch_size=len(batch_reqs),
                total_tokens=total_tokens,
                ttft_ms=ttft_sec * 1000.0,
                latency_ms=total_sec * 1000.0,
                prefill_ms=prefill_sec * 1000.0,
                decode_ms=decode_sec * 1000.0,
            )
            metrics.append(m)

            print(
                f"[dynamic bs={len(batch_reqs)}] req={req.request_id} "
                f"tokens={total_tokens} "
                f"TTFT={m.ttft_ms:.1f}ms "
                f"latency={m.latency_ms:.1f}ms"
            )

    return metrics


# ============================================================
# PART 6: Benchmark – Throughput / Latency / TTFT / P95
# ============================================================

def summarize_metrics(metrics: List[RequestMetrics], label: str) -> Dict[str, Any]:
    """
    對一組 metrics 做統計：
      - total_tokens
      - total_time
      - throughput tokens/sec
      - TTFT P50 / P95
      - latency P50 / P95
    """
    if not metrics:
        return {}

    ttft_list = np.array([m.ttft_ms for m in metrics], dtype=float)
    lat_list = np.array([m.latency_ms for m in metrics], dtype=float)
    total_tokens = int(sum(m.total_tokens for m in metrics))

    # 估算「總 wall time」：用 latency 最大值近似（所有 request 完成時間）
    total_time_ms = float(lat_list.max())
    total_time_sec = total_time_ms / 1000.0
    throughput_tok_per_s = total_tokens / total_time_sec if total_time_sec > 0 else 0.0

    summary = {
        "label": label,
        "num_requests": len(metrics),
        "total_tokens": total_tokens,
        "total_time_ms": total_time_ms,
        "throughput_tok_per_s": throughput_tok_per_s,
        "ttft_p50_ms": float(np.percentile(ttft_list, 50)),
        "ttft_p95_ms": float(np.percentile(ttft_list, 95)),
        "latency_p50_ms": float(np.percentile(lat_list, 50)),
        "latency_p95_ms": float(np.percentile(lat_list, 95)),
    }
    return summary


def run_full_benchmark():
    # 測試 prompt（和 PoC#1 類似，比較「需要思考」的題目）
    base_prompts = [
        "Explain the concept of quantum computing and its potential applications in detail.",
        "Describe the process of photosynthesis in plants and its importance for life on Earth.",
        "Compare classical machine learning algorithms with modern deep learning approaches.",
        "Explain how blockchain technology works and discuss its potential impact.",
        "Describe the human immune system's response to viral infections.",
        "Discuss renewable energy sources versus fossil fuels environmental impacts.",
        "Explain the theory of relativity and how it differs from Newtonian physics.",
        "Describe the water cycle in detail and how human activities affect this process.",
    ]

    # 建一批「模擬請求」
    num_requests = 32
    requests: List[InferenceRequest] = []
    for i in range(num_requests):
        prompt = random.choice(base_prompts)
        req = InferenceRequest(
            request_id=i,
            prompt=prompt,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
        )
        requests.append(req)

    engine = KVCacheBatchEngine(model, tokenizer, device=device)

    # 1) Static baseline (batch_size=1)
    static_metrics = run_static_baseline(engine, requests)
    static_summary = summarize_metrics(static_metrics, label="static_single")

    # 2) Dynamic batch modes
    dynamic_summaries = []
    all_metrics: List[RequestMetrics] = []
    all_metrics.extend(static_metrics)

    for bs in [2, 4, 8]:
        dyn_metrics = run_dynamic_batch(engine, requests, batch_size=bs)
        all_metrics.extend(dyn_metrics)
        summary = summarize_metrics(dyn_metrics, label=f"dynamic_bs={bs}")
        dynamic_summaries.append(summary)

    # 輸出總結表
    print("\n" + "=" * 70)
    print("📊 Benchmark Summary")
    print("=" * 70)

    df_rows = []
    df_rows.append(static_summary)
    df_rows.extend(dynamic_summaries)
    df = pd.DataFrame(df_rows)
    print(df.to_string(index=False))

    # 以 batch_size=8 的 dynamic_batch 為例，計算 speedup & latency reduction
    best_dyn = None
    for s in dynamic_summaries:
        if best_dyn is None or s["throughput_tok_per_s"] > best_dyn["throughput_tok_per_s"]:
            best_dyn = s

    if best_dyn is not None:
        static_tput = static_summary["throughput_tok_per_s"]
        dyn_tput = best_dyn["throughput_tok_per_s"]
        tput_speedup = dyn_tput / static_tput if static_tput > 0 else 0.0

        static_p95 = static_summary["latency_p95_ms"]
        dyn_p95 = best_dyn["latency_p95_ms"]
        lat_reduction = (static_p95 - dyn_p95) / static_p95 * 100.0 if static_p95 > 0 else 0.0

        print("\n🚀 Key Findings (example vs static_single)")
        print(f"  • Best dynamic mode: {best_dyn['label']}")
        print(f"  • Throughput speedup: {tput_speedup:.2f}x")
        print(f"  • P95 latency change: {lat_reduction:+.1f}%")

        # 這裡你可以把實際跑出的數值，對應到履歷上的「7.37x / -86%」敘述
        #（依照實驗結果更新即可）

    # 儲存 raw metrics 方便之後畫圖 / 分析
    raw_rows = []
    for m in all_metrics:
        raw_rows.append({
            "request_id": m.request_id,
            "mode": m.mode,
            "batch_size": m.batch_size,
            "total_tokens": m.total_tokens,
            "ttft_ms": m.ttft_ms,
            "latency_ms": m.latency_ms,
            "prefill_ms": m.prefill_ms,
            "decode_ms": m.decode_ms,
        })
    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv("poc2_dynamic_batch_metrics.csv", index=False)
    print("\n📁 Saved raw metrics to poc2_dynamic_batch_metrics.csv")

    print("\n🎉 PoC#2 Dynamic Batch Benchmark Completed!")


if __name__ == "__main__":
    run_full_benchmark()
