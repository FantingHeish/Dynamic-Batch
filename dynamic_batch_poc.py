# ============================================================
# Dynamic Batch Inference for LLM Throughput Optimization
# + GPU Utilization & Memory Profiling (NVML)
# ============================================================

import os
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

print("=" * 70)
print("PoC#2: Dynamic Batch Inference (with GPU Profiling)")
print("=" * 70)

# ------------------------------------------------------------
# Install dependencies if missing
# ------------------------------------------------------------
try:
    import torch
    import pandas as pd
    import numpy as np
    import pynvml
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    import subprocess, sys
    print("\n📦 Installing missing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "transformers>=4.44.0", "accelerate>=0.33.0",
        "torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "pandas", "numpy", "pynvml"])
    
    import torch
    import pandas as pd
    import numpy as np
    import pynvml
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# GPU Profiling Setup (NVML)
# ============================================================

gpu_enabled = False
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_enabled = True
    print("🔧 GPU Profiling Enabled")
except Exception as e:
    print("⚠️ GPU profiling disabled:", e)


def get_gpu_stats():
    """
    Return GPU metrics:
      - utilization (%)
      - memory used (GB)
      - total memory (GB)
    """
    if not gpu_enabled:
        return {"gpu_util": 0.0, "gpu_mem_used": 0.0, "gpu_mem_total": 0.0}

    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util": float(util.gpu),
        "gpu_mem_used": mem.used / 1e9,
        "gpu_mem_total": mem.total / 1e9,
    }


print("\n🔍 GPU Initial State:", get_gpu_stats())


# ============================================================
# Model Loading
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"\n✅ Device: {device}")
print(f"🚀 Loading model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, use_fast=True, trust_remote_code=True
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()

print("✅ Model Loaded Successfully\n")


# ============================================================
# Request & Metric Data Classes
# ============================================================

@dataclass
class InferenceRequest:
    request_id: int
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class RequestMetrics:
    request_id: int
    mode: str
    batch_size: int
    total_tokens: int
    ttft_ms: float
    latency_ms: float
    prefill_ms: float
    decode_ms: float
    gpu_util_avg: float
    gpu_util_max: float
    gpu_mem_used_max: float


# ============================================================
# Sampling Function
# ============================================================

@torch.no_grad()
def sample_next_token(logits, temperature=0.7, top_p=0.9):
    """Nucleus sampling for next-token generation."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)

    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1]
    mask[..., 0] = False
    sorted_probs = sorted_probs * (~mask)
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    next_sorted = torch.multinomial(sorted_probs, 1)
    return torch.gather(sorted_idx, -1, next_sorted)


# ============================================================
# KV-Cache Batch Engine
# ============================================================

class KVCacheBatchEngine:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def prefill(self, prompts):
        t0 = time.time()

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        input_lengths = attention_mask.sum(dim=-1)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = out.past_key_values

        last_ids = input_ids.gather(1, (input_lengths - 1).unsqueeze(-1))

        t1 = time.time()
        return {
            "past_key_values": past_key_values,
            "last_input_ids": last_ids,
            "input_lengths": input_lengths,
        }, (t1 - t0)

    @torch.no_grad()
    def decode(self, state, max_new_tokens, temperature=0.7, top_p=0.9):
        past = state["past_key_values"]
        last = state["last_input_ids"]
        B = last.size(0)

        generated = [[] for _ in range(B)]

        t_decode_start = time.time()
        ttft_sec = None

        gpu_util_list = []
        gpu_mem_list = []

        for _ in range(max_new_tokens):
            gpu_stats = get_gpu_stats()
            gpu_util_list.append(gpu_stats["gpu_util"])
            gpu_mem_list.append(gpu_stats["gpu_mem_used"])

            out = self.model(
                input_ids=last,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

            next_ids = sample_next_token(logits, temperature, top_p)

            if ttft_sec is None:
                ttft_sec = time.time()

            last = next_ids
            for i in range(B):
                generated[i].append(next_ids[i, 0].unsqueeze(0))

        t_decode_end = time.time()

        gen_tensors = [torch.stack(x) for x in generated]
        gpu_util_avg = float(sum(gpu_util_list) / len(gpu_util_list))
        gpu_util_max = float(max(gpu_util_list))
        gpu_mem_used_max = float(max(gpu_mem_list))

        return gen_tensors, (t_decode_end - t_decode_start), (ttft_sec - t_decode_start), gpu_util_avg, gpu_util_max, gpu_mem_used_max


# ============================================================
# STATIC BASELINE
# ============================================================

def run_static_baseline(engine, requests):
    results = []
    print("\n=== Running Static Baseline ===")

    for req in requests:
        prompts = [req.prompt]

        state, prefill_sec = engine.prefill(prompts)
        gen, decode_sec, ttft_sec, util_avg, util_max, mem_max = engine.decode(
            state, req.max_new_tokens, req.temperature, req.top_p
        )

        total_tokens = int(gen[0].numel())
        total_sec = prefill_sec + decode_sec

        m = RequestMetrics(
            request_id=req.request_id,
            mode="static_single",
            batch_size=1,
            total_tokens=total_tokens,
            ttft_ms=ttft_sec * 1000,
            latency_ms=total_sec * 1000,
            prefill_ms=prefill_sec * 1000,
            decode_ms=decode_sec * 1000,
            gpu_util_avg=util_avg,
            gpu_util_max=util_max,
            gpu_mem_used_max=mem_max,
        )
        results.append(m)

    return results


# ============================================================
# DYNAMIC BATCH INFERENCE
# ============================================================

def run_dynamic_batch(engine, requests, batch_size):
    print(f"\n=== Running Dynamic Batch (bs={batch_size}) ===")
    results = []
    n = len(requests)

    for i in range(0, n, batch_size):
        batch = requests[i : i + batch_size]
        prompts = [x.prompt for x in batch]
        max_new = max(x.max_new_tokens for x in batch)

        state, prefill_sec = engine.prefill(prompts)
        gen, decode_sec, ttft_sec, util_avg, util_max, mem_max = engine.decode(
            state, max_new
        )

        total_sec = prefill_sec + decode_sec

        for j, req in enumerate(batch):
            total_tokens = int(gen[j].numel())

            m = RequestMetrics(
                request_id=req.request_id,
                mode="dynamic_batch",
                batch_size=len(batch),
                total_tokens=total_tokens,
                ttft_ms=ttft_sec * 1000,
                latency_ms=total_sec * 1000,
                prefill_ms=prefill_sec * 1000,
                decode_ms=decode_sec * 1000,
                gpu_util_avg=util_avg,
                gpu_util_max=util_max,
                gpu_mem_used_max=mem_max,
            )
            results.append(m)

    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize(label, metrics):
    if len(metrics) == 0:
        return None

    ttfts = np.array([m.ttft_ms for m in metrics])
    lats = np.array([m.latency_ms for m in metrics])
    utils = np.array([m.gpu_util_avg for m in metrics])
    util_maxs = np.array([m.gpu_util_max for m in metrics])
    mems = np.array([m.gpu_mem_used_max for m in metrics])
    total_tokens = int(sum(m.total_tokens for m in metrics))

    total_time_ms = float(lats.max())
    throughput = total_tokens / (total_time_ms / 1000)

    return {
        "label": label,
        "num_requests": len(metrics),
        "total_tokens": total_tokens,
        "total_time_ms": total_time_ms,
        "throughput_tok_per_s": throughput,
        "ttft_p50": float(np.percentile(ttfts, 50)),
        "ttft_p95": float(np.percentile(ttfts, 95)),
        "latency_p50": float(np.percentile(lats, 50)),
        "latency_p95": float(np.percentile(lats, 95)),
        "gpu_util_avg": float(utils.mean()),
        "gpu_util_max": float(util_maxs.max()),
        "gpu_mem_used_max": float(mems.max()),
    }


# ============================================================
# BENCHMARK
# ============================================================

def run_full_benchmark():
    prompts = [
        "Explain quantum computing.",
        "Describe the process of photosynthesis.",
        "Compare classical ML and deep learning.",
        "Explain how blockchain works.",
        "Describe the immune response to viruses.",
        "Discuss renewable energy vs fossil fuels.",
        "Explain relativity vs Newtonian physics.",
        "Describe the water cycle.",
    ]

    requests = [
        InferenceRequest(i, random.choice(prompts)) for i in range(32)
    ]

    engine = KVCacheBatchEngine(model, tokenizer, device)

    # baseline
    static = run_static_baseline(engine, requests)
    static_summary = summarize("static_single", static)

    # dynamic
    dyn_summaries = []
    for bs in [2, 4, 8]:
        dyn = run_dynamic_batch(engine, requests, bs)
        dyn_summaries.append(summarize(f"dynamic_bs={bs}", dyn))

    # print summary
    print("\n" + "=" * 70)
    print("📊 Benchmark Summary")
    print("=" * 70)

    rows = [static_summary] + dyn_summaries
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # best throughput
    best = max(rows, key=lambda r: r["throughput_tok_per_s"])
    print("\n🚀 Best Mode:", best["label"])
    print(f"   • Throughput: {best['throughput_tok_per_s']:.2f} tok/s")
    print(f"   • GPU Util Avg: {best['gpu_util_avg']:.1f}%")
    print(f"   • Latency P95: {best['latency_p95']:.1f} ms")


if __name__ == "__main__":
    run_full_benchmark()
    print("\n🎉 Dynamic Batch PoC Completed!")
