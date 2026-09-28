#!/usr/bin/env python3
"""
Multi-hop generative cascade pipeline for provenance-decay experiments.

Given a unified corpus of source records, this script applies a fixed
sequence of LLM transformations (summarize -> translate en->fr ->
translate fr->en -> paraphrase -> tweet-style rewrite) to each source
text, repeated for several stochastic samples per source, and writes
every hop's output as a JSONL record.

Usage
-----
Pilot run (recommended first):
    python cascade_pipeline.py run \
        --input unified_corpus.json \
        --output cascade_output.jsonl \
        --limit-sources 50 \
        --num-samples 3

Full run:
    python cascade_pipeline.py run \
        --input unified_corpus.json \
        --output cascade_output.jsonl \
        --num-samples 5

Validate afterward:
    python cascade_pipeline.py validate \
        --output cascade_output.jsonl \
        --input unified_corpus.json \
        --num-samples 5

Only open-source models are used. Default backend is vLLM if installed
(fast, good GPU utilization); otherwise falls back to plain
transformers. Install one of:

    pip install vllm                       # preferred on a GPU box
    pip install transformers accelerate    # fallback

Model default: Qwen/Qwen2.5-7B-Instruct (fully open, no gating).
Swap with --model, e.g. meta-llama/Meta-Llama-3-8B-Instruct
(requires an accepted HF license + `huggingface-cli login`) or
mistralai/Mistral-7B-Instruct-v0.3.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

# --------------------------------------------------------------------------
# Hop definitions
# --------------------------------------------------------------------------

HOP_SEQUENCE = [
    "summarization",
    "translation_fr",
    "translation_en",
    "paraphrasing",
    "style_tweet",
]

PROMPTS = {
    "summarization": (
        "Summarize the following text into 2-3 sentences, preserving the "
        "key factual claims. Output only the summary, with no preamble, "
        "labels, or extra commentary.\n\nText:\n{text}"
    ),
    "translation_fr": (
        "Translate the following English text into French. Output only "
        "the French translation, with no preamble or notes.\n\nText:\n{text}"
    ),
    "translation_en": (
        "Translate the following French text into English. Output only "
        "the English translation, with no preamble or notes.\n\nText:\n{text}"
    ),
    "paraphrasing": (
        "Paraphrase the following text using different wording while "
        "keeping the meaning intact. Output only the paraphrase, with no "
        "preamble or notes.\n\nText:\n{text}"
    ),
    "style_tweet": (
        "Rewrite the following text as a short, engaging social media "
        "tweet of at most 280 characters that retains the main claim. "
        "Output only the tweet text, with no hashtags explanation, "
        "preamble, or quotation marks.\n\nText:\n{text}"
    ),
}

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cascade")


# --------------------------------------------------------------------------
# Model backends
# --------------------------------------------------------------------------

class ModelBackend:
    """Common interface: chat(prompt, temperature, top_p, seed, max_new_tokens) -> str"""

    name = "base"

    def chat(self, prompt: str, temperature: float, top_p: float,
              seed: int, max_new_tokens: int) -> str:
        raise NotImplementedError


class VLLMBackend(ModelBackend):
    def __init__(self, model_name: str, dtype: str = "auto",
                 gpu_memory_utilization: float = 0.90):
        from vllm import LLM  # imported lazily so the script still loads without vllm
        self.name = f"vllm:{model_name}"
        self.model_name = model_name
        logger.info(f"Loading model with vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _format(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def chat(self, prompt: str, temperature: float, top_p: float,
              seed: int, max_new_tokens: int) -> str:
        from vllm import SamplingParams
        formatted = self._format(prompt)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
        )
        outputs = self.llm.generate([formatted], params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


class HFBackend(ModelBackend):
    def __init__(self, model_name: str, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{model_name}"
        self.model_name = model_name
        self.torch = torch
        logger.info(f"Loading model with transformers: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if dtype in ("auto", "bfloat16") else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def chat(self, prompt: str, temperature: float, top_p: float,
              seed: int, max_new_tokens: int) -> str:
        from transformers import set_seed
        set_seed(seed)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_backend(backend_choice: str, model_name: str) -> ModelBackend:
    if backend_choice == "vllm":
        return VLLMBackend(model_name)
    if backend_choice == "hf":
        return HFBackend(model_name)
    # auto: try vllm, fall back to hf
    try:
        return VLLMBackend(model_name)
    except Exception as e:
        logger.warning(f"vLLM unavailable ({e}); falling back to transformers backend.")
        return HFBackend(model_name)


# --------------------------------------------------------------------------
# Corpus / output I/O helpers
# --------------------------------------------------------------------------

def load_corpus(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # tolerate {"records": [...]} style wrappers
        for key in ("records", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError("Input JSON is a dict but no list of records was found inside it.")
    if not isinstance(data, list):
        raise ValueError("Expected the corpus JSON to be a list of records.")
    return data


def load_completed_cascades(output_path: str) -> Dict[str, set]:
    """Return {cascade_id: set(hop_index already written)} for resume support."""
    completed: Dict[str, set] = {}
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("cascade_id")
            hop_idx = rec.get("hop_index")
            if cid is None or hop_idx is None:
                continue
            completed.setdefault(cid, set()).add(hop_idx)
    return completed


# --------------------------------------------------------------------------
# Cascade generation
# --------------------------------------------------------------------------

@dataclass
class GenConfig:
    model_name: str
    temperature: float
    top_p: float
    max_new_tokens: int


def run_single_cascade(backend: ModelBackend, source: dict, sample_index: int,
                         cfg: GenConfig, base_seed: int) -> List[dict]:
    """Run all hops sequentially for one (source, sample_index) pair."""
    records = []
    current_text = source["text"]
    source_record_id = source.get("record_id")
    cascade_id = f"{source_record_id}_{sample_index}"

    for hop_i, hop_name in enumerate(HOP_SEQUENCE, start=1):
        seed = base_seed + sample_index * 1000 + hop_i  # deterministic, unique per (sample, hop)
        prompt = PROMPTS[hop_name].format(text=current_text)
        output_text = backend.chat(
            prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=seed,
            max_new_tokens=cfg.max_new_tokens,
        )

        record = {
            "cascade_id": cascade_id,
            "source_record_id": source_record_id,
            "hop_index": hop_i,
            "hop_name": hop_name,
            "sample_index": sample_index,
            "model": cfg.model_name,
            "input_text": current_text,
            "output_text": output_text,
            "provenance_signal": source.get("provenance_signal"),
            "credibility_label": source.get("credibility_label"),
            "origin": source.get("origin"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_new_tokens": cfg.max_new_tokens,
            "seed": seed,
        }
        records.append(record)
        current_text = output_text  # output of hop i feeds hop i+1

    return records


def run_pipeline(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.input)
    if args.limit_sources:
        corpus = corpus[: args.limit_sources]
    logger.info(f"Loaded {len(corpus)} source records from {args.input}")

    completed = load_completed_cascades(args.output)
    logger.info(f"Found {len(completed)} cascades already present in {args.output} (resume mode)")

    backend = build_backend(args.backend, args.model)
    cfg = GenConfig(
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    error_log_path = args.output + ".errors.log"
    n_hops = len(HOP_SEQUENCE)
    total_cascades = len(corpus) * args.num_samples
    done = 0
    t0 = time.time()

    with open(args.output, "a", encoding="utf-8") as out_f, \
         open(error_log_path, "a", encoding="utf-8") as err_f:

        for source in corpus:
            source_id = source.get("record_id")
            if not source_id or not source.get("text"):
                err_f.write(json.dumps({
                    "error": "missing record_id or text",
                    "record": source,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                continue

            for sample_index in range(args.num_samples):
                cascade_id = f"{source_id}_{sample_index}"
                already = completed.get(cascade_id, set())
                if len(already) == n_hops:
                    done += 1
                    continue  # fully done, skip (resume)

                try:
                    records = run_single_cascade(
                        backend, source, sample_index, cfg, base_seed=args.seed
                    )
                    for rec in records:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                except Exception as e:
                    logger.error(f"Failed cascade {cascade_id}: {e}")
                    err_f.write(json.dumps({
                        "error": str(e),
                        "cascade_id": cascade_id,
                        "source_record_id": source_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")
                    err_f.flush()

                done += 1
                if done % 10 == 0 or done == total_cascades:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {done}/{total_cascades} cascades "
                        f"({rate:.2f}/s, elapsed {elapsed/60:.1f} min)"
                    )

    logger.info(f"Done. Output written to {args.output}")
    logger.info(f"Errors (if any) logged to {error_log_path}")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_pipeline(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.input) if args.input else None
    if corpus is not None and args.limit_sources:
        corpus = corpus[: args.limit_sources]

    source_by_id = {}
    if corpus is not None:
        source_by_id = {s.get("record_id"): s for s in corpus}

    records: List[dict] = []
    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n_hops = len(HOP_SEQUENCE)
    print(f"Total hop records: {len(records)}")

    if corpus is not None:
        expected = len(corpus) * args.num_samples * n_hops
        print(f"Expected (num_sources x num_samples x num_hops): "
              f"{len(corpus)} x {args.num_samples} x {n_hops} = {expected}")
        status = "OK" if len(records) == expected else "MISMATCH"
        print(f"  -> {status}")

    # Empty / very short outputs
    short_threshold = 5
    short_outputs = [
        r for r in records
        if len(r.get("output_text", "").split()) < short_threshold
    ]
    print(f"\nOutputs with < {short_threshold} words: {len(short_outputs)}")
    for r in short_outputs[:10]:
        print(f"  - {r.get('cascade_id')} hop {r.get('hop_index')} "
              f"({r.get('hop_name')}): {r.get('output_text')!r}")
    if len(short_outputs) > 10:
        print(f"  ... and {len(short_outputs) - 10} more")

    # Median word count per hop
    print("\nMedian word count per hop:")
    by_hop: Dict[str, List[int]] = {}
    for r in records:
        by_hop.setdefault(r.get("hop_name", "unknown"), []).append(
            len(r.get("output_text", "").split())
        )
    for hop_name in HOP_SEQUENCE:
        counts = by_hop.get(hop_name, [])
        if counts:
            med = statistics.median(counts)
            print(f"  {hop_name:<16} n={len(counts):<6} median_words={med}")
        else:
            print(f"  {hop_name:<16} n=0 (no records)")

    # Provenance / credibility propagation
    if corpus is not None:
        mismatches = []
        for r in records:
            src = source_by_id.get(r.get("source_record_id"))
            if src is None:
                mismatches.append((r.get("cascade_id"), r.get("hop_index"), "source not found"))
                continue
            if r.get("provenance_signal") != src.get("provenance_signal"):
                mismatches.append((r.get("cascade_id"), r.get("hop_index"), "provenance_signal mismatch"))
            if r.get("credibility_label") != src.get("credibility_label"):
                mismatches.append((r.get("cascade_id"), r.get("hop_index"), "credibility_label mismatch"))
            if r.get("origin") != src.get("origin"):
                mismatches.append((r.get("cascade_id"), r.get("hop_index"), "origin mismatch"))
        print(f"\nProvenance/credibility/origin propagation mismatches: {len(mismatches)}")
        for m in mismatches[:10]:
            print(f"  - {m}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("\n(Skipping provenance propagation check: no --input corpus given)")

    # Duplicate cascade_id + hop_index
    seen = set()
    dupes = 0
    for r in records:
        key = (r.get("cascade_id"), r.get("hop_index"))
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    print(f"\nDuplicate (cascade_id, hop_index) pairs: {dupes}")
    print("  -> OK" if dupes == 0 else "  -> FOUND DUPLICATES")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Generate the multi-hop cascade.")
    run_p.add_argument("--input", required=True, help="Path to unified_corpus.json")
    run_p.add_argument("--output", required=True, help="Path to output JSONL file (appended to / resumed)")
    run_p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name/path")
    run_p.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    run_p.add_argument("--num-samples", type=int, default=3, help="Stochastic samples per source (>=3)")
    run_p.add_argument("--limit-sources", type=int, default=None, help="For pilot runs, e.g. 20-50")
    run_p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    run_p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run_p.add_argument("--seed", type=int, default=42, help="Base seed; combined with sample/hop indices")
    run_p.set_defaults(func=run_pipeline)

    val_p = sub.add_parser("validate", help="Run post-hoc quality checks on the cascade output.")
    val_p.add_argument("--output", required=True, help="Path to the JSONL cascade file to validate")
    val_p.add_argument("--input", default=None, help="Path to unified_corpus.json (for propagation checks)")
    val_p.add_argument("--num-samples", type=int, default=3)
    val_p.add_argument("--limit-sources", type=int, default=None)
    val_p.set_defaults(func=validate_pipeline)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
