#!/usr/bin/env python3
"""TICKET-06 validation harness. Run on the 3090 box (sauron) against a
RUNNING Inkling-2-Mini server (the TICKET-05 reference command, TP2/EP2).

Usage:
  python t6_validate.py --port 8000 --model /path/to/Inkling-2-Mini \
      [--gsm8k-data test.jsonl] [--only gsm8k,determinism,prefix,swa,perf]
      [--det-tokens 2048]

Gates (ticket 1):
  gsm8k       repo run_eval, 200 samples greedy; PASS >= 92% (published ~94%)
  determinism same prompt twice, greedy; token-for-token text equality
  prefix      same ~4 k prefix twice; identical output + cached_tokens > 0
  swa         codeword needle at +~450 tokens (in the 1024 window) and
              +~2100 tokens (beyond it, full-attention layers must carry it)
Perf (ticket 2): repo benchmark.serving, decode bs1 + 8 k prefill.
Gate 2 (TP1 vs TP2) needs a server restart with --tp-size 1; the report
includes the exact commands. Writes t6_report.md + t6_gsm8k.log.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODEL_DEFAULT = "thinkingmachines/Inkling-2-Mini"
GSM8K_PASS = 0.92
GSM8K_WARN = 0.90


def api(base, path, payload, timeout=1800):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def completions(base, model, prompt, max_tokens, temperature=0.0):
    return api(
        base,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt if isinstance(prompt, list) else [prompt],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": False,
        },
    )


def gen_text(resp):
    return resp["choices"][0]["text"]


def padded_sentence(tokens):
    """Deterministic filler of roughly `tokens` tokens."""
    unit = (
        "The quarterly report for the manufacturing division described a steady "
        "improvement in unit throughput across all three production lines. "
    )
    per = 28  # ~28 tokens per unit (measured loosely; close enough for a filler)
    return (unit * max(1, int(tokens / per))).strip()


def check_gsm8k(base, model, args, py):
    cmd = [
        py,
        "-m",
        "sglang.test.run_eval",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--eval-name",
        "gsm8k",
        "--num-examples",
        "200",
    ]
    if args.gsm8k_data:
        cmd += ["--gsm8k-data-path", args.gsm8k_data]
    print(f"== gsm8k: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=14 * 3600
    )
    with open("t6_gsm8k.log", "w") as f:
        f.write(proc.stdout + "\n\nSTDERR:\n" + proc.stderr)
    dt = time.time() - t0
    acc = None
    m = re.search(r"accuracy\D*([01]\.\d+)", proc.stdout, re.IGNORECASE)
    if m:
        acc = float(m.group(1))
    status = (
        "PASS"
        if acc is not None and acc >= GSM8K_PASS
        else "WARN"
        if acc is not None and acc >= GSM8K_WARN
        else "FAIL"
    )
    print(f"   accuracy={acc} ({dt / 60:.1f} min) [{status}]")
    return (
        f"gsm8k (200 samples, greedy): **{acc if acc is not None else 'n/a'}** "
        f"[{status}] — target ≥ {GSM8K_PASS:.0%} (published ≈ 94%)"
    )


def check_determinism(base, model, args):
    prompt = [
        (
            "Solve the following step by step. A relay race team of four runners "
            "finishes in 3 minutes and 42 seconds; the first runner is 6 seconds "
            "faster than the second, the second 4 seconds faster than the third, "
            "and the third 2 seconds faster than the fourth. How fast was each "
            "runner?\n"
        )
    ]
    t0 = time.time()
    r1 = completions(base, model, prompt, args.det_tokens)
    t1 = time.time()
    r2 = completions(base, model, prompt, args.det_tokens)
    t2 = time.time()
    t1s, t2s = t1 - t0, t2 - t1
    same = gen_text(r1) == gen_text(r2)
    toks = r1.get("choices", [{}])[0].get("tokens")
    if toks and r2.get("choices", [{}])[0].get("tokens"):
        same = toks == r2["choices"][0]["tokens"]
    status = "PASS" if same else "FAIL"
    print(
        f"   deterministic {args.det_tokens}-token greedy: {same} [{status}] "
        f"({t1s:.1f}s / {t2s:.1f}s)"
    )
    return (
        f"determinism ({args.det_tokens} tokens, greedy, 2 runs): **{'identical' if same else 'DIVERGED'}** "
        f"[{status}]"
    )


def check_prefix(base, model, args):
    prefix = padded_sentence(4096)
    question = (
        "\n\nQuestion: what number of production lines is mentioned? Answer briefly."
    )
    full = prefix + question
    t0 = time.time()
    r1 = completions(base, model, [full], 64)
    t1 = time.time()
    r2 = completions(base, model, [full], 64)
    t2 = time.time()
    same = gen_text(r1) == gen_text(r2)
    cached = None
    usage = r2.get("usage") or {}
    details = (
        usage.get("prompt_tokens_details") or r2.get("prompt_tokens_details") or {}
    )
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
    status = "PASS" if (same and (cached or 0) > 0) else ("WARN" if same else "FAIL")
    print(
        f"   identical={same} cached_tokens(second)={cached} [{status}] "
        f"({t1 - t0:.1f}s cold, {t2 - t1:.1f}s cached)"
    )
    return (
        f"prefix-cache (≈4 k shared prefix, twice): identical output = **{same}**, "
        f"cached_tokens on 2nd request = **{cached}** [{status}]"
    )


def check_swa(base, model, args):
    fact = "The magic codeword is ZK-7391. Remember it well."
    question = (
        "\n\nEarlier in this conversation a magic codeword was stated. "
        "What is the magic codeword? Answer with only the codeword."
    )
    results = {}
    for label, pad in (("in-window (+~450)", 450), ("out-of-window (+~2100)", 2100)):
        prompt = [fact + "\n" + padded_sentence(pad) + question]
        resp = completions(base, model, prompt, 32)
        ok = "ZK-7391" in gen_text(resp)
        results[label] = ok
        print(f"   swa {label}: {'PASS' if ok else 'FAIL'} ({gen_text(resp)[:40]!r})")
    both = all(results.values())
    return (
        "SWA needle (codeword, 1024-token window): "
        + " ".join(f"{k} = **{('ok' if v else 'MISS')}**" for k, v in results.items())
        + f" [{'PASS' if both else 'FAIL'}]"
    )


def perf_scenario(base, model, args, py, name, *extra):
    cmd = [
        py,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        model,
        "--tokenizer",
        model,
        "--dataset-name",
        "random",
        "--request-rate",
        "inf",
        "--temperature",
        "0",
        "--disable-tqdm",
        *extra,
    ]
    print(f"== perf[{name}]: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, timeout=2 * 3600
    )
    out = proc.stdout
    in_tok = re.search(r"Input token throughput[^0-9]*([0-9.,]+)", out)
    out_tok = re.search(r"Output token throughput[^0-9]*([0-9.,]+)", out)
    ttft = re.search(r"Mean TTFT[^0-9]*([0-9.]+)", out)
    itl = re.search(r"Mean ITL[^0-9]*([0-9.]+)", out)
    vals = {
        "in": float(in_tok.group(1).replace(",", "")) if in_tok else None,
        "out": float(out_tok.group(1).replace(",", "")) if out_tok else None,
        "ttft": float(ttft.group(1)) if ttft else None,
        "itl": float(itl.group(1)) if itl else None,
    }
    print(
        f"   in={vals['in']} out={vals['out']} tok/s, TTFT={vals['ttft']}s, ITL={vals['itl']}ms"
    )
    if proc.returncode != 0:
        with open(f"t6_perf_{name}.log", "w") as f:
            f.write(out + "\n\nSTDERR:\n" + proc.stderr)
    return name, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--gsm8k-data", default=None)
    ap.add_argument("--only", default="gsm8k,determinism,prefix,swa,perf")
    ap.add_argument("--det-tokens", type=int, default=2048)
    args = ap.parse_args()
    py = sys.executable
    base = f"http://{args.host}:{args.port}"
    sections = [s.strip() for s in args.only.split(",") if s.strip()]

    probe = urllib.request.Request(base + "/v1/models")
    try:
        with urllib.request.urlopen(probe, timeout=10) as resp:
            print(f"server up: {resp.status} {resp.url}")
    except (urllib.error.URLError, ConnectionError) as e:
        print(f"server unreachable at {base}: {e}")
        return 2

    report = ["# T6 validation report — Inkling-2-Mini on 2x3090 (sauron)", ""]
    gates = {}
    for name in sections:
        t0 = time.time()
        if name == "gsm8k":
            gates[name] = check_gsm8k(base, args.model, args, py)
        elif name == "determinism":
            gates[name] = check_determinism(base, args.model, args)
        elif name == "prefix":
            gates[name] = check_prefix(base, args.model, args)
        elif name == "swa":
            gates[name] = check_swa(base, args.model, args)
        elif name == "perf":
            rows = [
                perf_scenario(
                    base,
                    args.model,
                    args,
                    py,
                    "decode-bs1",
                    "--num-prompts",
                    "8",
                    "--random-input",
                    "128",
                    "--random-output",
                    "256",
                    "--max-concurrency",
                    "1",
                ),
                perf_scenario(
                    base,
                    args.model,
                    args,
                    py,
                    "prefill-8k",
                    "--num-prompts",
                    "16",
                    "--random-input",
                    "8192",
                    "--random-output",
                    "64",
                    "--max-concurrency",
                    "8",
                ),
            ]
            lines = [
                "| scenario | in tok/s | out tok/s | TTFT s | ITL ms |",
                "|---|---|---|---|---|",
            ]
            for n, v in rows:
                lines.append(
                    f"| {n} | {v['in']} | {v['out']} | {v['ttft']} | {v['itl']} |"
                )
            gates[name] = "\n".join(lines)
            print(gates[name])
        else:
            print(f"unknown section {name!r}; skipping")
        print(f"({name}: {time.time() - t0:.0f}s)\n")
        report.append(f"## {name}\n\n{gates[name]}\n")

    ok = all("FAIL" not in v for v in gates.values())
    report += [
        "## Not run here (manual steps)",
        "",
        (
            "1. **TP1 vs TP2** (gate 2): restart the server with `--tp-size 1` "
            "(drop the EP flags; 21.5 GB of weights may not fit 24 GB with KV — "
            "if OOM, this gate degrades to determinism + gsm8k per the ticket), "
            "regenerate the determinism prompt, and compare token-for-token with "
            "the TP2 run above: `python t6_validate.py --only determinism "
            "--port <tp1-port>`."
        ),
        (
            "2. **Memory watermark**: `nvidia-smi --query-gpu=memory.used "
            "--format=csv` at steady state, 128 k ctx (target < 21 GB/GPU)."
        ),
        (
            "3. **CUDA-graph follow-up**: the recipe runs with "
            "`--disable-cuda-graph`; re-run perf with graphs on to size the "
            "follow-up ticket."
        ),
        "",
        f"## Verdict\n\n{'ALL RUN GATES PASS' if ok else 'ONE OR MORE GATES FAILED'}",
    ]
    with open("t6_report.md", "w") as f:
        f.write("\n".join(report))
    print("wrote t6_report.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
