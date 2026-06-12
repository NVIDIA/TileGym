# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from tilegym_hf_bench.forward import NaiveForwardWrapper
from tilegym_hf_bench.hf_shim import load_model_with_cache
from tilegym_hf_bench.hf_shim import load_tokenizer_with_cache
from tilegym_hf_bench.profiling.nsys import NsysKernelCoverageReporter
from tilegym_hf_bench.profiling.torch_profiler import run_torch_profiler
from tilegym_hf_bench.tilegym_patch import apply_tilegym_patch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument("--use_tilegym", action="store_true", help="Use tilegym kernel")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Model ID to load")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument(
        "--input_text",
        type=str,
        default="What is the capital of France?",
        help="Input text for generation",
    )
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for averaging performance")
    parser.add_argument("--warmup_runs", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--show_outputs", action="store_true", help="Show full model outputs")
    parser.add_argument("--summary_file", type=str, default=None, help="File to append summary lines")
    parser.add_argument("--use_attn", action="store_true", help="Use attention")
    parser.add_argument("--use_cutile", action="store_true", help="Use cutile")
    parser.add_argument("--profile", action="store_true", help="Profile the model")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/logs",
        help="Directory to save profiler logs (default: /logs)",
    )
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision")
    parser.add_argument("--sentence_file", type=str, default=None)
    parser.add_argument("--output_length", type=int, default=100, help="Output length")
    parser.add_argument("--mock_input_len", type=int, default=0, help="Mock input length")
    parser.add_argument(
        "--report_kernel_coverage",
        action="store_true",
        help="Run under nsys profiler and report cuTile kernel coverage (GPU time and launch count ratios)",
    )
    return parser.parse_args(argv)


def get_messages_list(args):
    messages_list = []
    if args.mock_input_len > 0:
        line = " Hello" * (args.mock_input_len - 1)
        for _ in range(args.batch_size):
            messages_list.append(line)
    elif args.sentence_file is not None:
        with open(args.sentence_file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for _ in range(args.batch_size):
                messages_list.append("\n".join(lines))
    else:
        for _ in range(args.batch_size):
            messages_list.append(args.input_text)
        print(messages_list)
    return messages_list


def _detect_backend(args):
    backend = "base"
    if args.use_tilegym:
        if args.use_cutile:
            backend = "cutile"
    return backend


def _model_name_for_case(model_id):
    model_name = Path(model_id.rstrip("/\\")).name
    if model_name:
        return model_name
    return model_id.replace("/", "_")


def _build_case_id(args, backend):
    case_id = _model_name_for_case(args.model_id)
    if args.use_tilegym:
        if args.use_cutile:
            case_id += "_cutile"
        if args.use_attn:
            case_id += "_attn"
    else:
        case_id += "_naive"
    case_id += f"_{args.precision}"
    return case_id


def _build_summary_line(case_id, avg_tokens_per_sec, avg_time):
    return f"{case_id:<40} | {avg_tokens_per_sec:>10.2f} | {avg_time:>9.4f}"


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)

    if args.report_kernel_coverage:
        NsysKernelCoverageReporter(args, raw_argv).run()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(
        f"Benchmark settings: model={args.model_id}, batch_size={args.batch_size}, output_length={args.output_length}"
    )

    print(f"Loading model {args.model_id}...")
    tokenizer_kwargs = {}
    if "qwen3.5" in args.model_id.lower() or "qwen3_5" in args.model_id.lower():
        tokenizer_kwargs["use_fast"] = True
    tokenizer = load_tokenizer_with_cache(args.model_id, **tokenizer_kwargs)

    backend = _detect_backend(args)
    if args.use_tilegym:
        print("########################")
        print("#######Use TileGym#########")
        print("########################")
        apply_tilegym_patch(args.model_id, args.use_attn, use_cutile=(backend == "cutile"))

    model_kwargs = {
        "trust_remote_code": False,
        "device_map": "cuda",
        "torch_dtype": torch.bfloat16 if args.precision == "bfloat16" else torch.float32,
    }

    model = load_model_with_cache(args.model_id, **model_kwargs)

    if args.show_outputs or args.profile:
        args.warmup_runs = 1
        if args.show_outputs:
            args.num_runs = 1
        do_sample = False
    else:
        do_sample = True

    forward_wrapper = NaiveForwardWrapper(
        model,
        tokenizer=tokenizer,
        messages_list=get_messages_list(args),
        args=args,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        early_stopping=False,
    )

    print("Performing warmup runs...")
    for i in range(args.warmup_runs):
        with torch.no_grad():
            _ = forward_wrapper.forward()
        print(f"  Warmup run {i + 1}/{args.warmup_runs} completed")

    print(f"\nRunning benchmark with {args.num_runs} iterations...")
    generation_times = []
    tokens_per_second = []
    torch.cuda.synchronize()
    for run in range(args.num_runs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        total_tokens = 0
        outputs_list = []
        input_length = forward_wrapper.get_input_seq_len()

        start_event.record()
        with torch.no_grad():
            outputs = forward_wrapper.forward()
        end_event.record()

        outputs = forward_wrapper.post_process(outputs)
        outputs_list.append(outputs)
        generated_tokens = outputs.shape[1] - input_length
        print(f"generated_tokens: {generated_tokens}")
        total_tokens += generated_tokens * args.batch_size

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        generation_times.append(elapsed_time)
        tokens_per_second.append(total_tokens / elapsed_time)

        print(
            f"Run {run + 1}: Generated {generated_tokens} tokens in {elapsed_time:.4f}s ({generated_tokens / elapsed_time:.2f} tokens/sec)"
        )

    avg_time = np.mean(generation_times)
    avg_tokens_per_sec = np.mean(tokens_per_second)
    std_tokens_per_sec = np.std(tokens_per_second)

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Use TileGym: {args.use_tilegym}")
    print(f"Backend: {backend}")
    print(f"Use attention: {args.use_attn}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output length: {args.output_length}")
    print(f"Input length: {input_length}")
    print(f"Average generation time: {avg_time:.4f}s")
    print(f"Average throughput: {avg_tokens_per_sec:.2f} ± {std_tokens_per_sec:.2f} tokens/sec")

    if args.show_outputs:
        print("\n===== GENERATED OUTPUTS =====")
        for batch_idx, outputs in enumerate(outputs_list):
            for i in range(outputs.shape[0]):
                decoded_output = tokenizer.decode(outputs[i][input_length:], skip_special_tokens=True)
                print(f"\nBatch {batch_idx + 1}, Output {i + 1}:")
                print(decoded_output)
                print("-" * 50)

    case_id = _build_case_id(args, backend)
    summary_line = _build_summary_line(case_id, avg_tokens_per_sec, avg_time)

    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(summary_line + "\n")
        print(f"Summary written to {args.summary_file}: {summary_line}")

    if args.profile:
        run_torch_profiler(forward_wrapper, args, case_id, avg_time, summary_line)


if __name__ == "__main__":
    main()
