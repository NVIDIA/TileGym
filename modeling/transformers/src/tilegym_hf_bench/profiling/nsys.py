# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import datetime
import glob
import os
import shlex
import sqlite3
import subprocess
import sys
from collections import defaultdict

from tilegym_hf_bench.kernel_filters import KernelFilter


class NsysKernelCoverageReporter:
    """Runs nsys profiling and reports cuTile kernel coverage."""

    def __init__(self, args, argv=None):
        self.args = args
        self.argv = list(sys.argv[1:] if argv is None else argv)
        self.kernel_filter = KernelFilter()
        self.log_dir = args.log_dir
        self.model_name = args.model_id.split("/")[-1]

    def run(self):
        inner_args = self._build_inner_args()
        nsys_output_base = self._build_output_path()
        nsys_cmd = self._build_nsys_command(inner_args, nsys_output_base)

        print(f"Running nsys profile command:\n  {shlex.join(nsys_cmd)}\n")

        env = os.environ.copy()
        env["TMPDIR"] = self.log_dir
        proc = subprocess.Popen(nsys_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()

        nsys_rep_path = self._find_nsys_report(nsys_output_base, proc.returncode)
        self._compute_and_report_ratio(nsys_rep_path)

    def _build_inner_args(self):
        inner_args = []
        skip_next = False
        for arg in self.argv:
            if skip_next:
                skip_next = False
                continue
            if arg == "--report_kernel_coverage":
                continue
            inner_args.append(arg)
        if "--profile" not in inner_args:
            inner_args.append("--profile")
        return inner_args

    def _build_output_path(self):
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"nsys_{self.model_name}_{timestamp}")

    @staticmethod
    def _build_nsys_command(inner_args, output_base):
        return [
            "nsys",
            "profile",
            "-c",
            "cudaProfilerApi",
            "--capture-range-end=stop-shutdown",
            "-o",
            output_base,
            "--trace=cuda",
            "--force-overwrite=true",
            "--",
            sys.executable,
            "-m",
            "tilegym_hf_bench._cli",
        ] + inner_args

    @staticmethod
    def _find_nsys_report(output_base, returncode):
        pattern = f"{output_base}*.nsys-rep"
        matches = sorted(glob.glob(pattern))

        if returncode != 0 and not matches:
            print(f"\nnsys profile exited with code {returncode} and no report was generated.")
            sys.exit(returncode)
        if returncode != 0:
            print(f"\nWarning: nsys exited with code {returncode}, but report was generated. Proceeding.")

        if not matches:
            print(f"Error: No .nsys-rep file found matching {pattern}")
            sys.exit(1)

        nsys_rep_path = matches[-1]
        print(f"\nFound nsys report: {nsys_rep_path}")
        return nsys_rep_path

    @staticmethod
    def _resolve_sqlite_path(path):
        if path.endswith(".sqlite"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"SQLite file not found: {path}")
            return path

        if path.endswith(".nsys-rep"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"nsys-rep file not found: {path}")
            sibling = path.replace(".nsys-rep", ".sqlite")
            if os.path.isfile(sibling):
                return sibling
            output_path = sibling
            try:
                subprocess.run(
                    ["nsys", "export", "--type=sqlite", "-o", output_path, path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                raise RuntimeError("nsys CLI not found. Install NVIDIA Nsight Systems or provide a .sqlite file.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"nsys export failed: {e.stderr}")
            return output_path

        raise ValueError(f"Unsupported file type: {path} (expected .nsys-rep or .sqlite)")

    def _extract_kernel_durations(self, path):
        sqlite_path = self._resolve_sqlite_path(path)
        conn = sqlite3.connect(sqlite_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT k.start, k.end, s.value AS name
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                ORDER BY k.start ASC
                """
            )
            result = {}
            for idx, (start, end, name) in enumerate(cursor):
                duration_ns = float(end - start)
                result[(idx, name)] = duration_ns
            return result
        finally:
            conn.close()

    def _classify_kernels(self, durations):
        tilegym_by_name = defaultdict(float)
        tilegym_count_by_name = defaultdict(int)
        other_by_name = defaultdict(float)
        other_count_by_name = defaultdict(int)
        for (_idx, name), dur_ns in durations.items():
            if self.kernel_filter.contains(name):
                tilegym_by_name[name] += dur_ns
                tilegym_count_by_name[name] += 1
            else:
                other_by_name[name] += dur_ns
                other_count_by_name[name] += 1
        return tilegym_by_name, tilegym_count_by_name, other_by_name, other_count_by_name

    def _compute_and_report_ratio(self, nsys_rep_path):
        durations = self._extract_kernel_durations(nsys_rep_path)
        if not durations:
            print("No kernel durations found in the nsys report.")
            return

        tilegym_by_name, tilegym_count_by_name, other_by_name, other_count_by_name = self._classify_kernels(durations)

        tilegym_total_ns = sum(tilegym_by_name.values())
        other_total_ns = sum(other_by_name.values())
        all_total_ns = tilegym_total_ns + other_total_ns
        tilegym_total_count = sum(tilegym_count_by_name.values())
        other_total_count = sum(other_count_by_name.values())
        all_total_count = tilegym_total_count + other_total_count

        self._print_report(
            tilegym_by_name,
            tilegym_count_by_name,
            tilegym_total_ns,
            tilegym_total_count,
            all_total_ns,
            all_total_count,
        )

        tilegym_time_pct = (tilegym_total_ns / all_total_ns * 100) if all_total_ns > 0 else 0
        tilegym_count_pct = (tilegym_total_count / all_total_count * 100) if all_total_count > 0 else 0
        time_ratio_str = f"{tilegym_time_pct:.2f}%"
        count_ratio_str = f"{tilegym_count_pct:.2f}%"

        if self.args.summary_file:
            self._append_summary(time_ratio_str, count_ratio_str)

    @staticmethod
    def _print_report(
        tilegym_by_name,
        tilegym_count_by_name,
        tilegym_total_ns,
        tilegym_total_count,
        all_total_ns,
        all_total_count,
    ):
        print("\n===== NSYS KERNEL GPU TIME ANALYSIS =====\n")
        header_fmt = "{:<60} {:>8} {:>15} {:>12}"
        row_fmt = "{:<60} {:>8} {:>15.3f} {:>11.1f}%"
        sep = "-" * 60
        print(header_fmt.format("Kernel Name", "# Calls", "GPU Time (ms)", "% of Total"))
        print(f"{sep}    {'--------':>8}    {'-------------':>15}    {'----------':>10}")

        for name, dur_ns in sorted(tilegym_by_name.items(), key=lambda x: -x[1]):
            dur_ms = dur_ns / 1e6
            pct = (dur_ns / all_total_ns * 100) if all_total_ns > 0 else 0
            count = tilegym_count_by_name[name]
            print(row_fmt.format(name[:60], count, dur_ms, pct))

        print(f"{sep}    {'--------':>8}    {'-------------':>15}    {'----------':>10}")
        tilegym_ms = tilegym_total_ns / 1e6
        all_ms = all_total_ns / 1e6
        tilegym_time_pct = (tilegym_total_ns / all_total_ns * 100) if all_total_ns > 0 else 0
        tilegym_count_pct = (tilegym_total_count / all_total_count * 100) if all_total_count > 0 else 0
        print(row_fmt.format("TileGym Total", tilegym_total_count, tilegym_ms, tilegym_time_pct))
        print(row_fmt.format("All Kernels Total", all_total_count, all_ms, 100.0))

        time_ratio_str = f"{tilegym_time_pct:.2f}%"
        count_ratio_str = f"{tilegym_count_pct:.2f}%"
        print(f"\n>>> cuTile Kernel Coverage (GPU Time):    {time_ratio_str} <<<")
        print(f">>> cuTile Kernel Coverage (# Launches):  {count_ratio_str} <<<\n")

    def _append_summary(self, time_ratio_str, count_ratio_str):
        with open(self.args.summary_file, "a") as f:
            f.write(
                f"nsys_cutile_coverage | {self.model_name:<40} | time={time_ratio_str} | launches={count_ratio_str}\n"
            )
        print(f"Coverage ratio appended to {self.args.summary_file}")
