# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tilegym.ops.tilecpp.utils import _cuda_utils


@pytest.fixture
def compiler_inputs(tmp_path, monkeypatch):
    """Isolate compilation behind an executable stub and temporary output paths."""
    monkeypatch.delenv("TILECPP_COMPILE_TIMEOUT_S", raising=False)
    compiler = tmp_path / "nvcc"
    header = tmp_path / "kernel.cuh"
    header.write_text("")
    monkeypatch.setattr(_cuda_utils, "NVCC_PATH", str(compiler))
    monkeypatch.setattr(_cuda_utils, "_get_current_device_arch_flag", lambda: "sm_90a")
    monkeypatch.setattr(_cuda_utils, "should_save_source", lambda: False)
    return compiler, header, tmp_path / "kernel.cubin"


def write_compiler(path, source):
    """Write an executable Python stub that accepts the compiler command line."""
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)


def process_running(pid):
    """Check whether a Linux process still executes, excluding unreaped zombies."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Checks Linux compiler descendants")
@pytest.mark.parametrize("exception_type", [TimeoutError, KeyboardInterrupt, subprocess.TimeoutExpired])
def test_compile_interruption_stops_descendants(compiler_inputs, tmp_path, exception_type, monkeypatch):
    """An interrupted compile stops its descendants and preserves unrelated processes."""
    compiler, header, output = compiler_inputs
    pid_file = tmp_path / "compiler-pids.json"
    write_compiler(
        compiler,
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"ready = Path({str(pid_file)!r})\n"
        "temporary = ready.with_suffix('.tmp')\n"
        "temporary.write_text(json.dumps([os.getpid(), child.pid]))\n"
        "temporary.replace(ready)\n"
        "child.wait()\n",
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    started = time.monotonic()
    if exception_type is subprocess.TimeoutExpired:
        monkeypatch.setenv("TILECPP_COMPILE_TIMEOUT_S", "0.5")
        message = "timed out after 0.5 seconds"
    else:
        message = "compiler interrupted"

    def interrupt_when_ready(signum, frame):
        """Interrupt only after the compiler records its child process."""
        if pid_file.exists() and exception_type is not subprocess.TimeoutExpired:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise exception_type("compiler interrupted")
        if time.monotonic() - started > 5:
            pytest.fail("The fake compiler did not start or time out")

    previous_handler = signal.signal(signal.SIGALRM, interrupt_when_ready)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.05, 0.05)
    try:
        with pytest.raises(exception_type, match=message):
            _cuda_utils.compile_cuda_to_cubin(header, output)
        signal.setitimer(signal.ITIMER_REAL, 0)
        pids = json.loads(pid_file.read_text())
        deadline = time.monotonic() + 3
        while any(process_running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(process_running(pid) for pid in pids)
        assert unrelated.poll() is None
        assert not output.with_suffix(".cu").exists()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        if pid_file.exists():
            for pid in json.loads(pid_file.read_text()):
                if process_running(pid):
                    os.kill(pid, signal.SIGKILL)
        unrelated.kill()
        unrelated.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Checks Linux compiler descendants")
@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("compile_timeout", [None, "5"])
def test_compile_result_and_diagnostics(compiler_inputs, tmp_path, returncode, compile_timeout, monkeypatch):
    """Preserve compiler diagnostics and stop descendants of a failed compile."""
    if compile_timeout is not None:
        monkeypatch.setenv("TILECPP_COMPILE_TIMEOUT_S", compile_timeout)
    compiler, header, output = compiler_inputs
    pid_file = tmp_path / "compiler-pids.json"
    write_compiler(
        compiler,
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"if {returncode}:\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"    Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
        "print('compiler stdout')\n"
        "print('compiler stderr', file=sys.stderr)\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'cubin')\n"
        f"sys.exit({returncode})\n",
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        if returncode:
            with pytest.raises(RuntimeError, match="CUDA compilation failed: compiler stderr") as exc:
                _cuda_utils.compile_cuda_to_cubin(header, output)
            assert exc.value.__cause__.returncode == returncode
            assert exc.value.__cause__.stdout == "compiler stdout\n"
            assert exc.value.__cause__.stderr == "compiler stderr\n"
            pids = json.loads(pid_file.read_text())
            deadline = time.monotonic() + 3
            while any(process_running(pid) for pid in pids) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not any(process_running(pid) for pid in pids)
        else:
            assert _cuda_utils.compile_cuda_to_cubin(header, output) == output
            assert output.read_bytes() == b"cubin"
        assert unrelated.poll() is None
        assert not output.with_suffix(".cu").exists()
    finally:
        if pid_file.exists():
            for pid in json.loads(pid_file.read_text()):
                if process_running(pid):
                    os.kill(pid, signal.SIGKILL)
        unrelated.kill()
        unrelated.wait()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_compile_rejects_invalid_timeout(compiler_inputs, monkeypatch, value):
    """Reject invalid time limits before creating the compiler input file."""
    _, header, output = compiler_inputs
    monkeypatch.setenv("TILECPP_COMPILE_TIMEOUT_S", value)
    with pytest.raises(ValueError):
        _cuda_utils.compile_cuda_to_cubin(header, output)
    assert not output.with_suffix(".cu").exists()
