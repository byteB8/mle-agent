"""Docker sandbox for agent-written code.

One long-lived container per run: task data mounted read-only at /data, a per-run
scratch dir mounted at /work, no network, CPU/memory/pid limits, runs as the host
user so files it creates stay ours. Commands are killed from *inside* the container
(`timeout -s KILL`) so a runaway training loop can't outlive its budget.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def truncate(text: str, limit: int = 12000) -> str:
    """Keep head and tail of long output; the tail usually holds the error/metric."""
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n\n... [{len(text) - limit} chars truncated] ...\n\n{text[-tail:]}"


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool
    duration: float


class DockerSandbox:
    def __init__(self, name: str, image: str, workdir: Path, data_dir: Path,
                 cpus: float = 8, memory: str = "32g", gpus: str | None = None,
                 network: bool = False):
        self.name = name
        self.image = image
        self.workdir = Path(workdir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.cpus, self.memory, self.gpus, self.network = cpus, memory, gpus, network

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "run", "-d", "--rm", "--name", self.name,
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--cpus", str(self.cpus), "--memory", self.memory, "--memory-swap", self.memory,
               "--pids-limit", "1024", "--shm-size", "8g",
               "--tmpfs", "/tmp:rw,exec,size=16g",
               "-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1",
               "-e", f"OMP_NUM_THREADS={int(self.cpus)}",
               "-v", f"{self.data_dir}:/data:ro", "-v", f"{self.workdir}:/work", "-w", "/work"]
        if not self.network:
            cmd += ["--network", "none"]
        if self.gpus:
            cmd += ["--gpus", f"device={self.gpus}"]
        cmd += [self.image, "sleep", "infinity"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"sandbox start failed: {r.stderr.strip()}")

    def exec(self, command: str, timeout: int = 600) -> ExecResult:
        t0 = time.monotonic()
        argv = ["docker", "exec", self.name, "timeout", "-s", "KILL", str(int(timeout)),
                "bash", "-lc", command]
        try:
            r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout + 30)
            out, code = r.stdout.decode(errors="replace"), r.returncode
        except subprocess.TimeoutExpired as e:  # docker exec itself hung
            out, code = (e.stdout or b"").decode(errors="replace"), 137
        dur = time.monotonic() - t0
        return ExecResult(exit_code=code, output=out, timed_out=code in (124, 137) and dur >= timeout - 1,
                          duration=dur)

    def host_path(self, rel: str) -> Path:
        """Map a sandbox path (/work/x or x) to the host, refusing escapes."""
        if rel == "/work" or rel.startswith("/work/"):
            rel = rel[len("/work"):].lstrip("/")
        elif rel.startswith("/"):
            raise ValueError("only paths under /work are accessible")
        p = (self.workdir / rel).resolve()
        if p != self.workdir and self.workdir not in p.parents:
            raise ValueError(f"path escapes /work: {rel}")
        return p

    def write_file(self, rel: str, content: str) -> Path:
        p = self.host_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
