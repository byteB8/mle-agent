"""GPU selection on a shared box: highest free index first (3 > 2 > 1 > 0)."""
from __future__ import annotations

import subprocess

FREE_MIB = 1000  # a GPU with less memory in use than this counts as free


def parse_free(nvidia_smi_csv: str, free_mib: int = FREE_MIB) -> list[int]:
    """Parse `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`
    and return free GPU indices in priority order (highest index first)."""
    free = []
    for line in nvidia_smi_csv.strip().splitlines():
        idx, used = (int(x.strip()) for x in line.split(",")[:2])
        if used < free_mib:
            free.append(idx)
    return sorted(free, reverse=True)


def pick_gpu(exclude: set[int] = frozenset()) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    for i in parse_free(out):
        if i not in exclude:
            return i
    raise RuntimeError("no free GPU")
