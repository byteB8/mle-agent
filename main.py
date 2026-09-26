"""LLM server entry point: vLLM's OpenAI-compatible API, configured from the environment.

Started by serve.sh as `python main.py`. Settings come from SRV_* environment variables rather than
the command line, and every process title vLLM would set (APIServer, EngineCore, workers) is replaced
by PROC_TITLE, so on a shared machine `ps`/nvitop show only a generic `python main.py`.
"""
import os
import sys


def _pin_process_title() -> None:
    title = os.environ.get("PROC_TITLE", "python main.py")
    try:
        import setproctitle
    except ImportError:
        return
    original = setproctitle.setproctitle
    setproctitle.setproctitle = lambda *_args, **_kwargs: original(title)
    original(title)


# Top level on purpose: spawned child processes re-import this module (as __mp_main__) before vLLM
# retitles them, so the patch is in place in the engine core and workers too.
_pin_process_title()


def main() -> None:
    env = os.environ.get
    sys.argv = [
        "vllm", "serve", env("SRV_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        "--served-model-name", env("SRV_NAME", "coder"),
        "--host", "127.0.0.1", "--port", env("SRV_PORT", "8011"),
        "--max-model-len", env("SRV_MAX_LEN", "65536"),
        "--gpu-memory-utilization", env("SRV_GPU_UTIL", "0.92"),
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--enable-prefix-caching",
    ]
    from vllm.entrypoints.cli.main import main as cli
    cli()


if __name__ == "__main__":
    main()
