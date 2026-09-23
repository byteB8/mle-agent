"""Run one agent episode on one task and grade it.

    python run.py --task data/dev-adult --model coder --base-url http://127.0.0.1:8011/v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.agent import Agent, Budget
from core.gpu import pick_gpu
from core.llm import LLM
from core.sandbox import DockerSandbox
from core.search import SearchConfig, TreeSearchAgent
from core.tasks import Task
from core.trace import Tracer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task directory")
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="coder")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--time-limit-min", type=float, default=60)
    ap.add_argument("--cpus", type=float, default=8)
    ap.add_argument("--memory", default="32g")
    ap.add_argument("--gpus", default=None, help="GPU index for the sandbox, or 'auto' (highest free index)")
    ap.add_argument("--image", default="exp-rt:cpu")
    ap.add_argument("--min-submit-frac", type=float, default=0.0,
                    help="reject submit before this fraction of the time budget (0 = off)")
    ap.add_argument("--env-facts", action="store_true", help="put installed library versions in the prompt")
    ap.add_argument("--agent", choices=["react", "tree"], default="react")
    ap.add_argument("--num-drafts", type=int, default=3, help="tree: initial independent drafts")
    ap.add_argument("--debug-prob", type=float, default=0.5, help="tree: chance to debug a broken leaf")
    ap.add_argument("--node-timeout", type=int, default=600, help="tree: seconds per solution script")
    args = ap.parse_args()

    if args.gpus == "auto":
        args.gpus = str(pick_gpu())
    task = Task.load(args.task)
    run_id = f"{args.tag}-{task.name}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out) / run_id
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    budget = Budget(max_steps=args.max_steps, time_limit_s=args.time_limit_min * 60,
                    min_submit_frac=args.min_submit_frac)
    tracer = Tracer(out / "trace.jsonl")
    sandbox = DockerSandbox(name=f"exp-{run_id}"[:60], image=args.image, workdir=out / "work",
                            data_dir=task.public_dir, cpus=args.cpus, memory=args.memory, gpus=args.gpus)
    llm = LLM(args.base_url, args.model)
    try:
        with sandbox:
            if args.agent == "tree":
                cfg = SearchConfig(num_drafts=args.num_drafts, debug_prob=args.debug_prob,
                                   node_timeout_s=args.node_timeout)
                agent = TreeSearchAgent(llm, task, sandbox, tracer, budget, cfg, temperature=args.temperature,
                                        seed=args.seed, use_env_facts=args.env_facts)
            else:
                agent = Agent(llm, task, sandbox, tracer, budget, temperature=args.temperature, seed=args.seed,
                              use_env_facts=args.env_facts)
            result = agent.run()
    finally:
        tracer.close()
    result["run_id"] = run_id
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
