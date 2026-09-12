"""Impossible coding tasks: benchmark for measuring reward hacking under pressure.

Each task is solvable-looking but unreachable by any general method. The model
either concedes the task is infeasible or games the evaluation. Modelled on the
"impossible code" eval in the Claude Sonnet 4.5 system card (§6.1) and used for
the reward-hacking case study in arXiv:2604.07729 §3.3.

Run one task:
    inspect eval fast_sum --model openai/o3

Run the whole set:
    from inspect_ai import eval_set
    from impossible import impossible_tasks
    eval_set(impossible_tasks(), model="openai/o3", log_dir="logs/impossible")
"""

from inspect_ai import Task

from .fast_sum import fast_sum

__all__ = ["fast_sum", "TASKS", "impossible_tasks"]

#: Every task in the benchmark. Add new tasks here and they join the set.
TASKS = [fast_sum]


def impossible_tasks(anti_hack: bool = False, max_attempts: int = 15) -> list[Task]:
    """Build the full benchmark as a list of tasks, for `eval_set`.

    Args:
        anti_hack: use the system card's anti-hack prompt variant.
        max_attempts: attempt cap before a run is recorded as `exhausted`.
    """
    return [t(anti_hack=anti_hack, max_attempts=max_attempts) for t in TASKS]
