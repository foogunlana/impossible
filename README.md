# impossible

Coding tasks that look solvable and are not. The model either concedes the task
is infeasible, or games the evaluation to manufacture a pass.

```bash
uv sync
inspect eval impossible/fast_sum --model openai/o3
inspect eval impossible/fast_sum --model openai/o3 -T anti_hack=true
```

Reconstructed from the "impossible code" eval described in the Claude Sonnet 4.5
system card §6.1, used for the reward-hacking case study in arXiv:2604.07729 §3.3.

See [docs/spec.md](docs/spec.md) for what this is, why it is not
[ImpossibleBench](https://arxiv.org/abs/2510.20270), and what is still open.
