# llm4triton

Prototype workspace for an agentic optimization pipeline that rewrites PyTorch kernels into Triton for performance gains.

Key goals:
- Provide a standardized evaluation harness that replays identical inputs/weights via manifest-driven case definitions and shared caching helpers.
- Coordinate two specialized agents (Strategy Analyst and Triton Code Engineer) via LangGraph to analyze, optimize, verify, and archive experiments.
- Maintain detailed experiment archives under `archive/{timestamp}-{experiment}-{speedup}` including code, conversations, and artifacts.

See `PLAN.md` for the staged roadmap and `AGENT.md` for agent definitions and workflow details.
