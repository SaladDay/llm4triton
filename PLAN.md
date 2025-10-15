# Roadmap

## Milestone 1 – Evaluation Harness Foundations
- Implement `harness/data.py` with deterministic `agent_provide_inputs()`/`agent_provide_weights()` fallbacks, RNG seeding utilities, and tensor caching so both variants replay identical buffers.
- Build `harness/runner.py` exposing `execute_variant(label, forward_impl, *, use_profiler=False)` that performs seeding, tensor attachment, warmup, timed/profiler loops, and cached-baseline parity checks.
- Add `harness/cases.py` to parse `manifest.yaml`, import the declared `torch.nn.Module`, and synthesize cached input payloads/weight factories to feed `make_case_forward()`.
- Serialize metrics (`latency`, `throughput`, `memory`, `correctness`) to `artifacts/{label}/metrics.json` and profiler traces to `artifacts/{label}/profiler/`, delivering structured evidence for the Strategy Analyst.
- Define the case contract under `cases/<case_name>/`: unmodified `model.py`, declarative `manifest.yaml` (model module/class, constructor kwargs, input distributions, optional weight provider, metadata). Provide a starter template plus registration guide that preserves the original PyTorch file and relies on zero glue code.

## Milestone 2 – Agentic Optimization Loop
- Assemble the LangGraph workflow linking Strategy Analyst and Triton Code Engineer nodes using shared state for metrics, manifests, and profiler outputs.
- Integrate `profiler_tool` and `verification_tool` invocations so strategy decisions trigger Triton code generation, iterative debugging, and correctness/performance validation.
- Implement workflow governance: decision -> implementation -> verification loop, culminating in handoff to the archiver node when optimization goals are met.
- Establish a central configuration module that loads a repository-level `.env` file (via `python-dotenv` or equivalent) so agent and experiment parameters can be managed without editing source code.

## Milestone 3 – Experiment Archiving & Developer Ops
- Finalize the archiver node to bundle code, agent dialogues, profiler traces, and metrics under `archive/{timestamp}-{experiment}-{speedup}`.
- Add developer tooling: CLI scripts or notebooks to inspect archives, regenerate reports, and seed new cases from the provided templates.
- Document end-to-end runbooks (from case creation to archive review) ensuring new contributors can operate the system with minimal ramp-up.
