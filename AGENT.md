# Agent Specification

This project uses a two-agent workflow orchestrated with LangGraph to profile PyTorch models, identify performance bottlenecks, and generate Triton-based fused kernels.

## Shared Conventions
- **Evaluation Harness**: Baseline and optimized variants call `make_case_forward(case_name, device=...)` to obtain a zero-argument runner that wires shared `agent_provide_inputs()` and (optional) `agent_provide_weights()` helpers. `execute_variant(label, forward_impl, *, use_profiler=False)` seeds RNG, attaches cached buffers, performs a deterministic warmup, and captures standardized timing/accuracy metrics.
- **Profiler Access**: Both agents may request measurements via the `torch.profiler` tool wrapper exposed as `profiler_tool`.
- **Verification Tooling**: A `verification_tool` executes end-to-end regression checks (correctness, performance, compilation). Any failure must be propagated back to the Triton Code Engineer for iterative fixes.
- **Archiving**: Experiments conclude by writing a bundle to `archive/{timestamp}-{experiment_name}-{speedup}` that captures generated code, agent dialogues, profiler traces, and intermediate assets.

### Evaluation Harness Design
- `harness/cases.py` parses `cases/<case_name>/manifest.yaml`, imports the declared `torch.nn.Module` from `model.py`, and builds deterministic input/weight factories.
- Deterministic data helpers in `harness/data.py` cache tensors on disk (`agent_provide_inputs()`, `agent_provide_weights()`), ensuring both variants replay identical buffers.
- `harness/runner.py` exposes `execute_variant()` that: seeds RNG, materializes inputs through the shared cache, runs one warmup pass, performs profiled timing loops, validates numerical parity against the baseline, and emits normalized metrics.
- Metric snapshots land in `artifacts/{label}/metrics.json` with latency, throughput, memory, and correctness flags, enabling the Strategy Analyst to compare variants without manual parsing.
- Profiler traces are exported to `artifacts/{label}/profiler/` for downstream inspection and archival.

### Case Specification
- Each case places the unmodified PyTorch module in `model.py` and documents metadata in `manifest.yaml` (model module/class, constructor kwargs, device preference, descriptive text). No bespoke runner glue is required.
- `manifest.yaml` also declares the input schema (`inputs.args` / `inputs.kwargs`). The harness synthesizes tensors from this spec, caches them, and feeds identical payloads to both variants so the model can observe concrete optimization targets.
- Optional deterministic weight initializers can be advertised via `weights.function` pointing to a helper inside `model.py`; otherwise the framework captures the module’s default `state_dict()` after seeding.
- Only PyTorch code that (1) exposes a pure `forward` entry, (2) executes on CUDA tensors, and (3) allows the performance-critical region to be isolated into a Triton kernel is considered in scope. Models may freely inspect inputs to enable targeted optimizations, but they must avoid side effects that break deterministic replays across variants.

## Agents

### Strategy Analyst
- **Role**: Diagnose current performance, interpret profiler data, and draft optimization strategies.
- **Inputs**: Profiler summaries, baseline metrics, verification outcomes.
- **Outputs**: Structured optimization plan (target ops, fusion opportunities, Triton kernel specs), go/no-go signals for subsequent iterations.
- **Restrictions**: Does **not** author code. Delegates all implementation steps.

### Triton Code Engineer
- **Role**: Implement the analyst's plan by generating Triton kernels, integrating them into the PyTorch codebase, and ensuring end-to-end executability.
- **Responsibilities**:
  - Translate profiler hotspots into Triton kernels.
  - Replace bottlenecked PyTorch segments with calls to the new Triton functions in a runnable Python module.
  - Run `verification_tool` and interpret failures to drive iterative fixes.
  - Package final artifacts for archiving.

## LangGraph Workflow
1. **Data Collection Node**: Runs the standardized harness, collects profiler data, and publishes artifacts.
2. **Strategy Decision Node**: Strategy Analyst interprets metrics, emits optimization directives or archive signal.
3. **Implementation Node**: Triton Code Engineer generates/updates code per directives, invokes verification.
4. **Verification Feedback Loop**:
   - On success, forward results to Strategy Analyst for final approval.
   - On failure, send diagnostics back to the Triton Code Engineer for another iteration.
5. **Archive Node (Terminal)**: Once approved, bundle all experiment artifacts under the timestamped directory within `archive/`.
