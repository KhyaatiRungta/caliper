# Caliper

An evaluation and observability harness for LLM agents. It runs a task suite
against your agent, scores the whole **trajectory** rather than just the final
answer, stores every run as a file, and diffs two runs to surface regressions.

It exits non-zero when a task goes `pass -> fail`, so it works as a CI gate:

```yaml
# .github/workflows/eval.yml
- run: caliper run suites/reference.yaml --agent reference-v2
- run: caliper compare latest latest~1 --fail-on-regression
```

```
SUITE: reference-agent v2   vs   v1
-------------------------------------------
task success      92.0%      36.0%   +56.0
steps (median)      2.0        4.0    -2.0
cost / task      $0.020     $0.035    -43%
p95 latency        0.3s       0.5s    -45%
tool-error rate    6.7%      60.3%   -53.6

REGRESSIONS (0)
  none
```

Those numbers are real output from `results/reference-v2-vs-v1.txt`, produced
by the two versions of the built-in reference agent on the 25-task reference
suite. Reproduce them in about ten seconds with no API key:

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python -m caliper.cli run suites/reference.yaml --agent reference-v1 --offline
python -m caliper.cli run suites/reference.yaml --agent reference-v2 --offline
python -m caliper.cli compare latest latest~1
```

## Why trajectory scoring

Here is one task from that run. Both agent versions returned the same answer:

| version | final answer | steps | cost   | `numeric_tolerance` | task |
|---------|--------------|-------|--------|---------------------|------|
| v1      | `42`         | 5     | $0.045 | pass                | FAIL |
| v2      | `42`         | 2     | $0.018 | pass                | pass |

A final-answer grader cannot tell these apart, because on the only axis it
measures they are identical. v1 called the knowledge base twice for an
arithmetic question, failed both times, then computed the answer and computed
it again to "double-check". It is 2.5x the cost and 2.5x the latency for the
same output, and it is one traffic spike away from being a production incident.

Caliper fails that task on `step_efficiency` and `no_redundant_calls`. That is
the whole argument for the project.

## Install

```bash
git clone <this-repo> caliper && cd caliper
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # optional: puts `caliper` on PATH
cp .env.example .env      # only needed for the llm_judge grader
```

Everything except `llm_judge` and live agent runs works with no API key.

## The adapter contract

This is the entire coupling between Caliper and your agent:

```python
class AgentAdapter(Protocol):
    name: str
    version: str
    def run(self, task: Task) -> Trajectory: ...
```

No framework, no base class, no monkeypatching of your LLM client. You record
steps and hand back a trajectory:

```python
from caliper.adapter import TrajectoryRecorder

class MyAgent:
    name = "my-agent"
    version = "v3"

    def run(self, task):
        rec = TrajectoryRecorder(task.id)
        with rec.step("llm", "plan") as s:
            resp = self.client.complete(messages)
            s.output = resp.text
            s.tokens_in = resp.usage.prompt_tokens
            s.tokens_out = resp.usage.completion_tokens
            s.cost_usd = resp.usage.cost_usd
        with rec.step("tool", "search", input=query) as s:
            s.output = self.search(query)
        return rec.finish(final_output=answer)
```

`rec.step` times the block, assigns the index, and captures any exception into
`step.error` before re-raising, so an agent that dies mid-tool still leaves an
accurate record of the tool that killed it.

Run it:

```bash
caliper run suites/mine.yaml --agent mypkg.adapter:build
```

## Graders

Registry-based, referenced by name from the suite YAML. Every grader is a pure
function with the same signature, so a run file can be re-graded later without
re-running the agent.

**Final answer**

| grader | what it checks |
|--------|----------------|
| `exact_match` | whitespace-normalised string equality |
| `numeric_tolerance` | numeric comparison, `rel` and/or `abs` |
| `contains_all` | every required substring present, partial credit |
| `regex_match` | `search` or `fullmatch` |
| `json_schema_match` | parses as JSON and satisfies a small schema subset |
| `llm_judge` | rubric-based via OpenRouter, `n_samples` majority vote |

**Trajectory** — the ones a final-answer harness structurally cannot provide

| grader | what it checks |
|--------|----------------|
| `step_efficiency` | steps taken vs a reference optimum, with slack |
| `tool_choice_precision` | were the tools called the ones actually needed |
| `no_redundant_calls` | same tool, same arguments, twice |
| `recovered_from_error` | did an errored step get followed by a success |
| `cost_budget` | trajectory cost against a task-level budget |
| `latency_budget` | wall clock against a task-level budget |
| `completed` | did the agent terminate normally at all |

Adding one:

```python
from caliper.graders.registry import grader
from caliper.types import Score

@grader("answers_in_one_sentence")
def answers_in_one_sentence(task, trajectory, max_sentences: int = 1) -> Score:
    n = str(trajectory.final_output).count(".")
    return Score(name="answers_in_one_sentence", value=1.0 if n <= max_sentences else 0.0,
                 passed=n <= max_sentences, detail=f"{n} sentence(s)")
```

Import the module from `caliper/graders/__init__.py` and reference it by name.
There is no step 4.

## Suites

```yaml
suite: reference
defaults:
  timeout_s: 15
  cost_budget_usd: 0.025
tasks:
  - id: reverse_looked_up_fact
    input: "Reverse the capital of Japan."
    expected: "oykoT"
    grader:
      - exact_match: {case_sensitive: true}
      - tool_choice_precision
      - step_efficiency
    expected_tools: [lookup, string_op]
    reference_steps: 3
    tags: [composite]
```

Shipped suites: `suites/reference.yaml` (25 tasks, exercises every grader),
`suites/strata.yaml` and `suites/quarry.yaml` for the two sibling agent
projects. The sibling adapters import lazily and fail with instructions if the
repo is not on the import path, if its environment is broken, or if its
constructor needs wiring the adapter cannot guess.

Those two adapters are **provisional**: Strata and Quarry are being written in
parallel, so the adapters bind to whichever conventional entry point those
repos expose and tolerate several shapes of trace. They will be pinned once the
sibling APIs settle. The reference agent, not those adapters, is what the test
suite guarantees.

## CLI

```
caliper run <suite> --agent <name> [--workers N] [--limit N] [--tag t] [--offline]
caliper list                                 recent runs
caliper show <run_id>                        summary plus per-task breakdown
caliper compare <run_a> <run_b>              the table above
caliper compare a b --fail-on-regression     CI gate, exits 1 on any pass -> fail
caliper trace <run_id> <task_id>             step-by-step trajectory dump
caliper report <run_id> --html out.html      standalone HTML report
caliper graders                              list registered graders
caliper prune --keep 50                      delete old runs
```

Run ids accept a unique prefix, `latest`, or `latest~1`.

Exit codes: `0` clean, `1` regressions detected, `2` usage or config error.

## Architecture

```
suites/*.yaml
     |
     v
  runner  --(ThreadPoolExecutor, N workers, per-task timeout)--> adapter --> agent
     |                                                                         |
     |                                      TrajectoryRecorder <---------------+
     v                                              |
  graders <---------------------------------------- Trajectory
     |   final-answer + trajectory
     v
  .caliper/runs/<run_id>.json   (run store)
     |
     v
  compare  -->  regression gate (exit 1)
```

Concretely: `caliper/types.py` is the data model, `caliper/adapter.py` the
boundary, `caliper/graders/` the scoring, `caliper/runner.py` the execution,
`caliper/store.py` persistence, `caliper/compare.py` the diff,
`caliper/render.py` the output, `caliper/cli.py` the entry point.

## Design decisions

**Adapters, not framework integration.** Caliper could have shipped callbacks
for LangChain and friends and got instrumentation for free. Instead the entire
coupling is `run(task) -> Trajectory`. Cost: every agent author writes about
ten lines, and Caliper cannot see steps the agent chooses not to record. Gain:
Caliper works on a hand-written loop, a framework agent, a remote HTTP service
and a shell script, and it does not break when a framework changes its callback
API.

**Trajectory graders as first-class.** Final-answer grading is a strict subset
of what matters. The table at the top of this file shows two agents with
identical answers on a task and a 2.5x cost difference; only trajectory grading
separates them. Cost: trajectory graders need the task author to supply a
reference (`reference_steps`, `expected_tools`), which is more suite-authoring
work than writing an expected string.

**Runs are files, not a database.** A run is an immutable artifact produced by
CI. Files diff, attach to a PR, commit as a pinned baseline, and copy with
`scp`. Cost: no indexed queries, and listing a thousand runs reads a thousand
files. `caliper/store.py` is the only module that would change if that tradeoff
flipped.

**Regressions exit non-zero.** A report nobody reads changes nothing. A gate
that blocks a merge changes behaviour. The gate is deliberately narrow: a named
task that used to pass and now does not, never a metric that moved the wrong
way and never an infrastructure error, because a gate that fires on a rate
limit teaches the team to pass `--no-verify`.

**Deterministic output ordering.** Results are emitted in suite order
regardless of completion order, so two run files diff cleanly.

## Limitations

- **LLM-judge variance is real.** Re-grading the same response can land in a
  different bucket, especially near the 0.75/0.50 boundary. Mitigated with
  temperature 0, a fixed five-point rubric, and `n_samples` majority vote with
  agreement reported in the score detail; not eliminated. A comparison that
  hinges entirely on a few points of `llm_judge` delta hinges on noise.
- **No distributed execution.** One process, a thread pool, one machine. Fine
  for a suite of a few hundred tasks; not a replacement for a job queue.
- **Timed-out tasks are abandoned, not killed.** Python cannot safely kill a
  thread. The result is recorded immediately and the run proceeds, but the
  worker may still be running underneath.
- **No live dashboard.** Plain text and a static HTML report. Deliberate, but
  it means no streaming view of a long run beyond the stderr progress lines.
- **A suite is only as good as its author.** Caliper measures what you tell it
  to measure. `reference_steps` and `expected_tools` are human judgements, and
  a badly chosen reference optimum produces a confidently wrong efficiency
  score.
- **Cost figures depend on OpenRouter's reported usage.** When the provider
  reports usage accounting Caliper uses it; otherwise it falls back to a
  per-model price table that will drift out of date.
- **The reference agent is deterministic and rule-based.** It makes no LLM
  calls. Its v1 and v2 differ in routing policy, which stands in for a prompt
  change. The measurements of it are real; the agent is a fixture.

## Tests

```bash
pytest          # 273 tests, no API key required
```

Covers every grader including empty trajectories and zero-denominator rate
metrics, runner isolation when a task raises, runner timeouts and queue-time
exclusion, deterministic ordering under 1/2/8 workers, regression detection in
both directions, the comparison table against a golden string, and store
round-trip.

## Related

Part of a three-repo set on building and measuring agents:

- [strata](https://github.com/Manavarya09/strata) - deep research agent (planner, hybrid
  retrieval, reflection)
- [quarry](https://github.com/Manavarya09/quarry) - data analyst agent (tool calls,
  sandboxed execution, repair loop)
- **caliper** - the measurement layer, used on both of the above

---

caliper - 2026
