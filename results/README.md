# Committed results

These files are the exact output of the runs whose numbers appear on the
website and in the project README. They are committed so that every figure
shown can be traced back to a run file rather than taken on trust.

- `reference-v1.json`        run of `suites/reference.yaml` against `reference-v1`
- `reference-v2.json`        the same suite against `reference-v2`
- `reference-v2-vs-v1.txt`   the comparison table, verbatim
- `reference-v2-vs-v1.json`  the same comparison, machine-readable
- `reference-v2-report.html` the standalone HTML report for the v2 run

Reproduce:

    caliper run suites/reference.yaml --agent reference-v1 --offline
    caliper run suites/reference.yaml --agent reference-v2 --offline
    caliper compare latest latest~1

Latency figures vary by machine; every other figure is deterministic and will
reproduce exactly.
