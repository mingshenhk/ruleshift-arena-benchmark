# RuleShift Arena — Competition Package

RuleShift Arena is a compact benchmark generator for the **Executive Functions** track of the Kaggle competition **Measuring Progress Toward AGI – Cognitive Abilities**.

This package is intentionally small:

* `run.py` generates benchmark datasets and split bundles
* `eval.py` evaluates models with a **semantic yes/no candidate scoring** method
* no extra validation/inspection tooling is required for normal competition use

## Files

* `run.py` — generator and environment module
* `eval.py` — semantic evaluator
* generated split files:

  * `public\_demo.jsonl`
  * `public\_demo.internal.jsonl`
  * `dev.jsonl`
  * `dev.internal.jsonl`
  * `private\_eval.jsonl`
  * `private\_eval.internal.jsonl`

## Quick start

### 1\) Generate a benchmark split bundle

```bash
python3 ./run.py \\
  --mode export-splits \\
  --output-dir ./arena\_splits\_comp \\
  --num-episodes 600 \\
  --difficulty-levels 1,2,3,4 \\
  --base-seed 20260327
```

### 2\) Evaluate Qwen small models

All four small official models:

```bash
python3 ./eval.py \\
  --generator-script ./run.py \\
  --dataset-path ./arena\_splits\_comp/dev.internal.jsonl \\
  --output-dir ./eval\_small\_core \\
  --preset qwen\_small\_core \\
  --save-traces
```

Only Qwen3 small:

```bash
python3 ./eval.py \\
  --generator-script ./run.py \\
  --dataset-path ./arena\_splits\_comp/dev.internal.jsonl \\
  --output-dir ./eval\_qwen3 \\
  --preset qwen3\_small \\
  --save-traces
```

Only Qwen3.5 small:

```bash
python3 ./eval.py \\
  --generator-script ./run.py \\
  --dataset-path ./arena\_splits\_comp/dev.internal.jsonl \\
  --output-dir ./eval\_qwen35 \\
  --preset qwen35\_small \\
  --save-traces
```

## Recommended presets

* `qwen3\_small`

  * `Qwen/Qwen3-0.6B`
  * `Qwen/Qwen3-1.7B`
* `qwen35\_small`

  * `Qwen/Qwen3.5-0.8B`
  * `Qwen/Qwen3.5-2B`
* `qwen\_small\_core`

  * all four above

## What the benchmark measures

The generator creates six task families:

* goal maintenance
* planning
* inhibitory control
* cognitive flexibility
* conflict resolution
* working memory

Each episode is a multi-step environment with rooms, routes, items, distractors, and family-specific rules. The benchmark is built so that models must repeatedly choose the best next action under uncertainty, instead of answering a one-shot recall question.

## Internal vs public files

* `\*.jsonl` files are public-facing exports
* `\*.internal.jsonl` keep hidden metadata used by evaluators

Use `dev.internal.jsonl` when running model evaluation.

## Evaluator design

The evaluator does **not** rely on fragile free-form action parsing.
Instead, it scores candidate actions semantically:

* for each candidate action, it asks whether that action is the best next step
* it compares `Yes` vs `No` conditional scores
* the highest-scoring candidate is selected

This makes comparison more stable than generation-then-parse baselines.

