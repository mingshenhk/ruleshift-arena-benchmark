# RuleShift Arena

A held-out executive-function benchmark for goal maintenance, planning, inhibitory control, cognitive flexibility, conflict resolution, and working memory.

## Why this benchmark

Existing benchmarks often over-reward recall and shallow pattern matching. RuleShift Arena focuses on multi-step adaptive control under distractors, rule shifts, and memory-dependent decisions.

## What is included

- `run.py`: benchmark generator
- `eval.py`: semantic candidate-scoring evaluator

## Quick start

### 1) Generate a benchmark split bundle

```bash
python3 ./run.py \
  --mode export-splits \
  --output-dir ./arena_splits_comp \
  --num-episodes 600 \
  --difficulty-levels 1,2,3,4 \
  --base-seed 20260327
```

### 2) Evaluate Qwen small models

All four small official models:

```bash
python3 ./eval.py \
  --generator-script ./run.py \
  --dataset-path ./arena_splits_comp/dev.internal.jsonl \
  --output-dir ./eval_small_core \
  --preset qwen_small_core \
  --save-traces
```

Only Qwen3 small:

```bash
python3 ./eval.py \
  --generator-script ./run.py \
  --dataset-path ./arena_splits_comp/dev.internal.jsonl \
  --output-dir ./eval_qwen3 \
  --preset qwen3_small \
  --save-traces
```

Only Qwen3.5 small:

```bash
python3 ./eval.py \
  --generator-script ./run.py \
  --dataset-path ./arena_splits_comp/dev.internal.jsonl \
  --output-dir ./eval_qwen35 \
  --preset qwen35_small \
  --save-traces
```

## Recommended presets

- `qwen3_small`
  - `Qwen/Qwen3-0.6B`
  - `Qwen/Qwen3-1.7B`

- `qwen35_small`
  - `Qwen/Qwen3.5-0.8B`
  - `Qwen/Qwen3.5-2B`

- `qwen_small_core`
  - all four above

## What the benchmark measures

The generator creates six task families:

- goal maintenance
- planning
- inhibitory control
- cognitive flexibility
- conflict resolution
- working memory

Each episode is a multi-step environment with rooms, routes, items, distractors, and family-specific rules. The benchmark is built so that models must repeatedly choose the best next action under uncertainty, instead of answering a one-shot recall question.

## Internal vs public files

- `*.jsonl` files are public-facing exports
- `*.internal.jsonl` files keep hidden metadata used by evaluators

Use `dev.internal.jsonl` when running model evaluation.

## Evaluator design

The evaluator does **not** rely on fragile free-form action parsing. Instead, it scores candidate actions semantically:

- for each candidate action, it asks whether that action is the best next step
- it compares `Yes` vs `No` conditional scores
- the highest-scoring candidate is selected

This makes comparison more stable than generation-then-parse baselines.

## Initial Results

We evaluated four official small Qwen models on the development split using the semantic candidate-scoring evaluator (`hf_semantic_yesno_scoring`) on `dev.internal.jsonl` with 72 held-out episodes. Across all four models, evaluation completed with zero fallback steps and zero scoring errors, so the reported differences reflect model decisions rather than parser or fallback artifacts.

### Overall ranking

| Model | Mean Raw Score | Success Rate | Mean Margin (Top1–Top2) |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 0.4742 | 0.4583 | 0.5531 |
| Qwen3-0.6B | 0.1432 | 0.0278 | 0.1155 |
| Qwen3.5-2B | 0.1377 | 0.0000 | 0.7257 |
| Qwen3-1.7B | 0.1160 | 0.0000 | 0.1734 |

A key result is that **Qwen3.5-0.8B substantially outperforms all other tested models**, showing that RuleShift Arena does not simply rank models by parameter count. The benchmark also captures **high-confidence failure**: Qwen3.5-2B has the highest decision margin but still achieves zero successful completions.

### Family-level observations

- **Goal maintenance** is highly separating: Qwen3.5-0.8B reaches **0.9460 mean raw score** with **100% success**, while the other three models remain far behind.
- **Working memory** is another strong separator: Qwen3.5-0.8B reaches **0.8387** with **75% success**, while the remaining models stay much lower.
- **Cognitive flexibility** remains challenging for larger models in this test set: Qwen3-0.6B scores **0.2115**, while Qwen3-1.7B and Qwen3.5-2B drop to **0.0292** and **0.0542**, respectively.
- **Planning** is still the weakest family in the current version: all four models achieve **0% success**, indicating that this family still needs refinement to better support partial recovery and non-zero completion.

### Main takeaway

RuleShift Arena already produces a **non-monotonic model ranking** and reveals **confident failure modes**, which are both useful benchmark properties for executive-function evaluation. At the same time, the current results suggest that the **planning** family should be further refined, and that **cognitive flexibility** and **inhibitory control** can still be made more elegant by reducing over-reliance on violation-heavy failure patterns.

## Limitations

The current benchmark already shows stable model separation, but some families still need refinement. In particular, the planning family remains too weak because all tested models still achieve zero successful completions, and cognitive flexibility can still be made more elegant by reducing over-reliance on violation-heavy failure patterns.
