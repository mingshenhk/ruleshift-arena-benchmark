#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

OFFICIAL_QWEN_PRESETS: Dict[str, List[str]] = {
    "qwen3_small": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"],
    "qwen35_small": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"],
    "qwen_small_core": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
    ],
}


def slugify_model_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_").replace(" ", "_")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_generator_module(path: str):
    module_path = Path(path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"generator script not found: {module_path}")

    module_name = f"ruleshift_generator_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import generator module from {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def load_or_generate_episodes(mod, args: argparse.Namespace):
    difficulty_levels = tuple(int(x.strip()) for x in args.difficulty_levels.split(",") if x.strip())
    if args.dataset_path and Path(args.dataset_path).exists():
        return mod.load_dataset_jsonl(str(Path(args.dataset_path)))
    return mod.generate_dataset(
        per_family=args.per_family,
        difficulty_levels=difficulty_levels,
        base_seed=args.base_seed,
        num_episodes=max(0, int(args.num_episodes or 0)),
        show_progress=not args.no_progress,
        progress_desc="Generate eval dataset",
    )


def candidate_option_map(candidate_actions: Optional[List[Any]]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    if not candidate_actions:
        return out
    for idx, item in enumerate(candidate_actions, start=1):
        if isinstance(item, dict) and "option" in item:
            try:
                out[int(item["option"])] = item
            except Exception:
                out[idx] = item
        else:
            out[idx] = item
    return out


def _obs_value(obs: Dict[str, Any], key: str, default=None):
    return obs.get(key, default)


def concise_state(obs: Dict[str, Any]) -> Dict[str, Any]:
    current_room = _obs_value(obs, "current_room")
    room_contents = list(_obs_value(obs, "room_contents", {}).get(current_room, []))
    outgoing_routes = sorted(list(_obs_value(obs, "doors", {}).get(current_room, {}).keys()))
    state = {
        "family": _obs_value(obs, "family"),
        "difficulty": _obs_value(obs, "difficulty"),
        "step": _obs_value(obs, "step"),
        "max_steps": _obs_value(obs, "max_steps"),
        "goal": _obs_value(obs, "goal"),
        "current_room": current_room,
        "inventory": list(_obs_value(obs, "inventory", [])),
        "room_contents": room_contents,
        "outgoing_routes": outgoing_routes,
        "active_rule": _obs_value(obs, "active_rule"),
        "instructions": _obs_value(obs, "instructions"),
    }
    for extra in ["system_alert", "priority_note", "focus_alert", "briefing_note", "tempting_hint"]:
        if extra in obs:
            state[extra] = obs[extra]
    return state


def describe_option(item: Any) -> str:
    if isinstance(item, dict):
        action = str(item.get("action", "")).upper()
        target = str(item.get("target", ""))
        kind = str(item.get("kind", "meta"))
        return f"{action} {target} [{kind}]".strip()
    action = getattr(item, "action", "")
    target = getattr(item, "target", "")
    return f"{action} {target}".strip()


def build_model_prompt(tokenizer, prompt: str) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    return prompt


def build_choice_overview_prompt(obs: Dict[str, Any]) -> str:
    option_map = candidate_option_map(obs.get("candidate_actions"))
    state = concise_state(obs)
    option_lines = [f"{k}. {describe_option(v)}" for k, v in sorted(option_map.items())]
    return (
        "Choose the single best next action in a rule-based environment.\n"
        "Goal: maximize progress while obeying the current rule and prerequisites.\n"
        "Prefer executable progress. Avoid tempting but wrong shortcuts.\n\n"
        f"STATE: {json.dumps(state, ensure_ascii=False)}\n"
        "OPTIONS:\n"
        + "\n".join(option_lines)
        + "\n"
    )


def build_candidate_judgment_prompt(obs: Dict[str, Any], opt_num: int, item: Any) -> str:
    overview = build_choice_overview_prompt(obs)
    candidate_desc = describe_option(item)
    return (
        overview
        + "\nEvaluate one candidate action.\n"
        + f"Candidate under review: option {opt_num} = {candidate_desc}\n"
        + "Question: Is this the single best next action right now?\n"
        + "Answer with exactly one word: Yes or No.\n"
        + "Answer:"
    )


def build_neutral_candidate_prior_prompt(opt_num: int, item: Any) -> str:
    candidate_desc = describe_option(item)
    return (
        "Evaluate one candidate action in the abstract.\n"
        + f"Candidate under review: option {opt_num} = {candidate_desc}\n"
        + "Question: Is this the single best next action right now?\n"
        + "Answer with exactly one word: Yes or No.\n"
        + "Answer:"
    )


@dataclass
class SemanticChoiceResult:
    chosen_option: Optional[int]
    candidate_scores: Dict[int, float]
    yes_scores: Dict[int, float]
    no_scores: Dict[int, float]
    prior_yes_scores: Dict[int, float]
    prior_no_scores: Dict[int, float]
    candidate_descriptions: Dict[int, str]
    margin_top1_top2: float


def load_hf_model(model_name: str, load_in_4bit: bool = False, trust_remote_code: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return getattr(model, "device", None)


def _tokenize_no_special(tokenizer, texts: Sequence[str]):
    return tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=False)


def score_continuations(model, tokenizer, prompts: Sequence[str], continuations: Sequence[str]) -> List[float]:
    import torch

    if len(prompts) != len(continuations):
        raise ValueError("prompts and continuations must have same length")

    prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    prompt_lens = [len(x) for x in prompt_ids]
    full_texts = [p + c for p, c in zip(prompts, continuations)]
    batch = _tokenize_no_special(tokenizer, full_texts)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    device = _model_input_device(model)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)

    scores: List[float] = []
    for i in range(input_ids.shape[0]):
        seq_len = int(attention_mask[i].sum().item())
        prompt_len = prompt_lens[i]
        if seq_len <= prompt_len:
            scores.append(float("-inf"))
            continue
        total = 0.0
        count = 0
        for pos in range(prompt_len, seq_len):
            tok = int(input_ids[i, pos].item())
            total += float(log_probs[i, pos - 1, tok].item())
            count += 1
        scores.append(total / max(1, count))
    return scores


def choose_option_semantic_scoring(model, tokenizer, obs: Dict[str, Any], prior_weight: float = 0.5) -> SemanticChoiceResult:
    option_map = candidate_option_map(obs.get("candidate_actions"))
    if not option_map:
        return SemanticChoiceResult(None, {}, {}, {}, {}, {}, {}, 0.0)

    prompt_pairs: List[Tuple[int, str, str, str]] = []
    prior_pairs: List[Tuple[int, str, str, str]] = []
    descriptions: Dict[int, str] = {}

    for opt_num, item in sorted(option_map.items()):
        prompt = build_model_prompt(tokenizer, build_candidate_judgment_prompt(obs, opt_num, item))
        prior_prompt = build_model_prompt(tokenizer, build_neutral_candidate_prior_prompt(opt_num, item))
        descriptions[int(opt_num)] = describe_option(item)
        prompt_pairs.append((int(opt_num), prompt, " Yes", "yes"))
        prompt_pairs.append((int(opt_num), prompt, " No", "no"))
        prior_pairs.append((int(opt_num), prior_prompt, " Yes", "yes"))
        prior_pairs.append((int(opt_num), prior_prompt, " No", "no"))

    raw_scores = score_continuations(model, tokenizer, [x[1] for x in prompt_pairs], [x[2] for x in prompt_pairs])
    prior_scores = score_continuations(model, tokenizer, [x[1] for x in prior_pairs], [x[2] for x in prior_pairs])

    yes_scores: Dict[int, float] = {}
    no_scores: Dict[int, float] = {}
    prior_yes_scores: Dict[int, float] = {}
    prior_no_scores: Dict[int, float] = {}

    for (opt_num, _, _, label), score in zip(prompt_pairs, raw_scores):
        if label == "yes":
            yes_scores[opt_num] = float(score)
        else:
            no_scores[opt_num] = float(score)
    for (opt_num, _, _, label), score in zip(prior_pairs, prior_scores):
        if label == "yes":
            prior_yes_scores[opt_num] = float(score)
        else:
            prior_no_scores[opt_num] = float(score)

    candidate_scores: Dict[int, float] = {}
    for opt_num in sorted(option_map.keys()):
        semantic_margin = yes_scores[opt_num] - no_scores[opt_num]
        prior_margin = prior_yes_scores[opt_num] - prior_no_scores[opt_num]
        candidate_scores[opt_num] = semantic_margin - prior_weight * prior_margin

    ranked = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen = int(ranked[0][0]) if ranked else None
    if len(ranked) >= 2:
        margin = float(ranked[0][1] - ranked[1][1])
    elif len(ranked) == 1:
        margin = float("inf")
    else:
        margin = 0.0

    return SemanticChoiceResult(
        chosen_option=chosen,
        candidate_scores={k: round(v, 6) for k, v in candidate_scores.items()},
        yes_scores={k: round(v, 6) for k, v in yes_scores.items()},
        no_scores={k: round(v, 6) for k, v in no_scores.items()},
        prior_yes_scores={k: round(v, 6) for k, v in prior_yes_scores.items()},
        prior_no_scores={k: round(v, 6) for k, v in prior_no_scores.items()},
        candidate_descriptions=descriptions,
        margin_top1_top2=margin,
    )


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def mean(key: str) -> float:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        return round(sum(vals) / max(1, len(vals)), 4)

    return {
        "episodes": len(rows),
        "mean_raw_score": mean("raw_score"),
        "raw_success_rate": mean("success"),
        "mean_steps": mean("steps"),
        "mean_invalid_actions": mean("invalid_actions"),
        "mean_constraint_violations": mean("constraint_violations"),
        "mean_scoring_errors": mean("scoring_errors"),
        "mean_margin_top1_top2": mean("margin_top1_top2"),
        "mean_fallback_steps": mean("fallback_steps"),
    }


def group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {k: summarize_rows(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}


def evaluate_model_hf(mod, model_name: str, episodes, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    model, tokenizer = load_hf_model(
        model_name,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    rows: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    scoring_errors_total = 0
    fallback_steps_total = 0
    option_margin_values: List[float] = []
    t0 = time.time()

    for idx, ep in enumerate(episodes, start=1):
        env = mod.RuleShiftEnv(ep)
        episode_scoring_errors = 0
        episode_fallback_steps = 0
        episode_margin_sum = 0.0
        episode_margin_count = 0

        while not env.done:
            obs = env.observe()
            option_map = candidate_option_map(obs.get("candidate_actions"))
            error_text = ""
            scoring_error = False
            try:
                choice = choose_option_semantic_scoring(model, tokenizer, obs, prior_weight=args.prior_weight)
            except Exception as exc:
                choice = SemanticChoiceResult(None, {}, {}, {}, {}, {}, {}, 0.0)
                scoring_error = True
                error_text = f"{type(exc).__name__}: {exc}"
                episode_scoring_errors += 1
                scoring_errors_total += 1

            chosen_action = None
            chosen_option = choice.chosen_option
            used_fallback = False
            if chosen_option is not None and chosen_option in option_map:
                item = option_map[chosen_option]
                chosen_action = mod.Action.from_any(item) if isinstance(item, dict) else item
            else:
                used_fallback = True
                episode_fallback_steps += 1
                fallback_steps_total += 1
                if option_map:
                    wait_item = None
                    for item in option_map.values():
                        if isinstance(item, dict) and str(item.get("action", "")).upper() == "WAIT":
                            wait_item = item
                            break
                    if wait_item is not None:
                        chosen_action = mod.Action.from_any(wait_item)
                        chosen_option = int(wait_item.get("option", 0) or 0)
                    else:
                        first_key = sorted(option_map.keys())[0]
                        first_item = option_map[first_key]
                        chosen_action = mod.Action.from_any(first_item) if isinstance(first_item, dict) else first_item
                        chosen_option = int(first_key)
                else:
                    chosen_action = mod.Action("WAIT", "none")
                    chosen_option = 0
                if not scoring_error:
                    error_text = error_text or "no_valid_option_selected"

            margin = choice.margin_top1_top2
            if math.isfinite(margin):
                episode_margin_sum += margin
                episode_margin_count += 1
                option_margin_values.append(margin)

            result = env.apply(chosen_action)
            if args.save_traces:
                traces.append(
                    {
                        "model": model_name,
                        "episode_id": ep.episode_id,
                        "family": ep.family,
                        "difficulty": ep.difficulty,
                        "step": obs.get("step"),
                        "chosen_option": chosen_option,
                        "used_fallback": used_fallback,
                        "chosen_action": {"action": chosen_action.action, "target": chosen_action.target},
                        "candidate_scores": choice.candidate_scores,
                        "yes_scores": choice.yes_scores,
                        "no_scores": choice.no_scores,
                        "prior_yes_scores": choice.prior_yes_scores,
                        "prior_no_scores": choice.prior_no_scores,
                        "candidate_descriptions": choice.candidate_descriptions,
                        "margin_top1_top2": None if not math.isfinite(margin) else round(margin, 6),
                        "scoring_error": scoring_error,
                        "scoring_error_text": error_text,
                        "candidate_actions": obs.get("candidate_actions", []),
                        "result": result,
                    }
                )

        score = env.score()
        rows.append(
            {
                "model": model_name,
                "episode_id": ep.episode_id,
                "family": ep.family,
                "difficulty": ep.difficulty,
                "raw_score": round(float(score.final_score), 6),
                "success": 1 if env.success else 0,
                "steps": env.step_id,
                "invalid_actions": env.invalid_actions,
                "constraint_violations": env.constraint_violations,
                "scoring_errors": episode_scoring_errors,
                "fallback_steps": episode_fallback_steps,
                "margin_top1_top2": round((episode_margin_sum / episode_margin_count), 6) if episode_margin_count else 0.0,
            }
        )

        if not args.no_progress and idx % max(1, args.progress_every) == 0:
            avg_margin = sum(option_margin_values) / max(1, len(option_margin_values)) if option_margin_values else 0.0
            total_steps = max(1, sum(r["steps"] for r in rows))
            print(
                f"[{model_name}] {idx}/{len(episodes)} episodes done | "
                f"fallback_step_rate_so_far={fallback_steps_total / total_steps:.3f} | "
                f"avg_margin_so_far={avg_margin:.3f}",
                flush=True,
            )

    summary = {
        "model": model_name,
        "backend": "hf_semantic_yesno_scoring",
        "num_episodes": len(rows),
        "elapsed_seconds": round(time.time() - t0, 3),
        "overall": summarize_rows(rows),
        "by_family": group_rows(rows, "family"),
        "by_difficulty": group_rows(rows, "difficulty"),
        "scoring_errors_total": scoring_errors_total,
        "fallback_steps_total": fallback_steps_total,
        "fallback_step_rate": round(fallback_steps_total / max(1, sum(r["steps"] for r in rows)), 6),
        "mean_margin_top1_top2_global": round(sum(option_margin_values) / max(1, len(option_margin_values)), 6)
        if option_margin_values else 0.0,
    }

    model_slug = slugify_model_name(model_name)
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    write_csv(model_dir / f"{model_slug}.episodes.csv", rows)
    (model_dir / f"{model_slug}.report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_traces:
        with (model_dir / f"{model_slug}.traces.jsonl").open("w", encoding="utf-8") as f:
            for item in traces:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    del model, tokenizer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Qwen3-0.6B and Qwen3-1.7B by semantic yes/no candidate scoring.")
    ap.add_argument("--generator-script", type=str, required=True)
    ap.add_argument("--dataset-path", type=str, default="")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--preset", type=str, default="qwen3_small", choices=sorted(OFFICIAL_QWEN_PRESETS.keys()))
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--difficulty-levels", type=str, default="1,2,3,4")
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--num-episodes", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=20260327)
    ap.add_argument("--prior-weight", type=float, default=0.5)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--save-traces", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    mod = load_generator_module(args.generator_script)
    episodes = load_or_generate_episodes(mod, args)
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    leaderboard_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}

    for model_name in OFFICIAL_QWEN_PRESETS[args.preset]:
        print(f"=== Evaluating {model_name} on {len(episodes)} episodes ===", flush=True)
        summary = evaluate_model_hf(mod, model_name, episodes, outdir, args)
        summaries[model_name] = summary
        leaderboard_rows.append(
            {
                "model": model_name,
                "backend": summary["backend"],
                "episodes": summary["num_episodes"],
                "mean_raw_score": summary["overall"]["mean_raw_score"],
                "raw_success_rate": summary["overall"]["raw_success_rate"],
                "mean_margin_top1_top2": summary["overall"]["mean_margin_top1_top2"],
                "fallback_step_rate": summary["fallback_step_rate"],
                "scoring_errors_total": summary["scoring_errors_total"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )

    leaderboard_rows = sorted(
        leaderboard_rows,
        key=lambda r: (-float(r["mean_raw_score"]), -float(r["raw_success_rate"]), r["model"]),
    )
    write_csv(outdir / "leaderboard.csv", leaderboard_rows)
    (outdir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "run_manifest.json").write_text(
        json.dumps(
            {
                "generator_script": str(Path(args.generator_script).resolve()),
                "dataset_path": args.dataset_path,
                "output_dir": str(outdir.resolve()),
                "models": OFFICIAL_QWEN_PRESETS[args.preset],
                "episodes": len(episodes),
                "backend": "hf_semantic_yesno_scoring",
                "prior_weight": args.prior_weight,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(outdir),
                "models": OFFICIAL_QWEN_PRESETS[args.preset],
                "episodes": len(episodes),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
