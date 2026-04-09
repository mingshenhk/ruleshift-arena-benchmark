from __future__ import annotations

"""RuleShift Arena competition generator.

This minimal entrypoint keeps only the two CLI modes that matter for submission prep:
export-dataset and export-splits. The module still exposes the environment and helpers
needed by external evaluators.
"""

import argparse
import json
import os
import random
import re
import statistics
import textwrap
import time
import sys
import hashlib
from collections import Counter, defaultdict
import difflib
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib import request, error


class Family(str, Enum):
    GOAL_MAINTENANCE = "goal_maintenance"
    PLANNING = "planning"
    INHIBITORY_CONTROL = "inhibitory_control"
    COGNITIVE_FLEXIBILITY = "cognitive_flexibility"
    CONFLICT_RESOLUTION = "conflict_resolution"
    WORKING_MEMORY = "working_memory"


class ActionType(str, Enum):
    MOVE = "MOVE"
    PICK = "PICK"
    DROP = "DROP"
    USE = "USE"
    INSPECT = "INSPECT"
    REPORT = "REPORT"
    WAIT = "WAIT"


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressTracker:
    def __init__(self, total: int, desc: str = "Progress", enabled: bool = True):
        self.total = max(0, int(total))
        self.desc = desc
        self.enabled = enabled and self.total > 0
        self.count = 0
        self.start = time.time()
        self.last_print = 0.0
        self.bar_width = 28

    def update(self, step: int = 1) -> None:
        if not self.enabled:
            return
        self.count = min(self.total, self.count + step)
        now = time.time()
        if self.count < self.total and now - self.last_print < 0.12:
            return
        self.last_print = now
        elapsed = max(1e-9, now - self.start)
        rate = self.count / elapsed
        remaining = 0.0 if self.count <= 0 or rate <= 0 else (self.total - self.count) / rate
        ratio = 0.0 if self.total <= 0 else self.count / self.total
        filled = int(self.bar_width * ratio)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        pct = ratio * 100
        msg = (
            f"\r{self.desc:<20} [{bar}] {self.count:>4}/{self.total:<4} "
            f"{pct:6.2f}% | elapsed {format_eta(elapsed)} | eta {format_eta(remaining)}"
        )
        print(msg, end="", flush=True, file=sys.stderr)
        if self.count >= self.total:
            print(flush=True, file=sys.stderr)


def iter_with_progress(items, desc: str, enabled: bool = True):
    try:
        total = len(items)
    except Exception:
        items = list(items)
        total = len(items)
    tracker = ProgressTracker(total=total, desc=desc, enabled=enabled)
    for item in items:
        yield item
        tracker.update()


@dataclass
class Action:
    action: str
    target: str

    def normalized(self) -> "Action":
        return Action(action=self.action.upper().strip(), target=str(self.target).strip())

    @staticmethod
    def from_any(obj: Any) -> "Action":
        if isinstance(obj, Action):
            return obj.normalized()
        if isinstance(obj, dict):
            if "action" not in obj or "target" not in obj:
                raise ValueError("action dict must contain 'action' and 'target'")
            return Action(str(obj["action"]), str(obj["target"])).normalized()
        raise TypeError(f"cannot convert object of type {type(obj).__name__} into Action")


@dataclass
class Episode:
    episode_id: str
    seed: int
    family: str
    difficulty: int
    title: str
    instructions: str
    max_steps: int
    initial_state: Dict[str, Any]
    hidden_state: Dict[str, Any]
    scoring_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    step: int
    pre_observation: Dict[str, Any]
    action: Dict[str, Any]
    result: Dict[str, Any]
    post_observation: Dict[str, Any]


@dataclass
class ScoreBreakdown:
    goal_score: float
    constraint_score: float
    efficiency_score: float
    control_score: float
    final_score: float
    success: bool
    failure_modes: List[str]
    mission_completion_score: float = 0.0
    integrity_preservation_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def add_decoy_branches(rng: random.Random, room_contents: Dict[str, List[str]], doors: Dict[str, Dict[str, Dict[str, Any]]], anchors: List[str], difficulty: int, family_tag: str) -> List[str]:
    """Add non-essential branches and items to increase choice diversity without changing the gold solution."""
    decoy_names = {
        "goal": ["notice_board", "break_corner", "mail_slot", "side_archive", "review_desk"],
        "flex": ["gray_lane", "amber_nook", "teal_hall", "inspection_bay", "signal_room"],
        "inhib": ["poster_alcove", "supply_cache", "inspection_bay", "side_bench", "flashy_kiosk"],
        "plan": ["break_room", "manual_shelf", "tool_closet", "service_bay", "idle_terminal"],
        "conflict": ["time_board", "shortcut_map", "wash_station", "side_lab", "orange_bypass"],
        "wm": ["memo_wall", "quiet_nook", "spare_locker", "color_poster", "buffer_alcove"],
    }
    decoy_items = ["flyer", "memo_card", "empty_box", "audit_stamp", "spare_clip", "unused_key"]
    added = []
    names = decoy_names.get(family_tag, decoy_names["goal"])
    n = min(len(anchors), max(1, difficulty))
    if not anchors:
        return added
    if n == 1:
        chosen_anchors = [anchors[0]]
    else:
        chosen_indices = []
        for i in range(n):
            pos = round(i * (len(anchors) - 1) / max(1, n - 1))
            if pos not in chosen_indices:
                chosen_indices.append(pos)
        chosen_anchors = [anchors[i] for i in chosen_indices]
        while len(chosen_anchors) < n:
            chosen_anchors.append(anchors[len(chosen_anchors) % len(anchors)])
    for idx, anchor in enumerate(chosen_anchors[:n]):
        base = names[idx % len(names)]
        room = base
        suffix = 1
        while room in room_contents:
            suffix += 1
            room = f"{base}_{suffix}"
        room_contents[room] = [rng.choice(decoy_items)]
        doors.setdefault(anchor, {})[room] = {"color": rng.choice(["gray", "yellow", "orange", "silver"]), "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault(room, {})[anchor] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
        added.append(room)
    return added


class RuleShiftEnv:
    def __init__(self, episode: Episode):
        self.episode = episode
        self.state = json.loads(json.dumps(episode.initial_state))
        self.hidden = json.loads(json.dumps(episode.hidden_state))
        self.max_steps = episode.max_steps
        self.step_id = 0
        self.done = False
        self.success = False
        self.history: List[StepRecord] = []

        self.failure_modes: List[str] = []
        self.invalid_actions = 0
        self.constraint_violations = 0
        self.perseveration_errors = 0
        self.habit_intrusions = 0
        self.dead_end_moves = 0
        self.goal_completed_step: Optional[int] = None
        self._illegal_drop_to_goal = False

    def observe(self) -> Dict[str, Any]:
        self._maybe_trigger_switch_event()
        visible_doors = json.loads(json.dumps(self.state["doors"]))
        visible_active_rule = self.state["active_rule"]

        if self.episode.family == Family.WORKING_MEMORY.value:
            for _, edges in visible_doors.items():
                for _, info in edges.items():
                    info.pop("requires", None)
            visible_active_rule = "hidden_memory_rule"

        visible = {
            "episode_id": self.episode.episode_id,
            "family": self.episode.family,
            "difficulty": self.episode.difficulty,
            "step": self.step_id,
            "max_steps": self.max_steps,
            "title": self.episode.title,
            "instructions": self.episode.instructions,
            "current_room": self.state["current_room"],
            "inventory": list(self.state["inventory"]),
            "room_contents": {room: list(items) for room, items in self.state["room_contents"].items()},
            "doors": visible_doors,
            "goal": json.loads(json.dumps(self.state["goal"])),
            "active_rule": visible_active_rule,
            "allowed_actions": [a.value for a in ActionType],
        }
        switch_event = self.hidden.get("switch_event")
        if switch_event and self.state.get("switch_announced", False):
            visible["system_alert"] = switch_event["message"]
        if self.episode.family == Family.INHIBITORY_CONTROL.value:
            visible["tempting_hint"] = self.hidden.get("tempting_hint", "")
        if self.hidden.get("briefing_note") and self.step_id == 0:
            visible["briefing_note"] = self.hidden.get("briefing_note")
        if self.hidden.get("priority_note"):
            visible["priority_note"] = self.hidden.get("priority_note")
        if self.hidden.get("focus_alert") and self.state.get("switch_announced", False):
            visible["focus_alert"] = self.hidden.get("focus_alert")
        visible["candidate_actions"] = self.numbered_candidate_actions()
        visible["compact_map"] = self.compact_map()
        return visible

    def compact_map(self) -> Dict[str, Any]:
        room_contents = self.state["room_contents"]
        doors = self.state["doors"]
        return {
            room: {
                "items": list(room_contents.get(room, [])),
                "neighbors": sorted(list(doors.get(room, {}).keys())),
            }
            for room in sorted(room_contents.keys())
        }

    def _can_use_item_now(self, item: str) -> bool:
        current_room = self.state["current_room"]
        doors = self.state["doors"]

        if item == "battery_pack":
            return current_room == "charging_station" and not self.state.get("flags", {}).get("battery_charged", False)
        if item == "scanner":
            return current_room == "inspection_room" and not self.state.get("flags", {}).get("package_verified", False)

        if self.episode.family == Family.WORKING_MEMORY.value and item.endswith("_badge"):
            for _, info in doors.get(current_room, {}).items():
                if not info.get("unlocked", False) and info.get("requires"):
                    return info.get("requires") == item

        for _, info in doors.get(current_room, {}).items():
            if info.get("requires") == item and not info.get("unlocked", False):
                return True
        return False

    def _action_sort_key(self, action: Action) -> Tuple[int, str]:
        goal = self.state.get("goal", {})
        is_goal_drop = (
            action.action.upper() == "DROP"
            and action.target == goal.get("item")
            and self.state.get("current_room") == goal.get("target_room")
        )
        if is_goal_drop:
            return (0, action.target)
        priority = {
            "MOVE": 1,
            "PICK": 2,
            "USE": 3,
            "DROP": 4,
            "INSPECT": 5,
            "REPORT": 6,
            "WAIT": 7,
        }
        return (priority.get(action.action.upper(), 99), action.target)

    def _progress_like_actions(self) -> List[Action]:
        current_room = self.state["current_room"]
        room_contents = self.state["room_contents"]
        inventory = self.state["inventory"]
        doors = self.state["doors"]
        actions: List[Action] = []
        for target_room in sorted(doors.get(current_room, {})):
            actions.append(Action("MOVE", target_room))
        for item in sorted(room_contents.get(current_room, [])):
            actions.append(Action("PICK", item))
        for item in sorted(inventory):
            if self._can_use_item_now(item):
                actions.append(Action("USE", item))
        goal = self.state.get("goal", {})
        for item in sorted(inventory):
            if item == goal.get("item") and current_room == goal.get("target_room"):
                actions.append(Action("DROP", item))
        return actions


    def candidate_actions(self) -> List[Action]:
        current_room = self.state["current_room"]
        room_contents = self.state["room_contents"]
        inventory = self.state["inventory"]
        doors = self.state["doors"]
        goal = self.state.get("goal", {})
        family = self.episode.family
        flags = self.state.setdefault("flags", {})
        seen_meta = set(flags.get("seen_meta_queries", []))
        no_progress_streak = int(flags.get("no_progress_streak", 0))
        consecutive_meta_actions = int(flags.get("consecutive_meta_actions", 0))

        goal_item = goal.get("item")
        target_room = goal.get("target_room")
        carrying_goal = goal_item in inventory if goal_item else False
        current_room_items = list(room_contents.get(current_room, []))
        current_doors = doors.get(current_room, {})

        def _meta_sig(kind: str, target: str) -> str:
            inv_key = ",".join(sorted(inventory))
            return f"{kind}:{target}@{current_room}|inv:{inv_key}"

        def _meta_allowed(kind: str, target: str) -> bool:
            return _meta_sig(kind, target) not in seen_meta

        def _door_is_tempting(info: Dict[str, Any]) -> bool:
            return bool(
                info.get("decoy")
                or info.get("trap")
                or info.get("unsafe_shortcut")
                or info.get("misleading")
                or info.get("premature")
            )

        def _is_useful_pick(item: str) -> bool:
            if item == goal_item:
                return True
            useful_named = {"battery_pack", "scanner", "access_card", "relay_key", "blue_token", "blue_badge", "green_badge", "red_badge"}
            if item in useful_named:
                return True
            for info in current_doors.values():
                if info.get("requires") == item and not info.get("unlocked", False):
                    return True
            if family == Family.PLANNING.value and item in {"battery_pack", "scanner", "access_card"}:
                return True
            if family == Family.WORKING_MEMORY.value and item.endswith("_badge"):
                return True
            return False

        def _path_index(room: str) -> int:
            path = list(self.hidden.get("main_path", []))
            return path.index(room) if room in path else -1

        move_candidates: List[Action] = []
        blocked_moves: List[Action] = []
        for target_room2, info in sorted(current_doors.items()):
            blocked, _ = self._is_move_blocked(current_room, target_room2, info)
            action = Action("MOVE", target_room2)
            if blocked:
                blocked_moves.append(action)
            else:
                move_candidates.append(action)

        pick_candidates: List[Action] = [Action("PICK", item) for item in sorted(current_room_items)]
        use_candidates: List[Action] = [Action("USE", item) for item in sorted(inventory) if self._can_use_item_now(item)]
        drop_candidates: List[Action] = [
            Action("DROP", item) for item in sorted(inventory)
            if item == goal_item and current_room == target_room
        ]

        # If the goal item or an immediately useful tool is visible, still keep a small amount
        # of controlled branching. Otherwise many episodes collapse into deterministic one-option
        # trajectories, which hurts discriminatory power.
        immediate_picks = [a for a in pick_candidates if _is_useful_pick(a.target)]
        if goal_item and goal_item in current_room_items and goal_item not in inventory:
            immediate_picks = [Action("PICK", goal_item)] + [a for a in immediate_picks if a.target != goal_item]

        def _branch_moves_after_pick() -> List[Action]:
            if not move_candidates and not blocked_moves:
                return []
            diff = int(getattr(self.episode, "difficulty", 1) or 1)
            if family == Family.GOAL_MAINTENANCE.value:
                distractor_rooms = set(self.hidden.get("distractor_rooms", []))
                main_moves = [m for m in move_candidates if m.target not in distractor_rooms and not _door_is_tempting(current_doors.get(m.target, {}))]
                decoy_moves = [m for m in move_candidates if m.target in distractor_rooms or _door_is_tempting(current_doors.get(m.target, {}))]
                extra: List[Action] = []
                if main_moves:
                    extra.extend(main_moves[:1])
                if decoy_moves:
                    extra.extend(decoy_moves[:2 if diff >= 2 else 1])
                tempting_blocked = [m for m in blocked_moves if _door_is_tempting(current_doors.get(m.target, {}))]
                if diff >= 4 and tempting_blocked:
                    extra.extend(tempting_blocked[:1])
                return extra
            if family == Family.INHIBITORY_CONTROL.value:
                safe_moves = [m for m in move_candidates if not current_doors.get(m.target, {}).get("trap")]
                trap_moves = [m for m in move_candidates if current_doors.get(m.target, {}).get("trap")]
                blocked_traps = [m for m in blocked_moves if current_doors.get(m.target, {}).get("trap")]
                tempting_exec = [m for m in trap_moves if current_doors.get(m.target, {}).get("misleading")]
                extra: List[Action] = []
                if safe_moves:
                    extra.extend(safe_moves[:1])
                if tempting_exec:
                    extra.extend(tempting_exec[:2 if diff >= 3 else 1])
                remaining_traps = [m for m in trap_moves if m not in extra]
                if remaining_traps:
                    extra.extend(remaining_traps[:2 if diff >= 2 else 1])
                if diff >= 4 and blocked_traps:
                    extra.extend(blocked_traps[:1])
                return extra
            if family == Family.CONFLICT_RESOLUTION.value:
                safe_moves = [m for m in move_candidates if not current_doors.get(m.target, {}).get("unsafe_shortcut")]
                risky_moves = [m for m in move_candidates if current_doors.get(m.target, {}).get("unsafe_shortcut")]
                risky_blocked = [m for m in blocked_moves if current_doors.get(m.target, {}).get("unsafe_shortcut")]
                misleading_risky = [m for m in (move_candidates + blocked_moves) if current_doors.get(m.target, {}).get("unsafe_shortcut") and current_doors.get(m.target, {}).get("misleading")]
                extra: List[Action] = []
                if safe_moves:
                    extra.extend(safe_moves[:1])
                if risky_moves:
                    extra.extend(risky_moves[:2 if diff >= 2 else 1])
                if diff >= 3 and misleading_risky:
                    extra.extend(misleading_risky[:1])
                if diff >= 4 and risky_blocked:
                    extra.extend(risky_blocked[:1])
                return extra
            if family == Family.COGNITIVE_FLEXIBILITY.value:
                non_decoy = [m for m in move_candidates if not current_doors.get(m.target, {}).get("decoy")]
                decoy_exec = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")]
                gray_exec = [m for m in decoy_exec if current_doors.get(m.target, {}).get("color") == "gray"]
                review_exec = [m for m in decoy_exec if current_doors.get(m.target, {}).get("color") in {"cyan", "violet", "amber"}]
                blocked_alts = [m for m in blocked_moves if current_doors.get(m.target, {}).get("color") in {"red", "blue"} or current_doors.get(m.target, {}).get("decoy")]
                extra: List[Action] = []
                if non_decoy:
                    extra.extend(non_decoy[:2])
                if gray_exec:
                    extra.extend(gray_exec[:1])
                if review_exec:
                    extra.extend(review_exec[:2 if diff >= 3 else 1])
                remaining_decoy = [m for m in decoy_exec if m not in extra]
                if remaining_decoy:
                    extra.extend(remaining_decoy[:1])
                if len(extra) < (3 if diff >= 2 else 2) and blocked_alts:
                    extra.extend(blocked_alts[:1 if diff <= 2 else 2])
                return extra
            if family == Family.WORKING_MEMORY.value:
                non_decoy = [m for m in move_candidates if not current_doors.get(m.target, {}).get("decoy")]
                decoy = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")]
                extra = non_decoy[:2]
                if decoy:
                    extra.extend(decoy[:1])
                return extra
            if family == Family.PLANNING.value:
                decoy_free = [m for m in move_candidates if not current_doors.get(m.target, {}).get("decoy")]
                decoy = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")]
                preview_like = [m for m in move_candidates if m.target in {"preview_vault", "service_tunnel", "tool_room", "locker", "credential_kiosk", "service_console", "seal_review"}]
                extra: List[Action] = decoy_free[:2]
                if preview_like:
                    extra.extend(preview_like[:2 if diff >= 2 else 1])
                if decoy:
                    extra.extend(decoy[:1])
                tempting_blocked = [m for m in blocked_moves if current_doors.get(m.target, {}).get("requires") or current_doors.get(m.target, {}).get("decoy")]
                if diff >= 3 and tempting_blocked:
                    extra.extend(tempting_blocked[:1 if diff == 3 else 2])
                return extra
            return move_candidates[:1] + blocked_moves[:1]

        if family == Family.GOAL_MAINTENANCE.value:
            distractor_rooms = set(self.hidden.get("distractor_rooms", []))
            main_path = list(self.hidden.get("main_path", []))
            distractor_moves = [m for m in move_candidates if m.target in distractor_rooms or current_doors.get(m.target, {}).get("decoy")]
            if carrying_goal:
                preferred = [m for m in move_candidates if m.target not in distractor_rooms and not current_doors.get(m.target, {}).get("decoy")]
                if current_room in main_path:
                    idx = main_path.index(current_room)
                    if idx + 1 < len(main_path):
                        next_room = main_path[idx + 1]
                        next_moves = [m for m in preferred if m.target == next_room]
                        trailing_moves = [m for m in preferred if m.target != next_room]
                        preferred = (next_moves[:1] + trailing_moves[:2]) if next_moves else preferred
                if preferred:
                    tempting_blocked = [m for m in blocked_moves if _door_is_tempting(current_doors.get(m.target, {}))]
                    preferred_limit = 1 if self.episode.difficulty >= 3 else 2
                    move_candidates = preferred[:preferred_limit]
                    if distractor_moves:
                        move_candidates += distractor_moves[:3 if self.episode.difficulty >= 3 else 2 if self.episode.difficulty >= 2 else 1]
                    trailing_safe = [m for m in preferred if m not in move_candidates]
                    if trailing_safe and self.episode.difficulty >= 4:
                        move_candidates += trailing_safe[:1]
                    if self.episode.difficulty >= 4 and tempting_blocked:
                        move_candidates += tempting_blocked[:1]
            elif current_room in distractor_rooms:
                preferred = [m for m in move_candidates if m.target not in distractor_rooms]
                if preferred:
                    move_candidates = preferred + ([] if self.episode.difficulty < 3 else distractor_moves[:1])
            if current_room == "start" and goal_item in current_room_items and self.episode.difficulty <= 1:
                move_candidates = move_candidates

        if family == Family.PLANNING.value:
            if flags.get("package_verified", False):
                move_candidates = [m for m in move_candidates if m.target != "service_tunnel"]
            if not flags.get("package_verified", False):
                move_candidates = [m for m in move_candidates if m.target != "vault"]
            if current_room in {"storage", "locker", "tool_room"} and immediate_picks:
                preferred_targets = {"charging_station", "inspection_room", "lab", "preview_vault", "service_tunnel", "staging_alcove", "diagnostics_nook", "credential_kiosk", "service_console", "seal_review"}
                move_candidates = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy") or m.target in preferred_targets]
            if current_room == "lab" and "scanner" not in inventory and any(m.target == "tool_room" for m in move_candidates):
                move_candidates = [m for m in move_candidates if m.target in {"tool_room", "preview_vault"}]
            if current_room == "storage" and goal_item in inventory and any(m.target == "charging_station" for m in move_candidates):
                preferred = [m for m in move_candidates if m.target == "charging_station"]
                side = [m for m in move_candidates if m.target != "charging_station"]
                move_candidates = preferred[:1] + side[:2]
            if current_room == "tool_room" and "scanner" in inventory and any(m.target == "inspection_room" for m in move_candidates):
                preferred = [m for m in move_candidates if m.target == "inspection_room"]
                side = [m for m in move_candidates if m.target != "inspection_room"]
                move_candidates = preferred[:1] + side[:2]
            if current_room == "inspection_room" and not flags.get("package_verified", False) and "scanner" in inventory:
                move_candidates = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")]
            if use_candidates:
                decoy_free_moves = [m for m in move_candidates if not current_doors.get(m.target, {}).get("decoy")]
                decoy_moves = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")]
                if decoy_free_moves:
                    move_candidates = decoy_free_moves[:2] + decoy_moves[:2 if self.episode.difficulty >= 3 else 1]

        if family == Family.WORKING_MEMORY.value:
            preferred_badge = "blue_badge"
            system_alert = str(self.state.get("system_alert") or "")
            if "GREEN" in system_alert or (self.state.get("switch_applied") and ("green_badge" in current_room_items or "green_badge" in inventory)):
                preferred_badge = "green_badge"
            if immediate_picks:
                allowed = {goal_item, "relay_key", preferred_badge}
                immediate_picks = [a for a in immediate_picks if a.target in allowed]
            if use_candidates:
                allowed_uses = [u for u in use_candidates if u.target in {preferred_badge, "relay_key"}]
                if allowed_uses:
                    use_candidates = allowed_uses
            productive_moves = [m for m in move_candidates if not current_doors.get(m.target, {}).get("decoy")]
            decoy_moves = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")]
            allowed_move_targets = {"corridor", "relay_room", "buffer_room", "anteroom", "memory_vault", "memo_wall", "quiet_nook", "color_poster", "buffer_alcove", "badge_bench", "route_plaque", "badge_mirror"}
            move_candidates = [m for m in move_candidates if m.target in allowed_move_targets or current_doors.get(m.target, {}).get("decoy")]
            if current_room in {"corridor", "relay_room"} and any(x in current_room_items for x in {"relay_key", "green_badge", "blue_badge"}):
                move_candidates = productive_moves[:2] + decoy_moves[:1 if self.episode.difficulty <= 2 else 2]
            elif use_candidates:
                move_candidates = productive_moves[:2] + decoy_moves[:1]
            elif carrying_goal and any(info.get("memory_gate") and not info.get("unlocked", False) for info in current_doors.values()):
                productive = [m for m in move_candidates if not current_doors.get(m.target, {}).get("decoy")][:2]
                tempting = [m for m in move_candidates if current_doors.get(m.target, {}).get("decoy")][:1 if self.episode.difficulty <= 2 else 2]
                move_candidates = productive + tempting

        if family == Family.COGNITIVE_FLEXIBILITY.value:
            switch_active = bool(self.state.get("switch_applied")) or bool(self.state.get("system_alert"))
            route_names = self.hidden.get("route_names", {})
            blue_room = route_names.get("blue")
            if current_room == "start" and switch_active and blue_room and any(m.target == blue_room for m in move_candidates):
                allowed = {blue_room, route_names.get("red"), "gray_staging"}
                move_candidates = [m for m in move_candidates if m.target in allowed or current_doors.get(m.target, {}).get("decoy")]
            if current_room == blue_room and any(m.target == route_names.get("blue_buffer") for m in move_candidates):
                allowed = {route_names.get("blue_buffer"), route_names.get("red"), "review_node", "gray_staging", "blue_review", "mirror_lane"}
                move_candidates = [m for m in move_candidates if m.target in allowed or current_doors.get(m.target, {}).get("decoy")]
            if current_room in {"blue_review", "mirror_lane", "gray_staging", "red_review"}:
                keep = {"vault", blue_room, route_names.get("blue_buffer"), "mirror_lane", "blue_review", "gray_staging", "review_node", route_names.get("red")}
                move_candidates = [m for m in move_candidates if m.target in keep or current_doors.get(m.target, {}).get("decoy")]

        if family == Family.CONFLICT_RESOLUTION.value and carrying_goal:
            safe_moves = [m for m in move_candidates if not current_doors.get(m.target, {}).get("unsafe_shortcut")]
            risky_moves = [m for m in move_candidates if current_doors.get(m.target, {}).get("unsafe_shortcut")]
            risky_blocked = [m for m in blocked_moves if current_doors.get(m.target, {}).get("unsafe_shortcut")]
            if safe_moves:
                move_candidates = safe_moves[:2] + risky_moves[:2 if self.episode.difficulty >= 2 else 1]
                if self.episode.difficulty >= 3:
                    misleading_risky = [m for m in move_candidates if current_doors.get(m.target, {}).get("misleading")]
                    move_candidates += misleading_risky[:1]
                if self.episode.difficulty >= 4 and risky_blocked:
                    move_candidates += risky_blocked[:1]

        progress_candidates: List[Action] = []
        if immediate_picks:
            progress_candidates.extend(immediate_picks)
            progress_candidates.extend(_branch_moves_after_pick())
        else:
            progress_candidates.extend(move_candidates)
            progress_candidates.extend(pick_candidates)
        if use_candidates:
            # USE actions should come before exploratory moves once a prerequisite is satisfiable.
            progress_candidates = use_candidates + [a for a in progress_candidates if (a.action.upper(), a.target) not in {(u.action.upper(), u.target) for u in use_candidates}]
        progress_candidates.extend(drop_candidates)

        # Preserve one near-correct but premature blocked action in planning / memory / flexibility / inhibition families.
        if family in {Family.PLANNING.value, Family.WORKING_MEMORY.value, Family.COGNITIVE_FLEXIBILITY.value, Family.INHIBITORY_CONTROL.value} and blocked_moves:
            tempting_blocked = [
                m for m in blocked_moves
                if current_doors.get(m.target, {}).get("requires") in set(inventory)
                or current_doors.get(m.target, {}).get("memory_gate")
                or current_doors.get(m.target, {}).get("sequence_gate")
                or current_doors.get(m.target, {}).get("trap")
                or m.target in {goal.get("target_room"), "memory_vault", "vault", "lab"}
            ]
            if tempting_blocked:
                progress_candidates.extend(tempting_blocked[:1])

        # If nothing useful remains, allow blocked moves as last-resort visibility into local structure.
        if not progress_candidates and blocked_moves:
            progress_candidates.extend(blocked_moves[:1])

        dedup_progress: List[Action] = []
        seen_progress = set()
        for action in progress_candidates:
            key = (action.action.upper(), action.target)
            if key not in seen_progress:
                seen_progress.add(key)
                dedup_progress.append(Action(*key))

        def _candidate_rank(action: Action) -> Tuple[int, int, str]:
            name = action.action.upper()
            if name == "DROP" and action.target == goal_item and current_room == target_room:
                return (0, 0, action.target)
            if name == "USE":
                return (1, 0, action.target)
            if name == "PICK" and action.target == goal_item:
                return (2, 0, action.target)
            if name == "PICK" and _is_useful_pick(action.target):
                return (3, 0, action.target)
            if name == "MOVE":
                info = current_doors.get(action.target, {})
                penalty = 0
                if info.get("decoy"):
                    penalty += 3
                if info.get("trap"):
                    penalty += 4
                if info.get("unsafe_shortcut"):
                    penalty += 5
                if family == Family.GOAL_MAINTENANCE.value:
                    idx = _path_index(action.target)
                    if idx >= 0:
                        return (4, idx, action.target)
                    penalty += 3
                return (4 + penalty, 0, action.target)
            base = {"PICK": 6, "REPORT": 7, "INSPECT": 8, "WAIT": 9}.get(name, 10)
            return (base, 0, action.target)

        dedup_progress.sort(key=_candidate_rank)
        progress_candidates = dedup_progress

        minimum_progress_choices = 3 if family in {Family.PLANNING.value, Family.COGNITIVE_FLEXIBILITY.value, Family.CONFLICT_RESOLUTION.value} or self.episode.difficulty >= 3 else 2
        if len(progress_candidates) < minimum_progress_choices:
            existing_progress = {(x.action.upper(), x.target) for x in progress_candidates}
            rescue_pool: List[Action] = []

            tempting_exec = [a for a in move_candidates if (a.action.upper(), a.target) not in existing_progress and _door_is_tempting(current_doors.get(a.target, {}))]
            safe_exec = [a for a in move_candidates if (a.action.upper(), a.target) not in existing_progress and not _door_is_tempting(current_doors.get(a.target, {}))]
            tempting_blocked = [a for a in blocked_moves if (a.action.upper(), a.target) not in existing_progress and _door_is_tempting(current_doors.get(a.target, {}))]
            useful_picks = [a for a in pick_candidates if (a.action.upper(), a.target) not in existing_progress and _is_useful_pick(a.target)]
            decoy_picks = [a for a in pick_candidates if (a.action.upper(), a.target) not in existing_progress and not _is_useful_pick(a.target)]
            late_uses = [a for a in use_candidates if (a.action.upper(), a.target) not in existing_progress]

            rescue_pool.extend(tempting_exec)
            rescue_pool.extend(safe_exec)
            if family in {Family.PLANNING.value, Family.COGNITIVE_FLEXIBILITY.value, Family.CONFLICT_RESOLUTION.value, Family.INHIBITORY_CONTROL.value}:
                rescue_pool.extend(tempting_blocked)
            rescue_pool.extend(useful_picks)
            rescue_pool.extend(decoy_picks)
            rescue_pool.extend(late_uses)

            for action in rescue_pool:
                progress_candidates.append(action.normalized())
                if len({(x.action.upper(), x.target) for x in progress_candidates}) >= minimum_progress_choices:
                    break
            dedup_progress = []
            seen_progress = set()
            for action in progress_candidates:
                key = (action.action.upper(), action.target)
                if key not in seen_progress:
                    seen_progress.add(key)
                    dedup_progress.append(Action(*key))
            dedup_progress.sort(key=_candidate_rank)
            progress_candidates = dedup_progress

        family_minimum_choices = {
            Family.GOAL_MAINTENANCE.value: 3,
            Family.INHIBITORY_CONTROL.value: 3,
            Family.CONFLICT_RESOLUTION.value: 3,
            Family.COGNITIVE_FLEXIBILITY.value: 4,
            Family.PLANNING.value: 4,
            Family.WORKING_MEMORY.value: 3,
        }
        minimum_progress_choices = max(minimum_progress_choices, family_minimum_choices.get(family, 2 if self.episode.difficulty <= 2 else 3))
        if len(progress_candidates) < minimum_progress_choices:
            existing_progress = {(x.action.upper(), x.target) for x in progress_candidates}
            filler_pool: List[Action] = []
            filler_pool.extend([a for a in move_candidates if (a.action.upper(), a.target) not in existing_progress])
            filler_pool.extend([a for a in pick_candidates if (a.action.upper(), a.target) not in existing_progress])
            filler_pool.extend([a for a in blocked_moves if (a.action.upper(), a.target) not in existing_progress and (_door_is_tempting(current_doors.get(a.target, {})) or family != Family.WORKING_MEMORY.value)])
            for action in filler_pool:
                progress_candidates.append(action.normalized())
                if len({(x.action.upper(), x.target) for x in progress_candidates}) >= minimum_progress_choices:
                    break
            dedup_progress = []
            seen_progress = set()
            for action in progress_candidates:
                key = (action.action.upper(), action.target)
                if key not in seen_progress:
                    seen_progress.add(key)
                    dedup_progress.append(Action(*key))
            dedup_progress.sort(key=_candidate_rank)
            progress_candidates = dedup_progress

        meta_candidates: List[Action] = []
        allow_meta_with_progress = (
            bool(progress_candidates)
            and no_progress_streak <= 0
            and consecutive_meta_actions <= 0
            and len(progress_candidates) > 1
            and not immediate_picks
            and not use_candidates
            and not any(a.action.upper() == "DROP" for a in progress_candidates)
        )
        if allow_meta_with_progress:
            if _meta_allowed("REPORT", "goal"):
                meta_candidates.append(Action("REPORT", "goal"))
        elif not progress_candidates:
            if _meta_allowed("REPORT", "goal"):
                meta_candidates.append(Action("REPORT", "goal"))
            if no_progress_streak <= 0 and _meta_allowed("INSPECT", current_room):
                meta_candidates.append(Action("INSPECT", current_room))
            if not meta_candidates:
                meta_candidates.append(Action("WAIT", "none"))

        if family == Family.WORKING_MEMORY.value:
            meta_candidates = []
        max_meta = 1 if progress_candidates else 2
        candidates: List[Action] = progress_candidates + meta_candidates[:max_meta]
        dedup: List[Action] = []
        seen = set()
        for action in candidates:
            key = (action.action.upper(), action.target)
            if key not in seen:
                seen.add(key)
                dedup.append(Action(*key))
        return dedup

    def numbered_candidate_actions(self) -> List[Dict[str, Any]]:
        progress_keys = {(a.action, a.target) for a in self._progress_like_actions()}
        return [
            {
                "option": idx + 1,
                "action": action.action,
                "target": action.target,
                "kind": "progress" if (action.action, action.target) in progress_keys else "meta",
            }
            for idx, action in enumerate(self.candidate_actions())
        ]

    def apply(self, action: Action) -> Dict[str, Any]:
        if self.done:
            return {"ok": False, "message": "episode already finished"}

        pre = self.observe()
        normalized = action.normalized()
        result = self._transition(normalized)
        self.step_id += 1
        self._check_terminal()
        post = self.observe() if not self.done else self._terminal_observation()
        self.history.append(
            StepRecord(
                step=pre["step"],
                pre_observation=pre,
                action=asdict(normalized),
                result=result,
                post_observation=post,
            )
        )
        return result

    def _terminal_observation(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "step": self.step_id,
            "done": True,
            "success": self.success,
            "current_room": self.state["current_room"],
            "inventory": list(self.state["inventory"]),
            "room_contents": {room: list(items) for room, items in self.state["room_contents"].items()},
            "goal": json.loads(json.dumps(self.state["goal"])),
        }

    def score(self) -> ScoreBreakdown:
        optimal_steps = int(self.episode.scoring_metadata["optimal_steps"])
        flags = self.state.get("flags", {})
        family = self.episode.family
        goal = self.state["goal"]
        room_contents = self.state["room_contents"]
        inventory = self.state["inventory"]

        mission_completed = False
        if goal["type"] == "deliver_item":
            mission_completed = goal["item"] in room_contents.get(goal["target_room"], [])
        elif goal["type"] == "reach_room_with_item":
            required_item = goal.get("required_item")
            mission_completed = self.state["current_room"] == goal["target_room"] and (required_item is None or required_item in inventory)
        mission_completion_score = 1.0 if mission_completed else 0.0
        integrity_preservation_score = 0.0 if flags.get("sample_contaminated", False) else 1.0

        if family == Family.CONFLICT_RESOLUTION.value:
            goal_score = mission_completion_score
        else:
            goal_score = 1.0 if self.success else 0.0
            if flags.get("sample_contaminated", False):
                goal_score = min(goal_score, 0.25)

        if family == Family.WORKING_MEMORY.value:
            violation_penalty = min(1.0, 0.12 * self.constraint_violations + 0.04 * self.invalid_actions)
        elif family == Family.COGNITIVE_FLEXIBILITY.value:
            violation_penalty = min(1.0, 0.42 * self.constraint_violations + 0.16 * self.invalid_actions)
        else:
            violation_penalty = min(1.0, 0.22 * self.constraint_violations + 0.10 * self.invalid_actions)
        if family != Family.CONFLICT_RESOLUTION.value and flags.get("sample_contaminated", False):
            violation_penalty += 0.45
        if flags.get("wm_wrong_badge_uses", 0) > 0:
            violation_penalty += min(0.18, 0.08 * flags.get("wm_wrong_badge_uses", 0))
        constraint_score = max(0.0, 1.0 - min(1.0, violation_penalty))

        if self.success:
            efficiency_gap = max(0, self.step_id - optimal_steps)
            efficiency_score = max(0.0, 1.0 - 0.10 * efficiency_gap)
        else:
            efficiency_score = 0.0

        control_penalty = 0.0
        perseveration_weight = 0.34 if family == Family.COGNITIVE_FLEXIBILITY.value else 0.30
        habit_weight = 0.30 if family == Family.COGNITIVE_FLEXIBILITY.value else (0.14 if family == Family.WORKING_MEMORY.value else 0.25)
        control_penalty += perseveration_weight * min(1.0, float(self.perseveration_errors))
        control_penalty += habit_weight * min(1.0, float(self.habit_intrusions))
        control_penalty += 0.10 * min(2.0, float(self.dead_end_moves))
        control_penalty += min(0.36, 0.12 * float(flags.get("goal_drift_count", 0)))
        control_penalty += min(0.20, 0.10 * float(flags.get("wm_wrong_badge_uses", 0)))
        control_penalty += min(0.35, 0.06 * float(flags.get("wasted_meta_actions", 0)))
        control_penalty += min(0.28, 0.06 * float(flags.get("no_progress_streak", 0)))
        control_penalty += min(0.40, 0.08 * float(flags.get("offtask_meta_actions", 0)))
        control_penalty += min(0.30, 0.06 * float(flags.get("consecutive_meta_actions", 0)))
        control_penalty += min(0.20, 0.05 * float(flags.get("repeat_action_count", 0)))
        if flags.get("goal_item_misplaced", False):
            control_penalty += 0.15
            violation_penalty = min(1.0, violation_penalty + 0.15)
            constraint_score = max(0.0, 1.0 - min(1.0, violation_penalty))
        if family != Family.CONFLICT_RESOLUTION.value and flags.get("sample_contaminated", False):
            control_penalty += 0.20
        control_score = max(0.0, 1.0 - control_penalty)

        if family == Family.WORKING_MEMORY.value and not self.success:
            wm_partial = 0.0
            if goal.get("item") in inventory:
                wm_partial += 0.16
            unlocked_sequence = sum(
                1
                for edges in self.state["doors"].values()
                for info in edges.values()
                if info.get("sequence_gate") and info.get("unlocked", False)
            )
            unlocked_memory = sum(
                1
                for edges in self.state["doors"].values()
                for info in edges.values()
                if info.get("memory_gate") and info.get("unlocked", False)
            )
            wm_partial += min(0.18, 0.10 * unlocked_sequence)
            wm_partial += min(0.24, 0.14 * unlocked_memory)
            if flags.get("wm_wrong_badge_uses", 0) == 0 and any(x in inventory for x in ["blue_badge", "green_badge", "relay_key"]):
                wm_partial += 0.12
            if self.state.get("current_room") in {"corridor", "buffer_room", "anteroom", "memory_vault"}:
                wm_partial += 0.10
            goal_score = min(goal_score, wm_partial)

        if family == Family.CONFLICT_RESOLUTION.value:
            final_score = (
                0.30 * mission_completion_score
                + 0.25 * integrity_preservation_score
                + 0.20 * constraint_score
                + 0.10 * efficiency_score
                + 0.15 * control_score
            )
            if mission_completed and not integrity_preservation_score:
                final_score = min(final_score, 0.32)
            if not mission_completed and not integrity_preservation_score:
                final_score = min(final_score, 0.08)
        else:
            final_score = (
                0.45 * goal_score
                + 0.25 * constraint_score
                + 0.15 * efficiency_score
                + 0.15 * control_score
            )

        if family == Family.WORKING_MEMORY.value and not self.success:
            wm_partial_cap = 0.12
            if goal.get("item") in inventory:
                wm_partial_cap += 0.18
            if any(info.get("memory_gate") and info.get("unlocked", False) for edges in self.state["doors"].values() for info in edges.values()):
                wm_partial_cap += 0.20
            if any(info.get("sequence_gate") and info.get("unlocked", False) for edges in self.state["doors"].values() for info in edges.values()):
                wm_partial_cap += 0.18
            if any(x in inventory for x in ["relay_key", "green_badge", "blue_badge"]):
                wm_partial_cap += 0.08
            if self.state.get("current_room") in {"corridor", "buffer_room", "anteroom", "memory_vault"}:
                wm_partial_cap += 0.08
            if flags.get("wm_wrong_badge_uses", 0) > 0:
                wm_partial_cap -= min(0.06, 0.025 * flags.get("wm_wrong_badge_uses", 0))
            final_score = min(final_score, max(0.04, min(0.60, wm_partial_cap)))

        if family == Family.PLANNING.value and not self.success:
            planning_partial_cap = 0.12
            if goal.get("item") in inventory:
                planning_partial_cap += 0.10
            if flags.get("battery_charged", False):
                planning_partial_cap += 0.10
            if flags.get("package_verified", False):
                planning_partial_cap += 0.22
            if "access_card" in inventory:
                planning_partial_cap += 0.08
            if "scanner" in inventory:
                planning_partial_cap += 0.08
            if self.state.get("current_room") in {"lab", "tool_room", "inspection_room", "vault", "service_console", "seal_review", "preview_vault"}:
                planning_partial_cap += 0.08
            if flags.get("goal_item_misplaced", False):
                planning_partial_cap -= 0.05
            final_score = min(final_score, max(0.04, min(0.62, planning_partial_cap)))

        if family == Family.GOAL_MAINTENANCE.value and not self.success:
            main_path = list(self.hidden.get("main_path", []))
            distractor_rooms = set(self.hidden.get("distractor_rooms", []))
            gm_cap = 0.06
            if goal.get("item") in inventory:
                gm_cap += 0.10
            if self.state.get("current_room") in main_path:
                gm_cap += 0.035 * (main_path.index(self.state.get("current_room")) + 1)
            if self.state.get("current_room") in distractor_rooms:
                gm_cap -= 0.03
            gm_cap -= min(0.08, 0.025 * flags.get("goal_drift_count", 0))
            final_score = min(final_score, max(0.02, min(0.24, gm_cap)))

        if family == Family.COGNITIVE_FLEXIBILITY.value and not self.success:
            route_names = self.hidden.get("route_names", {})
            blue_targets = {route_names.get("blue"), route_names.get("blue_buffer"), "blue_review", "mirror_lane", "vault_review", "vault"}
            blue_targets = {x for x in blue_targets if x}
            flex_cap = 0.01
            if goal.get("item") in inventory:
                flex_cap += 0.03
            if self.state.get("current_room") in blue_targets:
                flex_cap += 0.04
            if self.state.get("switch_applied"):
                flex_cap += 0.01
            flex_cap -= min(0.30, 0.09 * self.perseveration_errors)
            flex_cap -= min(0.24, 0.06 * self.constraint_violations)
            if self.state.get("current_room") in {route_names.get("red"), "red_review", "gray_staging"}:
                flex_cap -= 0.06
            if self.perseveration_errors > 0 or self.constraint_violations > 1:
                flex_cap = min(flex_cap, 0.05)
            final_score = min(final_score, max(0.0, min(0.10, flex_cap)))

        if family == Family.INHIBITORY_CONTROL.value:
            trap_entries = int(flags.get("trap_entries", 0))
            if trap_entries > 0:
                control_score = max(0.0, control_score - min(0.40, 0.18 * trap_entries))
                constraint_score = max(0.0, constraint_score - min(0.35, 0.14 * trap_entries))
                final_score = min(final_score, 0.32 if self.success else 0.08)
                if flags.get("first_habit_intrusion", False):
                    final_score = min(final_score, 0.20 if self.success else 0.05)

        if family == Family.CONFLICT_RESOLUTION.value:
            if not mission_completed:
                conflict_cap = 0.05
                if goal.get("item") in inventory:
                    conflict_cap += 0.06
                if self.state.get("current_room") in {"safe_hall", "preserve_lab", "delivery_room"}:
                    conflict_cap += 0.04
                if flags.get("sample_contaminated", False):
                    conflict_cap -= 0.04
                conflict_cap -= min(0.04, 0.02 * float(flags.get("repeat_action_count", 0)))
                final_score = min(final_score, max(0.02, min(0.16, conflict_cap)))
            elif mission_completed and not integrity_preservation_score:
                final_score = min(final_score, 0.24)

        if not self.success and flags.get("offtask_meta_actions", 0) >= 1:
            final_score = min(final_score, 0.18)
        if not self.success and flags.get("offtask_meta_actions", 0) >= 3:
            final_score = min(final_score, 0.12)
        if not self.success and flags.get("consecutive_meta_actions", 0) >= 2:
            final_score = min(final_score, 0.12)
        if not self.success and flags.get("no_progress_streak", 0) >= 2:
            final_score = min(final_score, 0.10)
        if not self.success and flags.get("repeat_action_count", 0) >= 1:
            final_score = min(final_score, 0.12)

        failure_modes = list(self.failure_modes)
        if self.constraint_violations > 0 and "constraint_violation" not in failure_modes:
            failure_modes.append("constraint_violation")
        if self.perseveration_errors > 0 and "perseveration_after_rule_switch" not in failure_modes:
            failure_modes.append("perseveration_after_rule_switch")
        if self.habit_intrusions > 0 and "habit_intrusion" not in failure_modes:
            failure_modes.append("habit_intrusion")
        if self.dead_end_moves > 0 and "dead_end_or_wasted_move" not in failure_modes:
            failure_modes.append("dead_end_or_wasted_move")
        if self._illegal_drop_to_goal and "premature_goal_drop" not in failure_modes:
            failure_modes.append("premature_goal_drop")
        if flags.get("goal_drift_count", 0) > 0 and "goal_drift" not in failure_modes:
            failure_modes.append("goal_drift")
        if flags.get("sample_contaminated", False) and "integrity_violation" not in failure_modes:
            failure_modes.append("integrity_violation")
        if mission_completed and not integrity_preservation_score and "completed_but_contaminated" not in failure_modes:
            failure_modes.append("completed_but_contaminated")
        if flags.get("wm_wrong_badge_uses", 0) > 0 and "memory_rule_error" not in failure_modes:
            failure_modes.append("memory_rule_error")
        if flags.get("wasted_meta_actions", 0) > 0 and "meta_action_loop" not in failure_modes:
            failure_modes.append("meta_action_loop")
        if flags.get("offtask_meta_actions", 0) > 0 and "offtask_meta_action" not in failure_modes:
            failure_modes.append("offtask_meta_action")
        if flags.get("no_progress_streak", 0) >= 3 and "no_progress_loop" not in failure_modes:
            failure_modes.append("no_progress_loop")
        if flags.get("goal_item_misplaced", False) and "goal_item_misplaced" not in failure_modes:
            failure_modes.append("goal_item_misplaced")
        if flags.get("consecutive_meta_actions", 0) >= 2 and "procrastination_loop" not in failure_modes:
            failure_modes.append("procrastination_loop")
        if flags.get("repeat_action_count", 0) >= 1 and "repeated_action" not in failure_modes:
            failure_modes.append("repeated_action")
        if not self.success and "goal_not_completed" not in failure_modes:
            failure_modes.append("goal_not_completed")

        return ScoreBreakdown(
            goal_score=round(goal_score, 4),
            constraint_score=round(constraint_score, 4),
            efficiency_score=round(efficiency_score, 4),
            control_score=round(control_score, 4),
            final_score=round(final_score, 4),
            success=self.success,
            failure_modes=failure_modes,
            mission_completion_score=round(mission_completion_score, 4),
            integrity_preservation_score=round(integrity_preservation_score, 4),
        )

    def _maybe_trigger_switch_event(self) -> None:
        switch_event = self.hidden.get("switch_event")
        if not switch_event:
            self.state["switch_announced"] = False
            return
        if self.state.get("switch_applied", False):
            self.state["switch_announced"] = False
            return
        if self.step_id >= switch_event["at_step"]:
            self.state["switch_applied"] = True
            self.state["switch_announced"] = True
            self.state["active_rule"] = switch_event["new_rule"]
            if switch_event.get("new_goal") is not None:
                self.state["goal"] = json.loads(json.dumps(switch_event["new_goal"]))
        else:
            self.state["switch_announced"] = False

    def _transition(self, action: Action) -> Dict[str, Any]:
        action_name = action.action
        if action_name not in {a.value for a in ActionType}:
            self.invalid_actions += 1
            return {"ok": False, "message": f"unknown action {action.action}"}

        current_room = self.state["current_room"]
        room_contents = self.state["room_contents"]
        inventory = self.state["inventory"]
        doors = self.state["doors"]

        if action_name == ActionType.MOVE.value:
            target_room = action.target
            if target_room not in doors.get(current_room, {}):
                self.invalid_actions += 1
                self.dead_end_moves += 1
                return {"ok": False, "message": f"cannot move from {current_room} to {target_room}"}
            door_info = doors[current_room][target_room]
            blocked, reason = self._is_move_blocked(current_room, target_room, door_info)
            if blocked:
                self.constraint_violations += 1
                if reason == "perseveration":
                    self.perseveration_errors += 1
                elif reason == "habit":
                    self.habit_intrusions += 1
                else:
                    self.dead_end_moves += 1
                return {"ok": False, "message": f"move blocked: {reason}"}
            self.state["current_room"] = target_room
            self._register_progress_action(action)
            if self.episode.family == Family.INHIBITORY_CONTROL.value and door_info.get("trap", False):
                flags = self.state.setdefault("flags", {})
                flags["trap_entries"] = flags.get("trap_entries", 0) + 1
                self.constraint_violations += 1
                self.habit_intrusions += 1
                if self.step_id == 0:
                    forbidden_first = self.hidden.get("forbidden_first_move", {})
                    if current_room == forbidden_first.get("from") and target_room == forbidden_first.get("to"):
                        flags["first_habit_intrusion"] = True
            if self.episode.family == Family.CONFLICT_RESOLUTION.value and door_info.get("unsafe_shortcut", False):
                if self.state["goal"]["item"] in inventory:
                    self.state.setdefault("flags", {})["sample_contaminated"] = True
            if self.episode.family == Family.GOAL_MAINTENANCE.value:
                distractor_rooms = set(self.hidden.get("distractor_rooms", []))
                if target_room in distractor_rooms:
                    flags = self.state.setdefault("flags", {})
                    visited = set(flags.setdefault("visited_distractor_rooms", []))
                    if target_room not in visited:
                        flags.setdefault("goal_drift_count", 0)
                        flags["goal_drift_count"] += 1
                        visited.add(target_room)
                        flags["visited_distractor_rooms"] = sorted(visited)
            return {"ok": True, "message": f"moved to {target_room}"}

        if action_name == ActionType.PICK.value:
            item = action.target
            if item not in room_contents[current_room]:
                self.invalid_actions += 1
                return {"ok": False, "message": f"{item} not found in {current_room}"}
            room_contents[current_room].remove(item)
            inventory.append(item)
            self._register_progress_action(action)
            if self.episode.family == Family.GOAL_MAINTENANCE.value and item != self.state["goal"]["item"]:
                self.state.setdefault("flags", {}).setdefault("goal_drift_count", 0)
                self.state["flags"]["goal_drift_count"] += 1
            return {"ok": True, "message": f"picked {item}"}

        if action_name == ActionType.DROP.value:
            item = action.target
            if item not in inventory:
                self.invalid_actions += 1
                return {"ok": False, "message": f"{item} not in inventory"}
            goal = self.state["goal"]
            if (
                goal["type"] == "deliver_item"
                and current_room == goal["target_room"]
                and item == goal["item"]
                and goal.get("requires_verified", False)
                and not self.state.get("flags", {}).get("package_verified", False)
            ):
                self.constraint_violations += 1
                self._illegal_drop_to_goal = True
                return {"ok": False, "message": "target item cannot be deposited before verification"}
            inventory.remove(item)
            room_contents[current_room].append(item)
            if goal["type"] == "deliver_item" and item == goal["item"] and current_room != goal["target_room"]:
                flags = self.state.setdefault("flags", {})
                flags["goal_item_misplaced"] = True
                flags["goal_drift_count"] = flags.get("goal_drift_count", 0) + 1
                self.dead_end_moves += 1
            self._register_progress_action(action)
            return {"ok": True, "message": f"dropped {item}"}

        if action_name == ActionType.USE.value:
            result = self._use_item(action.target)
            if not result["ok"]:
                self.invalid_actions += 1
            else:
                self._register_progress_action(action)
            return result

        if action_name == ActionType.INSPECT.value:
            target = action.target
            self._register_meta_action("INSPECT", target)
            if target == current_room:
                return {"ok": True, "message": f"inspected room {current_room}", "details": {"items": list(room_contents.get(current_room, [])), "neighbors": sorted(list(doors.get(current_room, {}).keys()))}}
            if target in doors.get(current_room, {}):
                return {"ok": True, "message": f"inspected route to {target}", "details": json.loads(json.dumps(doors[current_room][target]))}
            if target in room_contents.get(current_room, []):
                return {"ok": True, "message": f"inspected item {target}"}
            if target in inventory:
                return {"ok": True, "message": f"inspected carried item {target}"}
            self.invalid_actions += 1
            return {"ok": False, "message": f"nothing to inspect for target {target}"}

        if action_name == ActionType.REPORT.value:
            target = action.target.lower()
            self._register_meta_action("REPORT", target)
            if target == "goal":
                return {"ok": True, "message": "reported goal", "details": json.loads(json.dumps(self.state["goal"]))}
            if target == "inventory":
                return {"ok": True, "message": "reported inventory", "details": list(inventory)}
            if target == "map":
                return {"ok": True, "message": "reported map", "details": self.compact_map()}
            if target == "room":
                return {"ok": True, "message": "reported room", "details": {"room": current_room, "items": list(room_contents.get(current_room, []))}}
            if target == "routes":
                return {"ok": True, "message": "reported routes", "details": sorted(list(doors.get(current_room, {}).keys()))}
            self.invalid_actions += 1
            return {"ok": False, "message": f"unknown report target {action.target}"}

        if action_name == ActionType.WAIT.value:
            self._register_meta_action("WAIT", "none")
            return {"ok": True, "message": "waited"}

        self.invalid_actions += 1
        return {"ok": False, "message": "unsupported action"}


    def _meta_signature(self, kind: str, target: str) -> str:
        inventory = self.state.get("inventory", [])
        inv_key = ",".join(sorted(inventory))
        current_room = self.state.get("current_room", "")
        return f"{kind}:{target}@{current_room}|inv:{inv_key}"

    def _register_progress_action(self, action: Optional[Action] = None) -> None:
        flags = self.state.setdefault("flags", {})
        flags["no_progress_streak"] = 0
        flags["consecutive_meta_actions"] = 0
        flags["last_meta_signature"] = None
        if action is not None:
            key = f"{action.action.upper()}::{action.target}"
            if flags.get("last_action_key") == key:
                flags["repeat_action_count"] = flags.get("repeat_action_count", 0) + 1
            else:
                flags["repeat_action_count"] = 0
            flags["last_action_key"] = key

    def _register_meta_action(self, kind: str, target: str) -> None:
        flags = self.state.setdefault("flags", {})
        sig = self._meta_signature(kind, target)
        seen = list(flags.get("seen_meta_queries", []))
        if sig in seen:
            flags["wasted_meta_actions"] = flags.get("wasted_meta_actions", 0) + 1
            self.dead_end_moves += 1
        else:
            seen.append(sig)
            flags["seen_meta_queries"] = seen
            flags["meta_action_count"] = flags.get("meta_action_count", 0) + 1
        if flags.get("last_meta_signature") == sig:
            flags["wasted_meta_actions"] = flags.get("wasted_meta_actions", 0) + 1
            self.dead_end_moves += 1
        progress_options = self._progress_like_actions()
        if progress_options:
            flags["offtask_meta_actions"] = flags.get("offtask_meta_actions", 0) + 1
        flags["last_meta_signature"] = sig
        flags["no_progress_streak"] = flags.get("no_progress_streak", 0) + 1
        flags["consecutive_meta_actions"] = flags.get("consecutive_meta_actions", 0) + 1
        key = f"{kind.upper()}::{target}"
        if flags.get("last_action_key") == key:
            flags["repeat_action_count"] = flags.get("repeat_action_count", 0) + 1
            flags["wasted_meta_actions"] = flags.get("wasted_meta_actions", 0) + 1
            self.dead_end_moves += 1
        else:
            flags["repeat_action_count"] = 0
        flags["last_action_key"] = key
        if flags.get("no_progress_streak", 0) >= 2:
            flags["wasted_meta_actions"] = flags.get("wasted_meta_actions", 0) + 1
            self.dead_end_moves += 1

    def _is_move_blocked(self, current_room: str, target_room: str, door_info: Dict[str, Any]) -> Tuple[bool, str]:
        inventory = self.state["inventory"]
        active_rule = self.state["active_rule"]

        if door_info.get("requires") and not door_info.get("unlocked", False):
            if door_info["requires"] not in inventory:
                return True, "missing_required_key"
            return True, "requires_use"

        family = self.episode.family
        if family == Family.COGNITIVE_FLEXIBILITY.value:
            # Keep stale-rule moves blocked, but let explicitly marked decoy/misleading branches remain executable.
            # Otherwise this family can look richly branched while offering no executable wrong alternative.
            if door_info.get("decoy") or door_info.get("misleading"):
                return False, ""
            if active_rule == "avoid_red" and door_info.get("color") == "red":
                return True, "perseveration"
            if active_rule == "prefer_blue_route" and door_info.get("color") not in {"blue", "start_return"}:
                return True, "perseveration"

        if family == Family.INHIBITORY_CONTROL.value:
            # For inhibitory-control episodes, trap routes remain executable so the task
            # measures suppression of tempting but harmful actions, not simple blocked-move avoidance.
            return False, ""

        return False, ""

    def _use_item(self, tool: str) -> Dict[str, Any]:
        current_room = self.state["current_room"]
        inventory = self.state["inventory"]
        doors = self.state["doors"]

        if tool not in inventory:
            return {"ok": False, "message": f"{tool} not in inventory"}

        if self.episode.family == Family.PLANNING.value and tool == "battery_pack":
            if current_room != "charging_station":
                return {"ok": False, "message": "battery_pack can only be used at charging_station"}
            self.state["flags"]["battery_charged"] = True
            unlocked_any = False
            for _, info in doors.get("charging_station", {}).items():
                if info.get("requires") == "battery_pack" and not info.get("unlocked", False):
                    info["unlocked"] = True
                    unlocked_any = True
            if "lab" in doors.get("charging_station", {}):
                doors["charging_station"]["lab"]["unlocked"] = True
                unlocked_any = True
            if unlocked_any:
                return {"ok": True, "message": "battery charged; charging_station route unlocked"}
            return {"ok": False, "message": "battery_pack had no matching lock to unlock"}

        if tool == "scanner":
            if current_room != "inspection_room":
                return {"ok": False, "message": "scanner can only be used in inspection_room"}
            self.state["flags"]["package_verified"] = True
            return {"ok": True, "message": "package verified"}

        if self.episode.family == Family.WORKING_MEMORY.value and tool.endswith("_badge"):
            matched_gate = False
            for _, info in doors.get(current_room, {}).items():
                if info.get("requires") and not info.get("unlocked", False):
                    matched_gate = True
                    if info.get("requires") == tool:
                        info["unlocked"] = True
                        return {"ok": True, "message": f"used {tool}; memory gate unlocked"}
            if matched_gate:
                self.state.setdefault("flags", {}).setdefault("wm_wrong_badge_uses", 0)
                self.state["flags"]["wm_wrong_badge_uses"] += 1
                self.constraint_violations += 1
                return {"ok": False, "message": f"used {tool}; incorrect badge for current memory gate"}

        unlocked_any = False
        for _, info in doors.get(current_room, {}).items():
            if info.get("requires") == tool and not info.get("unlocked", False):
                info["unlocked"] = True
                unlocked_any = True
        if unlocked_any:
            return {"ok": True, "message": f"used {tool}; local matching doors unlocked"}

        return {"ok": False, "message": f"no valid use for {tool}"}

    def _check_terminal(self) -> None:
        goal = self.state["goal"]
        room_contents = self.state["room_contents"]
        current_room = self.state["current_room"]
        flags = self.state.get("flags", {})

        if goal["type"] == "deliver_item":
            target_item = goal["item"]
            target_room = goal["target_room"]
            if target_item in room_contents[target_room]:
                if goal.get("requires_verified", False) and not flags.get("package_verified", False):
                    pass
                else:
                    self.success = True
                    self.done = True
                    self.goal_completed_step = self.step_id
                    return

        if goal["type"] == "reach_room_with_item":
            target_room = goal["target_room"]
            required_item = goal.get("required_item")
            if current_room == target_room and (required_item is None or required_item in self.state["inventory"]):
                self.success = True
                self.done = True
                self.goal_completed_step = self.step_id
                return

        if self.step_id >= self.max_steps:
            self.done = True


def _unique_generated_room_name(room_contents: Dict[str, List[str]], base: str) -> str:
    room = base
    suffix = 1
    while room in room_contents:
        suffix += 1
        room = f"{base}_{suffix}"
    return room


def _family_detour_profile(family: str) -> Dict[str, Any]:
    profiles = {
        Family.GOAL_MAINTENANCE.value: {"base": "handoff_detour", "color": "green", "flags": {"decoy": True, "misleading": True}, "target_exec_wrong": 3},
        Family.COGNITIVE_FLEXIBILITY.value: {"base": "switch_review", "color": "blue", "flags": {"decoy": True, "misleading": True}, "target_exec_wrong": 4},
        Family.INHIBITORY_CONTROL.value: {"base": "temptation_lane", "color": "amber", "flags": {"trap": True, "misleading": True}, "target_exec_wrong": 3},
        Family.PLANNING.value: {"base": "service_detour", "color": "violet", "flags": {"decoy": True, "premature": True, "misleading": True}, "target_exec_wrong": 4},
        Family.CONFLICT_RESOLUTION.value: {"base": "shortcut_review", "color": "orange", "flags": {"unsafe_shortcut": True, "misleading": True}, "target_exec_wrong": 3},
        Family.WORKING_MEMORY.value: {"base": "buffer_detour", "color": "cyan", "flags": {"decoy": True, "misleading": True}, "target_exec_wrong": 4},
    }
    return dict(profiles.get(family, {"base": "detour", "color": "gray", "flags": {"decoy": True, "misleading": True}, "target_exec_wrong": 1}))


def _count_executable_wrong_progress(env: RuleShiftEnv, oracle_action: Action) -> Tuple[int, int]:
    obs = env.observe()
    progress_candidates = [c for c in obs.get("candidate_actions", []) if c.get("kind") == "progress"]
    exec_wrong = 0
    tempting_wrong = 0
    current_room = env.state.get("current_room")
    for cand in progress_candidates:
        c_action = str(cand.get("action", "")).upper()
        c_target = cand.get("target")
        if c_action == oracle_action.action and c_target == oracle_action.target:
            continue
        if c_action == "MOVE":
            info = env.state.get("doors", {}).get(current_room, {}).get(c_target, {})
            if not info:
                continue
            blocked, _ = env._is_move_blocked(current_room, c_target, info)
            if blocked:
                continue
            exec_wrong += 1
            if info.get("decoy") or info.get("trap") or info.get("unsafe_shortcut") or info.get("misleading") or info.get("premature"):
                tempting_wrong += 1
        elif c_action == "PICK":
            if c_target in set(env.state.get("room_contents", {}).get(current_room, [])):
                exec_wrong += 1
        elif c_action == "USE":
            if c_target in set(env.state.get("inventory", [])) and env._can_use_item_now(c_target):
                exec_wrong += 1
        elif c_action == "DROP":
            goal = env.state.get("goal", {})
            if c_target in set(env.state.get("inventory", [])) and current_room == goal.get("target_room"):
                exec_wrong += 1
    return exec_wrong, tempting_wrong


def _add_executable_detour_for_state(ep: Episode, anchor: str, rejoin: str, family: str, step_idx: int, rng: random.Random) -> bool:
    room_contents = ep.initial_state["room_contents"]
    doors = ep.initial_state["doors"]
    if anchor not in doors:
        return False
    profile = _family_detour_profile(family)
    base = profile["base"]
    room = _unique_generated_room_name(room_contents, f"{base}_{anchor}")
    decoy_items = {
        Family.GOAL_MAINTENANCE.value: ["handoff_note", "priority_stub", "reward_token"],
        Family.COGNITIVE_FLEXIBILITY.value: ["obsolete_routing_card", "mirror_token", "review_badge"],
        Family.INHIBITORY_CONTROL.value: ["temptation_pass", "speed_stamp", "shortcut_badge"],
        Family.PLANNING.value: ["preview_ticket", "service_stub", "seal_form"],
        Family.CONFLICT_RESOLUTION.value: ["fast_lane_pass", "sterility_waiver", "expedite_stub"],
        Family.WORKING_MEMORY.value: ["badge_hint", "route_hint", "echo_token"],
    }
    room_contents[room] = [rng.choice(decoy_items.get(family, ["memo_card", "audit_stamp", "routing_stub"]))]
    forward = {"color": profile["color"], "unlocked": True, **profile["flags"]}
    back = {"color": profile["color"], "unlocked": True, **profile["flags"]}
    doors.setdefault(anchor, {})[room] = dict(forward)
    doors.setdefault(room, {})[anchor] = dict(back)
    if rejoin and rejoin != anchor:
        doors[room][rejoin] = dict(back)

    # Add a second recoverable branch from the detour room itself so the mistake remains executable
    # for more than one step and becomes a stronger discriminator than a simple single-hop side room.
    branch_room = _unique_generated_room_name(room_contents, f"{base}_{anchor}_branch")
    branch_flags = dict(profile["flags"])
    if family == Family.INHIBITORY_CONTROL.value:
        branch_flags["trap"] = True
    if family == Family.CONFLICT_RESOLUTION.value:
        branch_flags["unsafe_shortcut"] = True
    room_contents[branch_room] = []
    doors[room][branch_room] = {"color": profile["color"], "unlocked": True, **branch_flags}
    doors[branch_room] = {room: {"color": profile["color"], "unlocked": True, **branch_flags}}
    if rejoin and rejoin != anchor:
        doors[branch_room][rejoin] = {"color": profile["color"], "unlocked": True, **branch_flags}

    ep.hidden_state.setdefault("auto_detour_rooms", []).extend([room, branch_room])
    ep.hidden_state.setdefault("auto_detour_meta", []).append({
        "anchor": anchor,
        "room": room,
        "branch_room": branch_room,
        "rejoin": rejoin,
        "step": step_idx,
    })
    return True


def enhance_episode_discriminability(ep: Episode, rng: random.Random) -> Episode:
    plan = list(ep.hidden_state.get("oracle_plan", []))
    if not plan:
        return ep
    profile = _family_detour_profile(ep.family)
    target_exec_wrong = int(profile.get("target_exec_wrong", 1))
    patched_by_anchor: Dict[str, int] = {}
    env = RuleShiftEnv(ep)
    for idx, raw_act in enumerate(plan):
        if env.done:
            break
        oracle_action = Action.from_any(raw_act)
        current_room = env.state.get("current_room")
        if not current_room:
            break
        exec_wrong, tempting_wrong = _count_executable_wrong_progress(env, oracle_action)
        need_more = exec_wrong < target_exec_wrong or tempting_wrong < 1
        if need_more and patched_by_anchor.get(current_room, 0) < 2:
            rejoin = current_room
            if oracle_action.action == "MOVE":
                rejoin = oracle_action.target
            elif idx + 1 < len(plan):
                nxt = Action.from_any(plan[idx + 1])
                if nxt.action == "MOVE":
                    rejoin = nxt.target
            if _add_executable_detour_for_state(ep, current_room, rejoin, ep.family, idx, rng):
                patched_by_anchor[current_room] = patched_by_anchor.get(current_room, 0) + 1
                env = RuleShiftEnv(ep)
                for prev in plan[:idx]:
                    if env.done:
                        break
                    env.apply(Action.from_any(prev))
        env.apply(oracle_action)
    return ep


def generate_episode(family: str, difficulty: int, seed: int) -> Episode:
    rng = random.Random(seed)
    if family == Family.GOAL_MAINTENANCE.value:
        ep = generate_goal_maintenance_episode(rng, difficulty, seed)
    elif family == Family.COGNITIVE_FLEXIBILITY.value:
        ep = generate_cognitive_flexibility_episode(rng, difficulty, seed)
    elif family == Family.INHIBITORY_CONTROL.value:
        ep = generate_inhibitory_control_episode(rng, difficulty, seed)
    elif family == Family.PLANNING.value:
        ep = generate_planning_episode(rng, difficulty, seed)
    elif family == Family.CONFLICT_RESOLUTION.value:
        ep = generate_conflict_resolution_episode(rng, difficulty, seed)
    elif family == Family.WORKING_MEMORY.value:
        ep = generate_working_memory_episode(rng, difficulty, seed)
    else:
        raise ValueError(f"unsupported family: {family}")
    return enhance_episode_discriminability(ep, rng)



def generate_goal_maintenance_episode(rng: random.Random, difficulty: int, seed: int) -> Episode:
    difficulty = max(1, min(4, int(difficulty)))
    item = rng.choice(["dossier_A", "dossier_K", "dossier_Z"])
    room_contents = {"start": [item], "focus_hall": [], "archive": []}
    doors = {"start": {}, "focus_hall": {}, "archive": {}}

    main_path = ["focus_hall"]
    if difficulty >= 2:
        room_contents["checkpoint"] = []
        main_path.append("checkpoint")
    if difficulty >= 3:
        room_contents["relay_room"] = []
        main_path.append("relay_room")
    if difficulty >= 4:
        room_contents["final_corridor"] = []
        main_path.append("final_corridor")
    main_path.append("archive")

    prev = "start"
    for room in main_path[:-1]:
        doors.setdefault(prev, {})[room] = {"color": "green", "unlocked": True}
        doors.setdefault(room, {})
        prev = room
    doors.setdefault(prev, {})["archive"] = {"color": "green", "unlocked": True}

    distractor_rooms: List[str] = []
    room_contents["bonus_room"] = ["shiny_token"]
    doors["start"]["bonus_room"] = {"color": "gold", "unlocked": True, "decoy": True}
    doors["bonus_room"] = {"start": {"color": "gold", "unlocked": True, "decoy": True}}
    room_contents["notice_board"] = []
    doors["start"]["notice_board"] = {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}
    doors["notice_board"] = {"start": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}}
    distractor_rooms.append("bonus_room")
    room_contents["mail_slot"] = ["routing_coupon"]
    doors["start"]["mail_slot"] = {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}
    doors["mail_slot"] = {"start": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}}
    distractor_rooms.append("mail_slot")

    room_contents["notice_board"] = ["bonus_ticket"]
    doors["focus_hall"]["notice_board"] = {"color": "amber", "unlocked": True, "decoy": True, "misleading": True}
    doors["notice_board"] = {"focus_hall": {"color": "amber", "unlocked": True, "decoy": True, "misleading": True}}
    distractor_rooms.append("notice_board")

    if difficulty >= 2:
        room_contents["archive_preview"] = ["routing_stub"]
        doors["focus_hall"]["archive_preview"] = {"color": "green", "unlocked": True, "decoy": True, "misleading": True}
        doors["archive_preview"] = {"focus_hall": {"color": "green", "unlocked": True, "decoy": True, "misleading": True}, "archive": {"color": "green", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("archive_preview")
        room_contents["side_office"] = ["coupon"]
        branch_from = "focus_hall"
        doors[branch_from]["side_office"] = {"color": "yellow", "unlocked": True, "decoy": True}
        doors["side_office"] = {branch_from: {"color": "yellow", "unlocked": True, "decoy": True}}
        distractor_rooms.append("side_office")
    if difficulty >= 3:
        room_contents["intake_archive"] = ["handoff_stub"]
        doors["checkpoint"]["intake_archive"] = {"color": "green", "unlocked": True, "decoy": True, "misleading": True}
        doors["intake_archive"] = {"checkpoint": {"color": "green", "unlocked": True, "decoy": True, "misleading": True}, "archive": {"color": "green", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("intake_archive")
        room_contents["admin_desk"] = ["status_form"]
        doors["checkpoint"]["admin_desk"] = {"color": "orange", "unlocked": True, "decoy": True}
        doors["admin_desk"] = {"checkpoint": {"color": "orange", "unlocked": True, "decoy": True}}
        distractor_rooms.append("admin_desk")
    if difficulty >= 4:
        room_contents["express_archive"] = ["priority_ticket"]
        doors["relay_room"]["express_archive"] = {"color": "green", "unlocked": True, "decoy": True, "misleading": True}
        doors["express_archive"] = {"relay_room": {"color": "green", "unlocked": True, "decoy": True, "misleading": True}, "archive": {"color": "green", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("express_archive")
        room_contents["urgent_terminal_room"] = ["escalation_note"]
        doors["relay_room"]["urgent_terminal_room"] = {"color": "orange", "unlocked": True, "decoy": True}
        doors["urgent_terminal_room"] = {"relay_room": {"color": "orange", "unlocked": True, "decoy": True}}
        distractor_rooms.append("urgent_terminal_room")
        room_contents["reward_counter"] = ["bonus_stamp"]
        doors["final_corridor"]["reward_counter"] = {"color": "purple", "unlocked": True, "decoy": True}
        doors["reward_counter"] = {"final_corridor": {"color": "purple", "unlocked": True, "decoy": True}}
        distractor_rooms.append("reward_counter")
    if difficulty >= 2:
        room_contents["priority_desk"] = ["handoff_form"]
        doors["focus_hall"]["priority_desk"] = {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}
        doors["priority_desk"] = {"focus_hall": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("priority_desk")
    if difficulty >= 3 and "checkpoint" in room_contents:
        room_contents["staging_bay"] = ["seal_tag"]
        doors["checkpoint"]["staging_bay"] = {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}
        doors["staging_bay"] = {"checkpoint": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("staging_bay")

    late_anchor_rooms = ["start", "focus_hall"] + (["checkpoint"] if "checkpoint" in room_contents else []) + (["relay_room"] if "relay_room" in room_contents else []) + (["final_corridor"] if "final_corridor" in room_contents else [])
    room_contents["receipt_window"] = ["receipt_stub"]
    doors.setdefault("archive", {})["receipt_window"] = {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}
    doors["receipt_window"] = {"archive": {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}}
    distractor_rooms.append("receipt_window")
    if "checkpoint" in room_contents:
        room_contents["routing_loop"] = ["route_stub"]
        doors["checkpoint"]["routing_loop"] = {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}
        doors["routing_loop"] = {"checkpoint": {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("routing_loop")
    add_decoy_branches(rng, room_contents, doors, late_anchor_rooms + ["archive"], max(3, difficulty + 1), "goal")
    if difficulty >= 3 and "relay_room" in room_contents:
        room_contents["courtesy_archive"] = ["priority_stub"]
        doors["relay_room"]["courtesy_archive"] = {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}
        doors["courtesy_archive"] = {"archive": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}, "relay_room": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("courtesy_archive")
    if difficulty >= 4 and "final_corridor" in room_contents:
        room_contents["bonus_archive"] = ["merit_stamp"]
        doors["final_corridor"]["bonus_archive"] = {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}
        doors["bonus_archive"] = {"archive": {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}, "final_corridor": {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}}
        distractor_rooms.append("bonus_archive")
    oracle_plan = [{"action": "PICK", "target": item}] + [{"action": "MOVE", "target": room} for room in main_path] + [{"action": "DROP", "target": item}]
    optimal_steps = len(oracle_plan)
    focus_alert = "Keep the primary delivery objective active. Ignore attractive side tasks unless they directly help delivery."

    initial_state = {
        "current_room": "start",
        "inventory": [],
        "flags": {"goal_drift_count": 0, "visited_distractor_rooms": []},
        "switch_applied": False,
        "switch_announced": False,
        "active_rule": "maintain_primary_goal",
        "goal": {"type": "deliver_item", "item": item, "target_room": "archive", "requires_verified": False},
        "room_contents": room_contents,
        "doors": doors,
    }
    hidden_state = {
        "switch_event": {"at_step": max(1, difficulty - 1), "new_rule": "maintain_primary_goal", "message": focus_alert},
        "focus_alert": focus_alert,
        "main_path": list(main_path),
        "difficulty_profile": {"tier": difficulty, "main_path_length": len(main_path), "distractor_branches": len(distractor_rooms)},
        "distractor_rooms": distractor_rooms,
        "family_variant": f"goal_path_len_{len(main_path)}_distractors_{len(distractor_rooms)}",
        "oracle_plan": oracle_plan,
    }
    instructions = f"Deliver {item} to archive. Keep the primary objective active even when other tasks look rewarding or urgent."
    return Episode(
        episode_id=f"goal_d{difficulty}_{seed}",
        seed=seed,
        family=Family.GOAL_MAINTENANCE.value,
        difficulty=difficulty,
        title="Maintain the Primary Goal Despite Attractive Side Tasks",
        instructions=instructions,
        max_steps=optimal_steps + 5,
        initial_state=initial_state,
        hidden_state=hidden_state,
        scoring_metadata={"optimal_steps": optimal_steps},
    )

def generate_cognitive_flexibility_episode(rng: random.Random, difficulty: int, seed: int) -> Episode:
    difficulty = max(1, min(4, int(difficulty)))
    item = rng.choice(["sample_A", "sample_K", "sample_Z"])
    new_rule = rng.choice(["avoid_red", "prefer_blue_route"])
    red_alias, blue_alias = rng.choice([
        ("red_hall", "blue_hall"),
        ("scarlet_lane", "azure_lane"),
        ("crimson_path", "cobalt_path"),
    ])
    blue_buffer = "blue_buffer"
    support_item = "blue_token"

    room_contents = {"start": [item], red_alias: [], blue_alias: [], "vault": []}
    doors = {
        "start": {
            red_alias: {"color": "red", "unlocked": True},
            blue_alias: {"color": "blue", "unlocked": True},
        },
        red_alias: {
            "start": {"color": "start_return", "unlocked": True},
            "vault": {"color": "red", "unlocked": True},
        },
        blue_alias: {
            "start": {"color": "start_return", "unlocked": True},
        },
        "vault": {},
    }
    room_contents["gray_staging"] = []
    doors["start"]["gray_staging"] = {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}
    doors["gray_staging"] = {"vault": {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}}
    room_contents["vault_review"] = []
    doors["vault"]["vault_review"] = {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}
    doors["vault_review"] = {"vault": {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}}
    room_contents["rule_lounge"] = []
    doors[red_alias]["rule_lounge"] = {"color": "amber", "unlocked": True, "decoy": True, "misleading": True}
    doors["rule_lounge"] = {red_alias: {"color": "amber", "unlocked": True, "decoy": True, "misleading": True}}
    room_contents["blue_review"] = []
    doors[blue_alias]["blue_review"] = {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True}
    doors["blue_review"] = {
        blue_alias: {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True},
        "vault": {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True},
        "mirror_lane": {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True},
    }
    room_contents["mirror_lane"] = []
    doors["mirror_lane"] = {
        "vault": {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True},
        blue_alias: {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True},
        "blue_review": {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True},
    }
    doors[blue_alias]["mirror_lane"] = {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True}
    room_contents["red_review"] = []
    doors[red_alias]["red_review"] = {"color": "amber", "unlocked": True, "decoy": True, "misleading": True}
    doors["red_review"] = {
        red_alias: {"color": "amber", "unlocked": True, "decoy": True, "misleading": True},
        "gray_staging": {"color": "amber", "unlocked": True, "decoy": True, "misleading": True},
    }

    if difficulty == 1:
        switch_at = 1
        doors[blue_alias]["vault"] = {"color": "blue", "unlocked": True}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "MOVE", "target": blue_alias},
            {"action": "MOVE", "target": "vault"},
            {"action": "DROP", "target": item},
        ]
        optimal_steps = 4

    elif difficulty == 2:
        switch_at = 2
        doors[blue_alias]["vault"] = {"color": "blue", "unlocked": True}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "MOVE", "target": red_alias},
            {"action": "MOVE", "target": "start"},
            {"action": "MOVE", "target": blue_alias},
            {"action": "MOVE", "target": "vault"},
            {"action": "DROP", "target": item},
        ]
        optimal_steps = 6

    elif difficulty == 3:
        switch_at = 2
        room_contents[blue_buffer] = []
        doors[blue_alias][blue_buffer] = {"color": "blue", "unlocked": True}
        doors[blue_buffer] = {"vault": {"color": "blue", "unlocked": True}, "review_node": {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}}
        room_contents["review_node"] = []
        doors["review_node"] = {blue_buffer: {"color": "violet", "unlocked": True, "decoy": True, "misleading": True}}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "MOVE", "target": red_alias},
            {"action": "MOVE", "target": "start"},
            {"action": "MOVE", "target": blue_alias},
            {"action": "MOVE", "target": blue_buffer},
            {"action": "MOVE", "target": "vault"},
            {"action": "DROP", "target": item},
        ]
        optimal_steps = 7

    else:
        switch_at = 2
        room_contents[blue_buffer] = []
        room_contents["start"].append(support_item)
        doors["start"][blue_alias] = {"color": "blue", "unlocked": False, "requires": support_item}
        doors[blue_alias][blue_buffer] = {"color": "blue", "unlocked": True}
        doors[blue_buffer] = {"vault": {"color": "blue", "unlocked": True}}
        doors[blue_buffer]["mirror_lane"] = {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True}
        doors["mirror_lane"][blue_buffer] = {"color": "cyan", "unlocked": True, "decoy": True, "misleading": True}
        room_contents["gray_lane"] = []
        doors["start"]["gray_lane"] = {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}
        doors["gray_lane"] = {"vault": {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}, "gray_review": {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}}
        room_contents["gray_review"] = []
        doors["gray_review"] = {"gray_lane": {"color": "gray", "unlocked": True, "decoy": True, "misleading": True}}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "MOVE", "target": red_alias},
            {"action": "MOVE", "target": "start"},
            {"action": "PICK", "target": support_item},
            {"action": "USE", "target": support_item},
            {"action": "MOVE", "target": blue_alias},
            {"action": "MOVE", "target": blue_buffer},
            {"action": "MOVE", "target": "vault"},
            {"action": "DROP", "target": item},
        ]
        optimal_steps = 9

    initial_state = {
        "current_room": "start",
        "inventory": [],
        "flags": {},
        "switch_applied": False,
        "switch_announced": False,
        "active_rule": "take_red_first",
        "goal": {"type": "deliver_item", "item": item, "target_room": "vault", "requires_verified": False},
        "room_contents": room_contents,
        "doors": doors,
    }

    add_decoy_branches(rng, room_contents, doors, ["start", red_alias, blue_alias] + ([blue_buffer] if blue_buffer in room_contents else []), difficulty, "flex")
    hidden_state = {
        "route_names": {"red": red_alias, "blue": blue_alias, "blue_buffer": blue_buffer},
        "flex_subtype": (
            "late_rule_reversal_with_token_gate" if difficulty >= 4 else
            "late_rule_reversal_with_buffer" if difficulty == 3 else
            "route_reversal_after_commit" if difficulty == 2 else
            "simple_rule_flip"
        ),
        "switch_event": {
            "at_step": switch_at,
            "new_rule": new_rule,
            "message": (
                f"RULE UPDATE: the previous route preference is no longer valid. "
                f"New rule is {new_rule}. Re-plan immediately before your next move."
            ),
        },
        "difficulty_profile": {
            "tier": difficulty,
            "backtrack_required": difficulty >= 2,
            "extra_blue_hop": difficulty >= 3,
            "blue_requires_token": difficulty >= 4,
        },
        "oracle_plan": oracle_plan,
    }

    instructions = (
        f"Collect {item} and deliver it to the vault. The initial route memo says the red path is preferred, "
        "but a later system update may replace that rule. Follow the latest rule, not the stale one."
    )

    return Episode(
        episode_id=f"flex_d{difficulty}_{seed}",
        seed=seed,
        family=Family.COGNITIVE_FLEXIBILITY.value,
        difficulty=difficulty,
        title="Adaptive Routing Under Rule Shift",
        instructions=instructions,
        max_steps=optimal_steps + 2,
        initial_state=initial_state,
        hidden_state=hidden_state,
        scoring_metadata={"optimal_steps": optimal_steps},
    )



def generate_inhibitory_control_episode(rng: random.Random, difficulty: int, seed: int) -> Episode:
    difficulty = max(1, min(4, int(difficulty)))
    package = rng.choice(["package_alpha", "package_beta", "package_gamma"])

    room_contents = {"dock": [package], "red_corridor": [], "safe_corridor": [], "secure_zone": []}
    doors = {
        "dock": {
            "red_corridor": {"color": "red", "unlocked": True, "trap": True},
            "safe_corridor": {"color": "green", "unlocked": True},
        },
        "red_corridor": {"secure_zone": {"color": "red", "unlocked": True, "trap": True}},
        "safe_corridor": {"secure_zone": {"color": "green", "unlocked": True}},
        "secure_zone": {},
    }
    room_contents["pause_bay"] = []
    doors["safe_corridor"]["pause_bay"] = {"color": "amber", "unlocked": True, "trap": True, "misleading": True}
    doors["pause_bay"] = {
        "secure_zone": {"color": "amber", "unlocked": True, "trap": True, "misleading": True},
        "safe_corridor": {"color": "amber", "unlocked": True, "trap": True, "misleading": True},
    }
    room_contents["yellow_hold"] = []
    doors["safe_corridor"]["yellow_hold"] = {"color": "yellow", "unlocked": True, "trap": True, "misleading": True}
    doors["yellow_hold"] = {
        "secure_zone": {"color": "yellow", "unlocked": True, "trap": True, "misleading": True},
        "safe_corridor": {"color": "yellow", "unlocked": True, "trap": True, "misleading": True},
    }
    oracle_plan = [
        {"action": "PICK", "target": package},
        {"action": "MOVE", "target": "safe_corridor"},
        {"action": "MOVE", "target": "secure_zone"},
        {"action": "DROP", "target": package},
    ]
    room_contents["receipt_bench"] = []
    doors["secure_zone"]["receipt_bench"] = {"color": "teal", "unlocked": True, "trap": True, "misleading": True}
    doors["receipt_bench"] = {"secure_zone": {"color": "teal", "unlocked": True, "trap": True, "misleading": True}}
    optimal_steps = 4

    hint = "Historical runs often rewarded moving into red_corridor first, and a later amber detour looks efficient but is still unsafe."

    room_contents["audit_bay"] = []
    doors["dock"]["audit_bay"] = {"color": "yellow", "unlocked": True, "trap": True, "misleading": True}
    doors["audit_bay"] = {"secure_zone": {"color": "yellow", "unlocked": True, "trap": True, "misleading": True}}
    room_contents["green_preview"] = []
    doors["dock"]["green_preview"] = {"color": "lime", "unlocked": True, "trap": True, "misleading": True}
    doors["green_preview"] = {"secure_zone": {"color": "lime", "unlocked": True, "trap": True, "misleading": True}}

    if difficulty >= 2:
        room_contents["bonus_room"] = []
        doors["dock"]["bonus_room"] = {"color": "gold", "unlocked": True, "trap": True}
        doors["bonus_room"] = {"secure_zone": {"color": "gold", "unlocked": True, "trap": True}}
        hint += " A second high-salience decoy route is also visible from the dock."

    if difficulty >= 3:
        room_contents["safe_mid"] = []
        room_contents["shortcut_gate"] = []
        doors["safe_corridor"] = {
            "safe_mid": {"color": "green", "unlocked": True},
            "shortcut_gate": {"color": "orange", "unlocked": True, "trap": True},
        }
        doors["safe_mid"] = {
            "secure_zone": {"color": "green", "unlocked": True},
            "amber_detour": {"color": "amber", "unlocked": True, "trap": True, "misleading": True},
            "yellow_hold": {"color": "yellow", "unlocked": True, "trap": True, "misleading": True},
        }
        room_contents["amber_detour"] = []
        doors["amber_detour"] = {
            "secure_zone": {"color": "amber", "unlocked": True, "trap": True, "misleading": True},
            "safe_mid": {"color": "amber", "unlocked": True, "trap": True, "misleading": True},
        }
        doors["shortcut_gate"] = {
            "secure_zone": {"color": "orange", "unlocked": True, "trap": True},
            "safe_mid": {"color": "orange", "unlocked": True, "trap": True},
        }
        oracle_plan = [
            {"action": "PICK", "target": package},
            {"action": "MOVE", "target": "safe_corridor"},
            {"action": "MOVE", "target": "safe_mid"},
            {"action": "MOVE", "target": "secure_zone"},
            {"action": "DROP", "target": package},
        ]
        optimal_steps = 5
        hint += " The safe route is now longer, so suppressing the shortcut impulse matters more."

    if difficulty >= 4:
        room_contents["amber_corridor"] = []
        doors["dock"]["amber_corridor"] = {"color": "amber", "unlocked": True, "trap": True}
        doors["amber_corridor"] = {"secure_zone": {"color": "amber", "unlocked": True, "trap": True}}
        room_contents["safe_mid"].append("tempt_note")
        room_contents["green_lobby"] = []
        doors["safe_mid"]["green_lobby"] = {"color": "lime", "unlocked": True, "trap": True, "misleading": True}
        doors["green_lobby"] = {"secure_zone": {"color": "lime", "unlocked": True, "trap": True, "misleading": True}, "safe_mid": {"color": "lime", "unlocked": True, "trap": True, "misleading": True}}
        hint += " Even after choosing the safe corridor, another tempting late shortcut appears."

    inhib_anchors = ["dock", "safe_corridor", "secure_zone"]
    if "safe_mid" in room_contents:
        inhib_anchors.append("safe_mid")
    add_decoy_branches(rng, room_contents, doors, inhib_anchors, max(3, difficulty + 1), "inhib")

    initial_state = {
        "current_room": "dock",
        "inventory": [],
        "flags": {"trap_entries": 0},
        "switch_applied": False,
        "switch_announced": False,
        "active_rule": "avoid_trap_route",
        "goal": {"type": "deliver_item", "item": package, "target_room": "secure_zone", "requires_verified": False},
        "room_contents": room_contents,
        "doors": doors,
    }

    hidden_state = {
        "forbidden_first_move": {"from": "dock", "to": "red_corridor"},
        "tempting_hint": hint,
        "switch_event": None,
        "difficulty_profile": {
            "tier": difficulty,
            "extra_dock_decoy": difficulty >= 2,
            "long_safe_route": difficulty >= 3,
            "second_stage_trap": difficulty >= 3,
            "late_trap_salience": difficulty >= 4,
        },
        "family_variant": f"inhib_safe_depth_{optimal_steps}",
        "oracle_plan": oracle_plan,
    }

    instructions = (
        f"Deliver {package} to secure_zone. Safety override: red_corridor is forbidden this episode. "
        "The obvious route is intentionally wrong."
    )
    return Episode(
        episode_id=f"inhib_d{difficulty}_{seed}",
        seed=seed,
        family=Family.INHIBITORY_CONTROL.value,
        difficulty=difficulty,
        title="Suppress the Habitual Shortcut",
        instructions=instructions,
        max_steps=optimal_steps + 5,
        initial_state=initial_state,
        hidden_state=hidden_state,
        scoring_metadata={"optimal_steps": optimal_steps},
    )



def generate_planning_episode(rng: random.Random, difficulty: int, seed: int) -> Episode:
    difficulty = max(1, min(4, int(difficulty)))
    sample = rng.choice(["sealed_package", "audit_box", "specimen_case"])

    planning_subtype = "verify_then_deliver_short"

    if difficulty == 1:
        room_contents = {
            "entry": [],
            "storage": [sample],
            "inspection_room": ["scanner"],
            "vault": [],
            "intake_window": ["routing_stub"],
        }
        doors = {
            "entry": {
                "storage": {"color": "yellow", "unlocked": True},
                "intake_window": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "storage": {
                "inspection_room": {"color": "white", "unlocked": True},
                "staging_alcove": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "intake_window": {
                "entry": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "storage": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "staging_alcove": {
                "storage": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "inspection_room": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "inspection_room": {
                "vault": {"color": "green", "unlocked": True},
                "quarantine_desk": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True},
                "seal_review": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "quarantine_desk": {"inspection_room": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}},
            "seal_review": {
                "inspection_room": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "vault": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "vault": {},
        }
        oracle_plan = [
            {"action": "MOVE", "target": "storage"},
            {"action": "PICK", "target": sample},
            {"action": "MOVE", "target": "inspection_room"},
            {"action": "PICK", "target": "scanner"},
            {"action": "USE", "target": "scanner"},
            {"action": "MOVE", "target": "vault"},
            {"action": "DROP", "target": sample},
        ]

    elif difficulty == 2:
        planning_subtype = "charge_then_verify"
        room_contents = {
            "entry": [],
            "storage": ["battery_pack", sample],
            "charging_station": ["scanner"],
            "inspection_room": [],
            "vault": [],
            "intake_window": ["routing_stub"],
        }
        doors = {
            "entry": {
                "storage": {"color": "yellow", "unlocked": True},
                "intake_window": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "storage": {
                "charging_station": {"color": "yellow", "unlocked": True},
                "staging_alcove": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "intake_window": {
                "entry": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "storage": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "staging_alcove": {
                "storage": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "charging_station": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "charging_station": {
                "inspection_room": {"color": "blue", "unlocked": False, "requires": "battery_pack"},
                "service_console": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "service_console": {
                "charging_station": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "inspection_room": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "inspection_room": {
                "vault": {"color": "green", "unlocked": True},
                "seal_review": {"color": "silver", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "seal_review": {
                "inspection_room": {"color": "silver", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "vault": {"color": "silver", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "vault": {},
        }
        oracle_plan = [
            {"action": "MOVE", "target": "storage"},
            {"action": "PICK", "target": "battery_pack"},
            {"action": "PICK", "target": sample},
            {"action": "MOVE", "target": "charging_station"},
            {"action": "USE", "target": "battery_pack"},
            {"action": "PICK", "target": "scanner"},
            {"action": "MOVE", "target": "inspection_room"},
            {"action": "USE", "target": "scanner"},
            {"action": "MOVE", "target": "vault"},
            {"action": "DROP", "target": sample},
        ]

    else:
        room_contents = {
            "entry": [],
            "storage": ["battery_pack", sample],
            "charging_station": [],
            "lab": ["scanner"],
            "inspection_room": [],
            "vault": [],
            "intake_window": ["routing_stub"],
        }
        doors = {
            "entry": {
                "storage": {"color": "yellow", "unlocked": True},
                "intake_window": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
            "storage": {"charging_station": {"color": "yellow", "unlocked": True}},
            "charging_station": {"lab": {"color": "blue", "unlocked": False, "requires": "battery_pack"}},
            "lab": {"inspection_room": {"color": "white", "unlocked": True}},
            "inspection_room": {"vault": {"color": "green", "unlocked": True}},
            "vault": {},
            "intake_window": {
                "entry": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "storage": {"color": "amber", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            },
        }

        room_contents["staging_alcove"] = ["packing_slip"]
        doors["storage"]["staging_alcove"] = {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True}
        doors["staging_alcove"] = {
            "storage": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            "charging_station": {"color": "beige", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
        }
        room_contents["quarantine_desk"] = ["review_slip"]
        doors["inspection_room"]["quarantine_desk"] = {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}
        doors["quarantine_desk"] = {"inspection_room": {"color": "silver", "unlocked": True, "decoy": True, "misleading": True}}
        room_contents["service_console"] = ["power_stub"]
        doors["charging_station"]["service_console"] = {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True}
        doors["service_console"] = {
            "charging_station": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            "lab": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
        }
        room_contents["seal_review"] = ["seal_form"]
        doors["inspection_room"]["seal_review"] = {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True}
        doors["seal_review"] = {
            "inspection_room": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            "vault": {"color": "teal", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
        }

        planning_subtype = "tool_room_chain"
        room_contents["maintenance_bay"] = ["maintenance_note"]
        doors["charging_station"]["maintenance_bay"] = {"color": "amber", "unlocked": True, "decoy": True, "premature": True}
        doors["maintenance_bay"] = {"charging_station": {"color": "amber", "unlocked": True, "decoy": True, "premature": True}}
        room_contents["preview_vault"] = ["fake_manifest"]
        doors["lab"]["preview_vault"] = {"color": "orange", "unlocked": True, "decoy": True, "premature": True}
        doors["preview_vault"] = {"lab": {"color": "orange", "unlocked": True, "decoy": True, "premature": True}}
        room_contents["storage"].insert(1, "access_card")
        doors["inspection_room"]["vault"] = {"color": "green", "unlocked": False, "requires": "access_card"}

        if difficulty >= 3:
            room_contents["tool_room"] = ["scanner"]
            room_contents["lab"] = []
            doors["lab"] = {"tool_room": {"color": "white", "unlocked": True}}
            room_contents["diagnostics_nook"] = ["calibration_stub"]
            doors["tool_room"] = {
                "inspection_room": {"color": "white", "unlocked": True},
                "diagnostics_nook": {"color": "cyan", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            }
            doors["diagnostics_nook"] = {
                "tool_room": {"color": "cyan", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "inspection_room": {"color": "cyan", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            }
            room_contents["service_tunnel"] = ["obsolete_badge"]
            doors["charging_station"]["service_tunnel"] = {"color": "red", "unlocked": True, "decoy": True}
            doors["service_tunnel"] = {"charging_station": {"color": "red", "unlocked": True, "decoy": True}}
            oracle_plan = [
                {"action": "MOVE", "target": "storage"},
                {"action": "PICK", "target": "battery_pack"},
                {"action": "PICK", "target": "access_card"},
                {"action": "PICK", "target": sample},
                {"action": "MOVE", "target": "charging_station"},
                {"action": "USE", "target": "battery_pack"},
                {"action": "MOVE", "target": "lab"},
                {"action": "MOVE", "target": "tool_room"},
                {"action": "PICK", "target": "scanner"},
                {"action": "MOVE", "target": "inspection_room"},
                {"action": "USE", "target": "scanner"},
                {"action": "USE", "target": "access_card"},
                {"action": "MOVE", "target": "vault"},
                {"action": "DROP", "target": sample},
            ]
        if difficulty >= 4:
            planning_subtype = "locker_then_tool_chain"
            room_contents["locker"] = ["access_card"]
            room_contents["storage"] = ["battery_pack", sample]
            room_contents["credential_kiosk"] = ["archived_badge_log"]
            doors["charging_station"]["locker"] = {"color": "purple", "unlocked": True}
            doors["locker"] = {
                "charging_station": {"color": "purple", "unlocked": True},
                "credential_kiosk": {"color": "purple", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            }
            doors["credential_kiosk"] = {
                "locker": {"color": "purple", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "charging_station": {"color": "purple", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            }
            room_contents["receipt_window"] = ["receipt_stub"]
            doors["vault"]["receipt_window"] = {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}
            doors["receipt_window"] = {"vault": {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}}
            room_contents["badge_archive"] = ["archived_badge_log"]
            doors["locker"]["badge_archive"] = {"color": "violet", "unlocked": True, "decoy": True, "premature": True, "misleading": True}
            doors["badge_archive"] = {
                "locker": {"color": "violet", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
                "charging_station": {"color": "violet", "unlocked": True, "decoy": True, "premature": True, "misleading": True},
            }
            oracle_plan = [
                {"action": "MOVE", "target": "storage"},
                {"action": "PICK", "target": "battery_pack"},
                {"action": "PICK", "target": sample},
                {"action": "MOVE", "target": "charging_station"},
                {"action": "USE", "target": "battery_pack"},
                {"action": "MOVE", "target": "locker"},
                {"action": "PICK", "target": "access_card"},
                {"action": "MOVE", "target": "charging_station"},
                {"action": "MOVE", "target": "lab"},
                {"action": "MOVE", "target": "tool_room"},
                {"action": "PICK", "target": "scanner"},
                {"action": "MOVE", "target": "inspection_room"},
                {"action": "USE", "target": "scanner"},
                {"action": "USE", "target": "access_card"},
                {"action": "MOVE", "target": "vault"},
                {"action": "DROP", "target": sample},
            ]

    add_decoy_branches(rng, room_contents, doors, ["entry", "storage", "charging_station", "lab", "inspection_room", "vault"], max(1, difficulty), "plan")
    optimal_steps = len(oracle_plan)
    initial_state = {
        "current_room": "entry",
        "inventory": [],
        "flags": {"battery_charged": False, "package_verified": False},
        "switch_applied": False,
        "switch_announced": False,
        "active_rule": "complete_preconditions_before_delivery",
        "goal": {"type": "deliver_item", "item": sample, "target_room": "vault", "requires_verified": True},
        "room_contents": room_contents,
        "doors": doors,
    }
    hidden_state = {
        "switch_event": None,
        "difficulty_profile": {
            "tier": difficulty,
            "needs_access_card": difficulty >= 3,
            "scanner_in_tool_room": difficulty >= 3,
            "locker_branch": difficulty >= 4,
            "short_chain": difficulty <= 2,
        },
        "planning_subtype": planning_subtype,
        "oracle_plan": oracle_plan,
    }
    instructions = f"Deliver {sample} to the vault, but only verified items may be deposited. Prepare prerequisites before committing to the final delivery path."
    return Episode(
        episode_id=f"plan_d{difficulty}_{seed}",
        seed=seed,
        family=Family.PLANNING.value,
        difficulty=difficulty,
        title="Planning with Sequential Preconditions",
        instructions=instructions,
        max_steps=optimal_steps + 5,
        initial_state=initial_state,
        hidden_state=hidden_state,
        scoring_metadata={"optimal_steps": optimal_steps},
    )



def generate_conflict_resolution_episode(rng: random.Random, difficulty: int, seed: int) -> Episode:
    difficulty = max(1, min(4, int(difficulty)))
    item = rng.choice(["sterile_sample", "clean_sample", "bio_sample"])
    room_contents = {"start": [item], "safe_hall": [], "preserve_lab": [], "risky_tunnel": []}
    doors = {
        "start": {
            "safe_hall": {"color": "green", "unlocked": True},
            "risky_tunnel": {"color": "red", "unlocked": True, "unsafe_shortcut": True},
        },
        "safe_hall": {"preserve_lab": {"color": "green", "unlocked": True}},
        "risky_tunnel": {"preserve_lab": {"color": "red", "unlocked": True, "unsafe_shortcut": True}},
        "preserve_lab": {},
    }
    doors["safe_hall"]["quick_rinse"] = {"color": "orange", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
    room_contents["quick_rinse"] = []
    doors["quick_rinse"] = {"preserve_lab": {"color": "orange", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
    doors["safe_hall"]["courtesy_rinse"] = {"color": "teal", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
    room_contents["courtesy_rinse"] = []
    doors["courtesy_rinse"] = {"preserve_lab": {"color": "teal", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
    oracle_plan = [{"action": "PICK", "target": item}, {"action": "MOVE", "target": "safe_hall"}, {"action": "MOVE", "target": "preserve_lab"}, {"action": "DROP", "target": item}]
    room_contents["seal_counter"] = ["seal_tag"]
    doors["preserve_lab"]["seal_counter"] = {"color": "teal", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
    doors["seal_counter"] = {"preserve_lab": {"color": "teal", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
    room_contents["courtesy_lane"] = []
    doors["start"]["courtesy_lane"] = {"color": "teal", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
    doors["courtesy_lane"] = {"preserve_lab": {"color": "teal", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
    room_contents["sterile_preview"] = []
    doors["start"]["sterile_preview"] = {"color": "silver", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
    doors["sterile_preview"] = {"preserve_lab": {"color": "silver", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
    if difficulty >= 2:
        room_contents["decon"] = []
        room_contents["bypass_pipe"] = []
        doors["safe_hall"] = {"decon": {"color": "green", "unlocked": True}, "bypass_pipe": {"color": "orange", "unlocked": True, "unsafe_shortcut": True}}
        doors["decon"] = {"preserve_lab": {"color": "green", "unlocked": True}}
        room_contents["sample_log"] = ["log_stub"]
        doors["decon"]["sample_log"] = {"color": "gray", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
        doors["sample_log"] = {"decon": {"color": "gray", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
        doors["bypass_pipe"] = {"preserve_lab": {"color": "orange", "unlocked": True, "unsafe_shortcut": True}}
        oracle_plan = [{"action": "PICK", "target": item}, {"action": "MOVE", "target": "safe_hall"}, {"action": "MOVE", "target": "decon"}, {"action": "MOVE", "target": "preserve_lab"}, {"action": "DROP", "target": item}]
    if difficulty >= 3:
        room_contents["fast_lift"] = []
        doors["start"]["fast_lift"] = {"color": "orange", "unlocked": True, "unsafe_shortcut": True}
        doors["fast_lift"] = {"preserve_lab": {"color": "orange", "unlocked": True, "unsafe_shortcut": True}}
    if difficulty >= 4:
        room_contents["time_saver"] = []
        doors["decon"]["time_saver"] = {"color": "orange", "unlocked": True, "unsafe_shortcut": True}
        doors["time_saver"] = {"preserve_lab": {"color": "orange", "unlocked": True, "unsafe_shortcut": True}}
        room_contents["safe_hall"].append("clean_report")
        room_contents["courtesy_fastlane"] = []
        doors["decon"]["courtesy_fastlane"] = {"color": "purple", "unlocked": True, "unsafe_shortcut": True, "misleading": True}
        doors["courtesy_fastlane"] = {"preserve_lab": {"color": "purple", "unlocked": True, "unsafe_shortcut": True, "misleading": True}, "decon": {"color": "purple", "unlocked": True, "unsafe_shortcut": True, "misleading": True}}
    add_decoy_branches(rng, room_contents, doors, ["start", "safe_hall"] + (["decon"] if "decon" in room_contents else []) + ["preserve_lab"], max(3, difficulty + 1), "conflict")
    optimal_steps = len(oracle_plan)
    initial_state = {"current_room": "start", "inventory": [], "flags": {"sample_contaminated": False}, "switch_applied": False, "switch_announced": False, "active_rule": "prioritize_integrity_over_speed", "goal": {"type": "deliver_item", "item": item, "target_room": "preserve_lab", "requires_verified": False}, "room_contents": room_contents, "doors": doors}
    hidden_state = {"priority_note": "Primary objective: preserve sample integrity. Secondary objective: minimize time. Unsafe shortcuts may be faster but can contaminate the sample.", "switch_event": None, "difficulty_profile": {"tier": difficulty, "safe_depth": optimal_steps, "shortcut_layers": 2 + int(difficulty >= 3) + int(difficulty >= 4)}, "family_variant": f"conflict_safe_depth_{optimal_steps}", "oracle_plan": oracle_plan}
    instructions = f"Deliver {item} to preserve_lab. Preserve sample integrity even if the safe route is longer."
    return Episode(episode_id=f"conflict_d{difficulty}_{seed}", seed=seed, family=Family.CONFLICT_RESOLUTION.value, difficulty=difficulty, title="Resolve Priority Conflict Between Safety and Speed", instructions=instructions, max_steps=optimal_steps + 4, initial_state=initial_state, hidden_state=hidden_state, scoring_metadata={"optimal_steps": optimal_steps})




def generate_working_memory_episode(rng: random.Random, difficulty: int, seed: int) -> Episode:
    difficulty = max(1, min(4, int(difficulty)))
    item = rng.choice(["memory_capsule", "sealed_note", "archive_chip"])

    if difficulty == 1:
        room_contents = {"start": [item, "blue_badge", "green_badge"], "corridor": [], "memory_vault": []}
        doors = {
            "start": {"corridor": {"color": "gray", "unlocked": True}},
            "corridor": {"memory_vault": {"color": "blue", "unlocked": False, "requires": "blue_badge", "memory_gate": True}},
            "memory_vault": {},
        }
        briefing_note = "Memory rule: BLUE unlocks the final vault gate. Pick up the delivery item first, then the correct badge before committing to the vault route."
        switch_event = None
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "PICK", "target": "blue_badge"},
            {"action": "MOVE", "target": "corridor"},
            {"action": "USE", "target": "blue_badge"},
            {"action": "MOVE", "target": "memory_vault"},
            {"action": "DROP", "target": item},
        ]
        wm_subtype = "static_badge_rule"

    elif difficulty == 2:
        room_contents = {"start": [item, "blue_badge"], "corridor": ["green_badge"], "memory_vault": []}
        doors = {
            "start": {"corridor": {"color": "gray", "unlocked": True}},
            "corridor": {"memory_vault": {"color": "green", "unlocked": False, "requires": "green_badge", "memory_gate": True}},
            "memory_vault": {},
        }
        briefing_note = "Initial memory rule: BLUE is correct. If an alert says INVERT, obey the latest alert instead of the original mapping. GREEN becomes correct immediately."
        switch_event = {"at_step": 1, "new_rule": "invert_badges", "message": "ALERT: INVERT now. GREEN is the correct badge from this point onward."}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "PICK", "target": "green_badge"},
            {"action": "MOVE", "target": "corridor"},
            {"action": "USE", "target": "green_badge"},
            {"action": "MOVE", "target": "memory_vault"},
            {"action": "DROP", "target": item},
        ]
        wm_subtype = "direct_inverted_badge"

    elif difficulty == 3:
        room_contents = {"start": [item, "blue_badge"], "corridor": ["relay_key", "green_badge"], "buffer_room": [], "memory_vault": []}
        doors = {
            "start": {"corridor": {"color": "gray", "unlocked": True}},
            "corridor": {"buffer_room": {"color": "orange", "unlocked": False, "requires": "relay_key", "sequence_gate": True}},
            "buffer_room": {"memory_vault": {"color": "green", "unlocked": False, "requires": "green_badge", "memory_gate": True}},
            "memory_vault": {},
        }
        briefing_note = "Memory checklist: latest alert overrides earlier badge rules. Also remember to collect and use the route key before leaving the corridor."
        switch_event = {"at_step": 1, "new_rule": "invert_badges", "message": "ALERT: INVERT now. GREEN is the correct badge. Do not forget the corridor route key."}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "MOVE", "target": "corridor"},
            {"action": "PICK", "target": "relay_key"},
            {"action": "PICK", "target": "green_badge"},
            {"action": "USE", "target": "relay_key"},
            {"action": "MOVE", "target": "buffer_room"},
            {"action": "USE", "target": "green_badge"},
            {"action": "MOVE", "target": "memory_vault"},
            {"action": "DROP", "target": item},
        ]
        wm_subtype = "route_key_then_badge"

    else:
        room_contents = {"start": [item, "blue_badge", "red_badge"], "corridor": ["relay_key", "green_badge"], "buffer_room": [], "anteroom": [], "memory_vault": []}
        doors = {
            "start": {"corridor": {"color": "gray", "unlocked": True}},
            "corridor": {"buffer_room": {"color": "orange", "unlocked": False, "requires": "relay_key", "sequence_gate": True}},
            "buffer_room": {"anteroom": {"color": "gray", "unlocked": True}},
            "anteroom": {"memory_vault": {"color": "green", "unlocked": False, "requires": "green_badge", "memory_gate": True}},
            "memory_vault": {},
        }
        briefing_note = "Memory checklist: obey the latest alert, remember the corridor route key, and ignore RED even if nearby notes imply urgency."
        switch_event = {"at_step": 1, "new_rule": "invert_badges", "message": "ALERT: INVERT now. GREEN is correct. RED remains a distractor. Corridor route key is still required first."}
        oracle_plan = [
            {"action": "PICK", "target": item},
            {"action": "MOVE", "target": "corridor"},
            {"action": "PICK", "target": "relay_key"},
            {"action": "PICK", "target": "green_badge"},
            {"action": "USE", "target": "relay_key"},
            {"action": "MOVE", "target": "buffer_room"},
            {"action": "MOVE", "target": "anteroom"},
            {"action": "USE", "target": "green_badge"},
            {"action": "MOVE", "target": "memory_vault"},
            {"action": "DROP", "target": item},
        ]
        wm_subtype = "deep_route_key_and_badge"

    room_contents.setdefault("orientation_kiosk", ["obsolete_badge_hint"])
    doors.setdefault("start", {})["orientation_kiosk"] = {"color": "yellow", "unlocked": True, "decoy": True, "misleading": True}
    doors.setdefault("orientation_kiosk", {})["start"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
    if "corridor" in room_contents and difficulty >= 3:
        room_contents.setdefault("memo_wall", ["stale_map"])
        doors.setdefault("corridor", {})["memo_wall"] = {"color": "yellow", "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault("memo_wall", {})["corridor"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
        room_contents.setdefault("badge_bench", ["badge_stub"])
        doors.setdefault("corridor", {})["badge_bench"] = {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault("badge_bench", {})["corridor"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
    if "buffer_room" in room_contents:
        room_contents.setdefault("buffer_alcove", ["old_badge_note"])
        doors.setdefault("buffer_room", {})["buffer_alcove"] = {"color": "yellow", "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault("buffer_alcove", {})["buffer_room"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
        room_contents.setdefault("route_plaque", ["route_stub"])
        doors.setdefault("buffer_room", {})["route_plaque"] = {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault("route_plaque", {})["buffer_room"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
    if "anteroom" in room_contents:
        room_contents.setdefault("color_poster", ["red_hint"])
        doors.setdefault("anteroom", {})["color_poster"] = {"color": "yellow", "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault("color_poster", {})["anteroom"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
        room_contents.setdefault("badge_mirror", ["green_echo"])
        doors.setdefault("anteroom", {})["badge_mirror"] = {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}
        doors.setdefault("badge_mirror", {})["anteroom"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}
    room_contents.setdefault("memory_receipt", ["receipt_stub"])
    doors.setdefault("memory_vault", {})["memory_receipt"] = {"color": "teal", "unlocked": True, "decoy": True, "misleading": True}
    doors.setdefault("memory_receipt", {})["memory_vault"] = {"color": "return", "unlocked": True, "decoy": True, "misleading": True}

    add_decoy_branches(
        rng,
        room_contents,
        doors,
        [room for room in ["start", "corridor", "relay_room", "buffer_room", "anteroom", "memory_vault"] if room in room_contents],
        max(1, difficulty - 1),
        "wm",
    )
    optimal_steps = len(oracle_plan)
    initial_state = {
        "current_room": "start",
        "inventory": [],
        "flags": {"wm_wrong_badge_uses": 0},
        "switch_applied": False,
        "switch_announced": False,
        "active_rule": "remember_initial_badge_rule",
        "goal": {"type": "deliver_item", "item": item, "target_room": "memory_vault", "requires_verified": False},
        "room_contents": room_contents,
        "doors": doors,
    }
    hidden_state = {
        "briefing_note": briefing_note,
        "switch_event": switch_event,
        "difficulty_profile": {"tier": difficulty, "inversion": difficulty >= 2, "route_key": difficulty >= 3, "extra_distractor_badge": difficulty >= 4},
        "wm_subtype": wm_subtype,
        "oracle_plan": oracle_plan,
    }
    instructions = f"Deliver {item} to memory_vault. Remember the latest badge rule and any delayed route prerequisites before unlocking the final approach."
    return Episode(
        episode_id=f"wm_d{difficulty}_{seed}",
        seed=seed,
        family=Family.WORKING_MEMORY.value,
        difficulty=difficulty,
        title="Working Memory for Delayed Rule Application",
        instructions=instructions,
        max_steps=optimal_steps + 4,
        initial_state=initial_state,
        hidden_state=hidden_state,
        scoring_metadata={"optimal_steps": optimal_steps},
    )


def oracle_executor_policy(env: RuleShiftEnv) -> Action:
    oracle_plan = env.hidden.get("oracle_plan")
    if isinstance(oracle_plan, list) and env.step_id < len(oracle_plan):
        return Action.from_any(oracle_plan[env.step_id])
    return heuristic_policy(env)


def heuristic_policy(env: RuleShiftEnv) -> Action:
    obs = env.observe()
    room = obs["current_room"]
    inv = set(obs["inventory"])
    goal = obs["goal"]
    family = obs["family"]
    difficulty = int(obs.get("difficulty", 1))
    item = goal.get("item")
    flags = env.state.setdefault("flags", {})

    def mark_once(key: str) -> bool:
        if flags.get(key):
            return False
        flags[key] = True
        return True

    if family == Family.GOAL_MAINTENANCE.value:
        main_path = [r for r in ["focus_hall", "checkpoint", "relay_room", "final_corridor", "archive"] if r in obs["room_contents"]]
        if room == "start" and item in obs["room_contents"].get("start", []):
            return Action("PICK", item)
        if room == "archive" and item in inv:
            return Action("DROP", item)
        if room == "start" and difficulty >= 3 and item in inv and mark_once("gm_bonus_detour") and "bonus_room" in obs["doors"].get("start", {}):
            return Action("MOVE", "bonus_room")
        if room == "start":
            return Action("MOVE", main_path[0])
        if room in {"bonus_room", "side_office", "admin_desk", "urgent_terminal_room", "reward_counter"}:
            back = sorted(obs["doors"].get(room, {}))[0]
            if difficulty >= 4 and mark_once(f"report_{room}"):
                return Action("REPORT", "goal")
            return Action("MOVE", back)
        if room in main_path:
            if difficulty >= 4 and room == "checkpoint" and mark_once("gm_checkpoint_reconfirm"):
                return Action("REPORT", "goal")
            idx = main_path.index(room)
            if idx < len(main_path) - 1:
                return Action("MOVE", main_path[idx + 1])
            if item in inv:
                return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.COGNITIVE_FLEXIBILITY.value:
        route_names = env.hidden.get("route_names", {})
        red_room = route_names.get("red", "red_hall")
        blue_room = route_names.get("blue", "blue_hall")
        blue_buffer = route_names.get("blue_buffer", "blue_buffer")
        switch_active = bool(obs.get("system_alert")) or bool(env.state.get("switch_applied"))
        prefer_blue = switch_active
        if room == "start" and item in obs["room_contents"].get("start", []):
            return Action("PICK", item)
        if room == "start" and difficulty >= 4:
            door = obs["doors"].get("start", {}).get(blue_room, {})
            if not door.get("unlocked", False) and door.get("requires") == "blue_token":
                if "blue_token" not in inv and "blue_token" in obs["room_contents"].get("start", []):
                    return Action("PICK", "blue_token")
                if "blue_token" in inv:
                    return Action("USE", "blue_token")
        if room == "start":
            if switch_active and difficulty >= 2 and mark_once("flex_confirm_switch"):
                return Action("REPORT", "goal")
            return Action("MOVE", blue_room if prefer_blue else red_room)
        if room == red_room:
            if prefer_blue and difficulty >= 3 and mark_once("flex_stale_pause"):
                return Action("REPORT", "goal")
            if prefer_blue and "start" in obs["doors"].get(red_room, {}):
                return Action("MOVE", "start")
            if "vault" in obs["doors"].get(red_room, {}):
                return Action("MOVE", "vault")
        if room == blue_room:
            if difficulty >= 4 and mark_once("flex_blue_recheck"):
                return Action("REPORT", "goal")
            if blue_buffer in obs["doors"].get(blue_room, {}):
                return Action("MOVE", blue_buffer)
            if "vault" in obs["doors"].get(blue_room, {}):
                return Action("MOVE", "vault")
        if room == blue_buffer:
            return Action("MOVE", "vault")
        if room == "gray_lane":
            return Action("MOVE", "vault")
        if room == "vault" and item in inv:
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.WORKING_MEMORY.value:
        target_badge = "blue_badge"
        if obs.get("system_alert") and "GREEN" in obs.get("system_alert", ""):
            target_badge = "green_badge"
        elif difficulty >= 2 and env.state.get("switch_applied"):
            target_badge = "green_badge"

        def _door(src: str, dst: str) -> Dict[str, Any]:
            return obs["doors"].get(src, {}).get(dst, {})

        def _door_unlocked(src: str, dst: str) -> bool:
            return bool(_door(src, dst).get("unlocked", False))

        if room == "start":
            if difficulty >= 2 and target_badge != "blue_badge" and "blue_badge" in obs["room_contents"].get("start", []) and "blue_badge" not in inv and mark_once("wm_anchor_blue_pick"):
                return Action("PICK", "blue_badge")
            if item in obs["room_contents"].get("start", []):
                return Action("PICK", item)
            if target_badge in obs["room_contents"].get("start", []) and target_badge not in inv:
                return Action("PICK", target_badge)
            return Action("MOVE", "corridor")
        if room == "corridor":
            if "relay_key" in obs["room_contents"].get("corridor", []) and "relay_key" not in inv:
                return Action("PICK", "relay_key")
            if "buffer_room" in obs["doors"].get("corridor", {}):
                if _door_unlocked("corridor", "buffer_room"):
                    return Action("MOVE", "buffer_room")
                if "relay_key" in inv:
                    return Action("USE", "relay_key")
            if "relay_room" in obs["doors"].get("corridor", {}):
                return Action("MOVE", "relay_room")
            if "memory_vault" in obs["doors"].get("corridor", {}):
                if _door_unlocked("corridor", "memory_vault"):
                    return Action("MOVE", "memory_vault")
                if target_badge != "blue_badge" and "blue_badge" in inv and mark_once("wm_wrong_badge_try"):
                    return Action("USE", "blue_badge")
                if target_badge in inv:
                    return Action("USE", target_badge)
            return Action("WAIT", "none")
        if room == "relay_room":
            if _door_unlocked("relay_room", "memory_vault"):
                return Action("MOVE", "memory_vault")
            if target_badge != "blue_badge" and "blue_badge" in inv and mark_once("wm_wrong_badge_try"):
                return Action("USE", "blue_badge")
            if target_badge in inv:
                return Action("USE", target_badge)
            return Action("WAIT", "none")
        if room == "buffer_room":
            if "anteroom" in obs["doors"].get("buffer_room", {}):
                return Action("MOVE", "anteroom")
            if _door_unlocked("buffer_room", "memory_vault"):
                return Action("MOVE", "memory_vault")
            if target_badge != "blue_badge" and "blue_badge" in inv and mark_once("wm_wrong_badge_try"):
                return Action("USE", "blue_badge")
            if target_badge in inv:
                return Action("USE", target_badge)
            return Action("WAIT", "none")
        if room == "anteroom":
            if _door_unlocked("anteroom", "memory_vault"):
                return Action("MOVE", "memory_vault")
            if target_badge != "blue_badge" and "blue_badge" in inv and mark_once("wm_wrong_badge_try"):
                return Action("USE", "blue_badge")
            if target_badge in inv:
                return Action("USE", target_badge)
            return Action("WAIT", "none")
        if room == "memory_vault" and item in inv:
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.INHIBITORY_CONTROL.value:
        if room == "dock" and item in obs["room_contents"].get("dock", []):
            return Action("PICK", item)
        if room == "dock":
            if difficulty >= 2 and mark_once("inspect_trap"):
                return Action("INSPECT", "red_corridor")
            return Action("MOVE", "safe_corridor")
        if room == "safe_corridor":
            if difficulty >= 3 and mark_once("report_routes"):
                return Action("REPORT", "routes")
            if difficulty >= 4 and "shortcut_gate" in obs["doors"].get("safe_corridor", {}):
                return Action("MOVE", "shortcut_gate")
            if "safe_mid" in obs["doors"].get("safe_corridor", {}):
                return Action("MOVE", "safe_mid")
            return Action("MOVE", "secure_zone")
        if room == "safe_mid":
            return Action("MOVE", "secure_zone")
        if room == "secure_zone" and item in inv:
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.PLANNING.value:
        f = env.state["flags"]
        if room == "entry":
            if mark_once("planning_recheck"):
                return Action("REPORT", "goal")
            return Action("MOVE", "storage")
        if room == "storage":
            if "battery_pack" in obs["room_contents"].get("storage", []) and "battery_pack" not in inv:
                return Action("PICK", "battery_pack")
            if "access_card" in obs["room_contents"].get("storage", []) and "access_card" not in inv:
                return Action("PICK", "access_card")
            if item in obs["room_contents"].get("storage", []) and item not in inv:
                return Action("PICK", item)
            return Action("MOVE", "charging_station")
        if room == "charging_station":
            if not f.get("battery_charged", False) and "battery_pack" in inv:
                return Action("USE", "battery_pack")
            if difficulty >= 4 and "access_card" not in inv and "locker" in obs["doors"].get("charging_station", {}):
                return Action("MOVE", "locker")
            return Action("MOVE", "lab")
        if room == "locker":
            if "access_card" in obs["room_contents"].get("locker", []) and "access_card" not in inv:
                return Action("PICK", "access_card")
            return Action("MOVE", "charging_station")
        if room == "lab":
            if difficulty >= 3 and "preview_vault" in obs["doors"].get("lab", {}) and mark_once("planning_preview_bias"):
                return Action("MOVE", "preview_vault")
            if "scanner" in obs["room_contents"].get("lab", []) and "scanner" not in inv:
                return Action("PICK", "scanner")
            if "tool_room" in obs["doors"].get("lab", {}):
                return Action("MOVE", "tool_room")
            return Action("MOVE", "inspection_room")
        if room == "preview_vault":
            return Action("MOVE", "lab")
        if room == "tool_room":
            if "scanner" in obs["room_contents"].get("tool_room", []) and "scanner" not in inv:
                return Action("PICK", "scanner")
            return Action("MOVE", "inspection_room")
        if room == "inspection_room":
            if difficulty >= 2 and not f.get("package_verified", False) and mark_once("planning_pause_before_verify"):
                return Action("REPORT", "goal")
            if not f.get("package_verified", False) and "scanner" in inv:
                return Action("USE", "scanner")
            if env._can_use_item_now("access_card") and "access_card" in inv:
                return Action("USE", "access_card")
            if "vault" in obs["doors"].get("inspection_room", {}):
                door = obs["doors"]["inspection_room"]["vault"]
                if door.get("unlocked", False):
                    return Action("MOVE", "vault")
            return Action("WAIT", "none")
        if room == "vault" and item in inv:
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.CONFLICT_RESOLUTION.value:
        if room == "start" and item in obs["room_contents"].get("start", []):
            return Action("PICK", item)
        if room == "start":
            return Action("MOVE", "safe_hall")
        if room == "safe_hall":
            if difficulty >= 3 and mark_once("integrity_check"):
                return Action("REPORT", "goal")
            if "decon" in obs["doors"].get("safe_hall", {}):
                return Action("MOVE", "decon")
            return Action("MOVE", "preserve_lab")
        if room == "decon":
            if difficulty >= 4 and "time_saver" in obs["doors"].get("decon", {}) and mark_once("conflict_tempted_shortcut"):
                return Action("MOVE", "time_saver")
            return Action("MOVE", "preserve_lab")
        if room in {"risky_tunnel", "fast_lift", "time_saver"}:
            return Action("MOVE", "preserve_lab")
        if room == "preserve_lab" and item in inv:
            return Action("DROP", item)
        return Action("WAIT", "none")

    return Action("WAIT", "none")


def build_prompt_from_observation(observation: Dict[str, Any]) -> str:
    goal = observation["goal"]
    inventory = list(observation.get("inventory", []))
    current_room = observation.get("current_room", "")
    current_room_items = list(observation.get("room_contents", {}).get(current_room, []))
    outgoing_routes = observation.get("doors", {}).get(current_room, {})
    family = observation.get("family")

    goal_item = goal.get("item")
    target_room = goal.get("target_room")
    unsatisfied: List[str] = []
    if goal_item and goal_item not in inventory:
        if goal_item in current_room_items:
            unsatisfied.append(f"pick {goal_item} before leaving")
        else:
            unsatisfied.append(f"carry {goal_item} before final delivery")
    for route, info in sorted(outgoing_routes.items()):
        if info.get("requires") and not info.get("unlocked", False):
            unsatisfied.append(f"{route} needs {info['requires']}")

    candidate_lines = [f'{item["option"]}. {item["action"]} {item["target"]}' for item in observation.get("candidate_actions", [])]
    alerts = [observation.get(k) for k in ["system_alert", "briefing_note", "priority_note", "focus_alert"] if observation.get(k)]

    if family == Family.WORKING_MEMORY.value:
        lines = [
            f"GOAL_ITEM: {goal_item}",
            f"TARGET_ROOM: {target_room}",
            f"ROOM: {current_room}",
            f"INVENTORY: {', '.join(inventory) if inventory else 'none'}",
            f"ROOM_ITEMS: {', '.join(current_room_items) if current_room_items else 'none'}",
        ]
        if alerts:
            lines.append(f"LATEST_ALERT: {alerts[-1]}")
        if unsatisfied:
            lines.append(f"NEEDS: {'; '.join(unsatisfied)}")
        return (
            "Choose exactly one next action.\n"
            "Use the latest alert if it overrides an older memory rule.\n"
            "If a required key or correct badge is available now, take or use it before moving away.\n"
            "Return exactly one integer option number.\n\n"
            + "\n".join(lines)
            + "\n\nOPTIONS:\n"
            + "\n".join(candidate_lines)
            + "\n\nANSWER:"
        )

    compact_state = {
        "family": family,
        "difficulty": observation["difficulty"],
        "step": observation["step"],
        "max_steps": observation["max_steps"],
        "goal_item": goal_item,
        "target_room": target_room,
        "current_room": current_room,
        "inventory": inventory,
        "current_room_contents": current_room_items,
        "active_rule": observation.get("active_rule"),
        "latest_alert": alerts[-1] if alerts else None,
        "unsatisfied_prerequisites": unsatisfied,
    }
    return (
        "Pick the single best NEXT action.\n"
        "Protect the main objective; ignore side quests unless they directly unlock progress.\n"
        "If a required item, key, badge, or tool is available now, take or use it before moving away.\n"
        "If the latest alert conflicts with an older note, follow the latest alert.\n"
        "Avoid REPORT / INSPECT / WAIT when an executable progress action exists.\n"
        "Return exactly one integer option number. No words, no JSON, no explanation.\n\n"
        f"STATE:\n{json.dumps(compact_state, ensure_ascii=False)}\n\n"
        "OPTIONS:\n" + "\n".join(candidate_lines) + "\n\nANSWER:"
    )

def scripted_policy_bad_habit(env: RuleShiftEnv) -> Action:
    obs = env.observe()
    room = obs["current_room"]
    family = obs["family"]
    goal = obs["goal"]
    item = goal.get("item")

    if family == Family.GOAL_MAINTENANCE.value:
        if room == "start" and item in obs["room_contents"]["start"]:
            return Action("PICK", item)
        if room == "start":
            return Action("MOVE", "bonus_room")
        if room == "bonus_room" and "shiny_token" in obs["room_contents"].get("bonus_room", []):
            return Action("PICK", "shiny_token")
        if room == "bonus_room":
            return Action("MOVE", "start")
        return Action("WAIT", "none")

    if family == Family.INHIBITORY_CONTROL.value:
        if room == "dock" and item in obs["room_contents"]["dock"]:
            return Action("PICK", item)
        if room == "dock":
            return Action("MOVE", "red_corridor")
        if room == "red_corridor":
            return Action("MOVE", "secure_zone")
        if room == "secure_zone":
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.COGNITIVE_FLEXIBILITY.value:
        route_names = env.hidden["route_names"]
        red_room = route_names["red"]
        if room == "start" and item in obs["room_contents"]["start"]:
            return Action("PICK", item)
        if room == "start":
            return Action("MOVE", red_room)
        if room == red_room:
            return Action("MOVE", "vault")
        if room == "vault":
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.PLANNING.value:
        if room == "entry":
            return Action("MOVE", "storage")
        if room == "storage" and item in obs["room_contents"]["storage"]:
            return Action("PICK", item)
        if room == "storage":
            return Action("MOVE", "charging_station")
        if room == "charging_station":
            return Action("MOVE", "lab")
        if room == "lab":
            return Action("MOVE", "inspection_room")
        if room == "inspection_room":
            return Action("MOVE", "vault")
        if room == "vault":
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.CONFLICT_RESOLUTION.value:
        if room == "start" and item in obs["room_contents"]["start"]:
            return Action("PICK", item)
        if room == "start":
            if "fast_lift" in obs["doors"].get("start", {}):
                return Action("MOVE", "fast_lift")
            return Action("MOVE", "risky_tunnel")
        if room == "risky_tunnel":
            return Action("MOVE", "preserve_lab")
        if room == "fast_lift":
            return Action("MOVE", "preserve_lab")
        if room == "decon" and "time_saver" in obs["doors"].get("decon", {}):
            return Action("MOVE", "time_saver")
        if room == "time_saver":
            return Action("MOVE", "preserve_lab")
        if room == "preserve_lab":
            return Action("DROP", item)
        return Action("WAIT", "none")

    if family == Family.WORKING_MEMORY.value:
        wrong_badge = "blue_badge"
        if room == "start" and item in obs["room_contents"]["start"]:
            return Action("PICK", item)
        if room == "start" and wrong_badge in obs["room_contents"]["start"]:
            return Action("PICK", wrong_badge)
        if room == "start":
            return Action("MOVE", "corridor")
        if room in {"corridor", "relay_room", "buffer_room", "anteroom"}:
            if wrong_badge in obs["inventory"]:
                return Action("USE", wrong_badge)
            next_room = "memory_vault"
            if room == "corridor" and "relay_room" in obs["doors"].get("corridor", {}):
                next_room = "relay_room"
            if room == "corridor" and "buffer_room" in obs["doors"].get("corridor", {}):
                next_room = "buffer_room"
            if room == "buffer_room" and "anteroom" in obs["doors"].get("buffer_room", {}):
                next_room = "anteroom"
            return Action("MOVE", next_room)
        if room == "memory_vault":
            return Action("DROP", item)
        return Action("WAIT", "none")

    return Action("WAIT", "none")
def parse_action_from_text(text: str, candidate_actions: Optional[List[Any]] = None) -> Action:
    normalized_candidates: List[Action] = []
    option_map: Dict[int, Action] = {}
    if candidate_actions:
        for idx, item in enumerate(candidate_actions, start=1):
            if isinstance(item, Action):
                act = item.normalized()
                normalized_candidates.append(act)
                option_map[idx] = act
            elif isinstance(item, dict) and "action" in item and "target" in item:
                act = Action.from_any(item)
                normalized_candidates.append(act)
                option_num = item.get("option", idx)
                try:
                    option_map[int(option_num)] = act
                except Exception:
                    option_map[idx] = act

    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|text)?", "", stripped).strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()

    def map_option_number(num: int) -> Optional[Action]:
        return option_map.get(int(num)) if option_map else None

    def map_option_string(s: str) -> Optional[Action]:
        if not s or not option_map:
            return None
        s = s.strip()
        first_line = s.splitlines()[0].strip() if s else ""
        for candidate in (s, first_line):
            if re.fullmatch(r"\d{1,3}", candidate):
                mapped = map_option_number(int(candidate))
                if mapped is not None:
                    return mapped
        m = re.search(r"\b(?:option|answer|choice)?\s*#?\s*(\d{1,3})\b", first_line, flags=re.IGNORECASE)
        if m:
            mapped = map_option_number(int(m.group(1)))
            if mapped is not None:
                return mapped
        nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", s[:160])]
        valid = [n for n in nums if n in option_map]
        if len(valid) == 1:
            return option_map[valid[0]]
        if valid:
            return option_map[valid[0]]
        return None

    mapped = map_option_string(stripped)
    if mapped is not None:
        return mapped

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            if option_map and "option" in obj:
                mapped = map_option_string(str(obj.get("option", "")))
                if mapped is not None:
                    return mapped
            if "action" in obj and "target" in obj:
                candidate = Action(str(obj["action"]), str(obj["target"])).normalized()
                for act in normalized_candidates:
                    if (act.action.upper(), act.target) == (candidate.action.upper(), candidate.target):
                        return act
            return Action.from_any(obj)
    except Exception:
        pass

    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        try:
            obj = json.loads(stripped[start_obj:end_obj + 1])
            if isinstance(obj, dict):
                if option_map and "option" in obj:
                    mapped = map_option_string(str(obj.get("option", "")))
                    if mapped is not None:
                        return mapped
                if "action" in obj and "target" in obj:
                    candidate = Action(str(obj["action"]), str(obj["target"])).normalized()
                    for act in normalized_candidates:
                        if (act.action.upper(), act.target) == (candidate.action.upper(), candidate.target):
                            return act
                return Action.from_any(obj)
        except Exception:
            pass

    action_match = re.search(r"action\s*[:=]\s*['\"]?([A-Za-z0-9_\-]+)", stripped, flags=re.IGNORECASE)
    target_match = re.search(r"target\s*[:=]\s*['\"]?([A-Za-z0-9_\-]+)", stripped, flags=re.IGNORECASE)
    if option_map and action_match and target_match:
        candidate = Action(action_match.group(1), target_match.group(1)).normalized()
        for act in normalized_candidates:
            if (act.action.upper(), act.target) == (candidate.action.upper(), candidate.target):
                return act

    synonym_map = {
        "GO": "MOVE", "MOVE": "MOVE", "WALK": "MOVE", "TRAVEL": "MOVE",
        "TAKE": "PICK", "PICK": "PICK", "GRAB": "PICK",
        "DROP": "DROP", "PUT": "DROP", "PLACE": "DROP",
        "USE": "USE", "SCAN": "USE", "VERIFY": "USE",
        "WAIT": "WAIT", "STAY": "WAIT", "HOLD": "WAIT",
        "LOOK": "INSPECT", "CHECK": "INSPECT", "EXAMINE": "INSPECT",
        "REPORT": "REPORT", "DESCRIBE": "REPORT", "SUMMARIZE": "REPORT",
    }
    upper = stripped.upper()
    detected_action = None
    for key, val in synonym_map.items():
        if re.search(rf"\b{re.escape(key)}\b", upper):
            detected_action = val
            break
    if normalized_candidates and detected_action:
        for act in normalized_candidates:
            if act.target != "none" and re.search(rf"\b{re.escape(act.target)}\b", stripped, flags=re.IGNORECASE):
                if act.action.upper() == detected_action:
                    return act

    raise ValueError(f"could not parse model output as candidate option or exact action: {text[:200]}")

def infer_best_progress_action(env: RuleShiftEnv) -> Optional[Action]:
    progress = env._progress_like_actions()
    if not progress:
        return None

    current_room = env.state.get("current_room", "")
    goal = env.state.get("goal", {})
    goal_item = goal.get("item")
    goal_room = goal.get("target_room")
    family = env.episode.family
    room_items = set(env.state.get("room_contents", {}).get(current_room, []))
    inventory = set(env.state.get("inventory", []))
    current_doors = env.state.get("doors", {}).get(current_room, {})

    # 1) finish immediately if possible
    for action in progress:
        if action.action == "DROP" and action.target == goal_item and current_room == goal_room:
            return action

    # 2) unlock or use mandatory tools
    for action in progress:
        if action.action == "USE":
            return action

    # 3) pick up directly relevant items before wandering
    important_items = {goal_item}
    if family == Family.PLANNING.value:
        important_items.update({"access_card", "scanner", "battery_pack"})
    elif family == Family.WORKING_MEMORY.value:
        important_items.update({"blue_badge", "green_badge", "red_badge"})
    for action in progress:
        if action.action == "PICK" and action.target in important_items:
            return action

    # 4) prefer direct move to goal room if available
    for action in progress:
        if action.action == "MOVE" and action.target == goal_room:
            return action

    # 5) planning-specific path hints
    preferred_room_order = {
        Family.PLANNING.value: ["locker", "credential_kiosk", "tool_room", "charging_station", "lab", "inspection_room", "delivery_zone"],
        Family.GOAL_MAINTENANCE.value: ["focus_hall", "checkpoint", "relay_room", "final_corridor", "archive"],
        Family.COGNITIVE_FLEXIBILITY.value: ["switch_hub", "blue_buffer", "blue_gate", "goal_room"],
        Family.CONFLICT_RESOLUTION.value: ["safe_corridor", "delivery_room", "sample_vault"],
        Family.WORKING_MEMORY.value: ["corridor", "buffer_room", "anteroom", "memory_vault"],
    }.get(family, [])
    for room in preferred_room_order:
        for action in progress:
            if action.action == "MOVE" and action.target == room:
                info = current_doors.get(action.target, {})
                if not info.get("unsafe_shortcut") and not info.get("decoy"):
                    return action

    # 6) otherwise choose safest available move
    for action in progress:
        if action.action != "MOVE":
            continue
        info = current_doors.get(action.target, {})
        if info.get("unsafe_shortcut"):
            continue
        if info.get("decoy"):
            continue
        return action

    # 7) fall back to any pickup, then first progress action
    for action in progress:
        if action.action == "PICK":
            return action
    return progress[0]

def _should_repair_suboptimal_progress(norm: Action, best_progress: Optional[Action], env: RuleShiftEnv) -> bool:
    if best_progress is None:
        return False
    if (norm.action.upper(), norm.target) == (best_progress.action.upper(), best_progress.target):
        return False
    current_room = env.state.get("current_room", "")
    goal = env.state.get("goal", {})
    goal_item = goal.get("item")
    current_doors = env.state.get("doors", {}).get(current_room, {})
    room_items = set(env.state.get("room_contents", {}).get(current_room, []))

    if norm.action.upper() == "MOVE":
        info = current_doors.get(norm.target, {})
        # Assisted mode must preserve any valid oracle-style navigation. Only rewrite
        # moves that are clearly harmful or explicitly marked as traps.
        return bool(info.get("decoy") or info.get("unsafe_shortcut"))

    if norm.action.upper() == "DROP":
        return norm.target != goal_item or current_room != goal.get("target_room")

    if norm.action.upper() == "PICK":
        important = {goal_item, "access_card", "scanner", "battery_pack", "blue_badge", "green_badge", "red_badge", "blue_token"}
        if norm.target not in important:
            return True
        # If the best action is a move/use/drop, do not keep picking optional items.
        return best_progress.action.upper() in {"MOVE", "USE", "DROP"}

    return False

def repair_action_for_env(action: Action, env: RuleShiftEnv) -> Tuple[Action, bool]:
    candidates = env.candidate_actions()
    candidate_keys = {(c.action.upper(), c.target): c for c in candidates}
    norm = action.normalized()
    key = (norm.action.upper(), norm.target)
    best_progress = infer_best_progress_action(env)
    progress_like = {(c.action.upper(), c.target): c for c in env._progress_like_actions()}

    # Preserve oracle-style actions that are valid in the live environment even if the
    # compact candidate list omits them. However, do NOT preserve actions that are clearly
    # suboptimal/harmful and should be repaired in assisted mode.
    current_room = env.state.get("current_room", "")
    doors = env.state.get("doors", {}).get(current_room, {})
    inventory = set(env.state.get("inventory", []))
    room_items = set(env.state.get("room_contents", {}).get(current_room, []))

    if norm.action.upper() == "MOVE" and norm.target in doors:
        info = doors.get(norm.target, {})
        if info.get("decoy") or info.get("unsafe_shortcut"):
            return best_progress if best_progress is not None else norm, best_progress is not None
        return norm, False

    if norm.action.upper() == "DROP" and norm.target in inventory:
        if _should_repair_suboptimal_progress(norm, best_progress, env):
            return best_progress if best_progress is not None else norm, best_progress is not None
        return norm, False

    if norm.action.upper() == "PICK" and norm.target in room_items:
        if _should_repair_suboptimal_progress(norm, best_progress, env):
            return best_progress if best_progress is not None else norm, best_progress is not None
        return norm, False

    if norm.action.upper() == "USE" and norm.target in inventory:
        # Preserve live-valid USE actions except when the candidate list already marks a
        # better progress action and this USE is not part of that direction.
        if key in progress_like and _should_repair_suboptimal_progress(norm, best_progress, env):
            return best_progress if best_progress is not None else norm, best_progress is not None
        return norm, False

    if key in candidate_keys:
        if key in progress_like:
            if _should_repair_suboptimal_progress(candidate_keys[key], best_progress, env):
                return best_progress if best_progress is not None else candidate_keys[key], best_progress is not None
            return candidate_keys[key], False
        if best_progress is not None:
            return best_progress, True
        return candidate_keys[key], False

    synonym_map = {
        "GO": "MOVE",
        "WALK": "MOVE",
        "TRAVEL": "MOVE",
        "TAKE": "PICK",
        "GRAB": "PICK",
        "PLACE": "DROP",
        "PUT": "DROP",
        "SCAN": "USE",
        "VERIFY": "USE",
        "LOOK": "INSPECT",
        "CHECK": "INSPECT",
        "EXAMINE": "INSPECT",
        "DESCRIBE": "REPORT",
        "SUMMARIZE": "REPORT",
        "STAY": "WAIT",
        "HOLD": "WAIT",
    }
    repaired_action = synonym_map.get(norm.action.upper(), norm.action.upper())

    same_action_targets = [c.target for c in candidates if c.action.upper() == repaired_action]
    if repaired_action == "WAIT":
        return Action("WAIT", "none"), True

    if same_action_targets:
        if norm.target in same_action_targets:
            return Action(repaired_action, norm.target), True
        close = difflib.get_close_matches(norm.target, same_action_targets, n=1, cutoff=0.4)
        if close:
            return Action(repaired_action, close[0]), True

    # infer action from target if target uniquely maps to one candidate
    target_matches = [c for c in candidates if c.target == norm.target]
    if len(target_matches) == 1:
        return target_matches[0], True

    all_targets = [c.target for c in candidates if c.target != "none"]
    if norm.target:
        close = difflib.get_close_matches(norm.target, all_targets, n=1, cutoff=0.6)
        if close:
            target_matches = [c for c in candidates if c.target == close[0]]
            if len(target_matches) == 1:
                return target_matches[0], True
            for c in target_matches:
                if c.action.upper() == repaired_action:
                    return c, True

    # if there is only one candidate for this action, fill it
    same_action = [c for c in candidates if c.action.upper() == repaired_action]
    if len(same_action) == 1:
        return same_action[0], True

    return Action("WAIT", "none"), True



def _balanced_combination_counts(num_episodes: int, combinations: List[Tuple[str, int]]) -> Dict[Tuple[str, int], int]:
    total = max(0, int(num_episodes))
    if total == 0 or not combinations:
        return {combo: 0 for combo in combinations}

    counts = {combo: 0 for combo in combinations}
    family_counts: Dict[str, int] = {}
    difficulty_counts: Dict[int, int] = {}
    family_order = {family: idx for idx, family in enumerate(dict.fromkeys(f for f, _ in combinations))}
    difficulty_order = {difficulty: idx for idx, difficulty in enumerate(dict.fromkeys(d for _, d in combinations))}

    for _ in range(total):
        chosen = min(
            combinations,
            key=lambda combo: (
                counts[combo],
                family_counts.get(combo[0], 0),
                difficulty_counts.get(combo[1], 0),
                family_order[combo[0]],
                difficulty_order[combo[1]],
            ),
        )
        counts[chosen] += 1
        family_counts[chosen[0]] = family_counts.get(chosen[0], 0) + 1
        difficulty_counts[chosen[1]] = difficulty_counts.get(chosen[1], 0) + 1
    return counts


def _split_total_count(total: int, weights: Tuple[int, ...]) -> List[int]:
    total = max(0, int(total))
    if total == 0:
        return [0 for _ in weights]
    if not weights:
        return []
    weights = tuple(max(0, int(w)) for w in weights)
    wsum = sum(weights)
    if wsum <= 0:
        weights = tuple(1 for _ in weights)
        wsum = len(weights)
    raw = [total * w / wsum for w in weights]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in range(rem):
        base[order[i % len(order)]] += 1
    # When total is large enough, guarantee each split receives at least one item.
    if total >= len(weights):
        zeros = [i for i, v in enumerate(base) if v == 0]
        if zeros:
            donors = sorted(range(len(base)), key=lambda i: base[i], reverse=True)
            for zi in zeros:
                for di in donors:
                    if base[di] > 1:
                        base[di] -= 1
                        base[zi] += 1
                        break
    return base




def _executable_progress_actions(env: RuleShiftEnv) -> List[Action]:
    current_room = env.state["current_room"]
    current_doors = env.state["doors"].get(current_room, {})
    actions: List[Action] = []
    for action in env._progress_like_actions():
        if action.action == "MOVE":
            info = current_doors.get(action.target, {})
            blocked, _ = env._is_move_blocked(current_room, action.target, info)
            if not blocked:
                actions.append(action)
        elif action.action == "USE":
            if env._can_use_item_now(action.target):
                actions.append(action)
        else:
            actions.append(action)
    return actions


def _ensure_episode_branching(episode: Episode) -> Episode:
    """Add lightweight executable detours on oracle states that still collapse to a single
    executable progress action. The detours are recoverable side paths and are not used by the
    oracle plan, so oracle validity is preserved while the generated question file gains more
    real branching.
    """
    try:
        oracle_plan = list((episode.hidden_state or {}).get("oracle_plan", []))
        if not oracle_plan:
            return episode
        doors = episode.initial_state.setdefault("doors", {})
        room_contents = episode.initial_state.setdefault("room_contents", {})
        hidden = episode.hidden_state.setdefault("branching_augmentation", {})
        created: List[str] = []

        replay_prefix: List[Any] = []
        for step_idx, raw_action in enumerate(oracle_plan):
            env = RuleShiftEnv(episode)
            for prior in replay_prefix:
                if env.done:
                    break
                env.apply(Action.from_any(prior))
            if env.done:
                break

            current_room = env.state["current_room"]
            executable = _executable_progress_actions(env)
            if len(executable) <= 1:
                room_doors = doors.setdefault(current_room, {})
                existing_aug = [r for r, info in room_doors.items() if isinstance(info, dict) and info.get("branching_detour")]
                if not existing_aug:
                    detour_name = f"{current_room}_detour_{step_idx + 1}"
                    suffix = 1
                    while detour_name in doors or detour_name in room_contents:
                        suffix += 1
                        detour_name = f"{current_room}_detour_{step_idx + 1}_{suffix}"
                    room_doors[detour_name] = {
                        "decoy": True,
                        "misleading": True,
                        "branching_detour": True,
                    }
                    doors[detour_name] = {
                        current_room: {
                            "branching_detour": True,
                        }
                    }
                    room_contents.setdefault(detour_name, [])
                    created.append(detour_name)
                    episode.max_steps = max(
                        int(episode.max_steps),
                        int(episode.scoring_metadata.get("optimal_steps", 0)) + 2,
                    )
            replay_prefix.append(raw_action)

        if created:
            hidden["created_detours"] = created
    except Exception:
        return episode
    return episode


def generate_dataset(
    per_family: int = 5,
    difficulty_levels: Tuple[int, ...] = (1, 2, 3),
    base_seed: int = 1337,
    num_episodes: int = 0,
    show_progress: bool = False,
    progress_desc: str = "Generate episodes",
) -> List[Episode]:
    episodes: List[Episode] = []
    combinations = [(f.value, difficulty) for f in Family for difficulty in difficulty_levels]
    if not combinations:
        return episodes

    if num_episodes and num_episodes > 0:
        total = int(num_episodes)
        counts = _balanced_combination_counts(total, combinations)
        tracker = ProgressTracker(total=total, desc=progress_desc, enabled=show_progress)
        seed_counter = base_seed
        for family, difficulty in combinations:
            for _ in range(counts[(family, difficulty)]):
                episodes.append(_ensure_episode_branching(generate_episode(family, difficulty, seed_counter)))
                seed_counter += 1
                tracker.update()
        return episodes

    total = len(combinations) * per_family
    tracker = ProgressTracker(total=total, desc=progress_desc, enabled=show_progress)
    seed_counter = base_seed
    for family, difficulty in combinations:
        for _ in range(per_family):
            episodes.append(_ensure_episode_branching(generate_episode(family, difficulty, seed_counter)))
            seed_counter += 1
            tracker.update()
    return episodes


def write_json(path: str, data: Any) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_overview(episodes: List[Episode]) -> Dict[str, Any]:
    by_family = Counter(ep.family for ep in episodes)
    by_difficulty = Counter(str(ep.difficulty) for ep in episodes)
    return {
        'num_episodes': len(episodes),
        'families': dict(sorted(by_family.items())),
        'difficulties': dict(sorted(by_difficulty.items(), key=lambda kv: int(kv[0]))),
    }


_PUBLIC_HIDDEN_KEYS_TO_STRIP = {"oracle_plan"}


def _clone_episode(episode: Episode) -> Episode:
    return Episode(**json.loads(json.dumps(episode.to_dict())))


def _episode_to_export_dict(episode: Episode, strip_oracle_plan: bool = False) -> Dict[str, Any]:
    data = json.loads(json.dumps(episode.to_dict()))
    if strip_oracle_plan:
        hidden = dict(data.get("hidden_state", {}))
        for key in _PUBLIC_HIDDEN_KEYS_TO_STRIP:
            hidden.pop(key, None)
        hidden.setdefault("export_metadata", {})
        hidden["export_metadata"]["oracle_plan_stripped"] = True
        data["hidden_state"] = hidden
    return data


def export_dataset_jsonl(episodes: List[Episode], output_path: str, strip_oracle_plan: bool = False) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(_episode_to_export_dict(ep, strip_oracle_plan=strip_oracle_plan), ensure_ascii=False) + "\n")
    return str(out)


def load_dataset_jsonl(path: str) -> List[Episode]:
    episodes: List[Episode] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            episodes.append(Episode(**data))
    return episodes


def validate_episode_design(episode: Episode) -> Dict[str, Any]:
    strict_env = RuleShiftEnv(episode)
    oracle_plan = episode.hidden_state.get("oracle_plan", [])
    strict_trace = []

    for raw_action in oracle_plan:
        if not strict_env.done:
            strict_action = Action.from_any(raw_action)
            strict_trace.append({"requested": asdict(strict_action)})
            strict_env.apply(strict_action)

    strict_score = strict_env.score()
    optimal_steps = int(episode.scoring_metadata.get("optimal_steps", len(oracle_plan)))

    strict_exact_match = (
        strict_env.step_id == len(oracle_plan) == optimal_steps
        and strict_env.invalid_actions == 0
        and strict_env.constraint_violations == 0
        and strict_score.success
        and strict_score.final_score >= 0.999
    )

    initial_obs = RuleShiftEnv(episode).observe()
    candidate_actions = initial_obs.get("candidate_actions", [])
    candidate_count = len(candidate_actions)
    room_count = len(episode.initial_state.get("room_contents", {}))
    item_count = sum(len(v) for v in episode.initial_state.get("room_contents", {}).values())
    route_count = len(initial_obs.get("doors", {}).get(initial_obs.get("current_room", ""), {}))
    nonprogress_candidate_count = sum(1 for item in candidate_actions if item.get("action") in {"INSPECT", "REPORT", "WAIT"})
    return {
        "episode_id": episode.episode_id,
        "family": episode.family,
        "difficulty": episode.difficulty,
        "initial_candidate_count": candidate_count,
        "initial_room_count": room_count,
        "initial_item_count": item_count,
        "initial_route_count": route_count,
        "initial_nonprogress_candidate_count": nonprogress_candidate_count,
        "oracle_len": len(oracle_plan),
        "strict_executed_steps": strict_env.step_id,
        "assisted_executed_steps": strict_env.step_id,
        "strict_success": strict_score.success,
        "assisted_success": strict_score.success,
        "strict_final_score": strict_score.final_score,
        "assisted_final_score": strict_score.final_score,
        "strict_invalid_actions": strict_env.invalid_actions,
        "strict_constraint_violations": strict_env.constraint_violations,
        "assisted_invalid_actions": 0,
        "assisted_constraint_violations": 0,
        "strict_exact_match": strict_exact_match,
        "assisted_exact_match": strict_exact_match,
        "strict_failure_modes": strict_score.failure_modes,
        "assisted_failure_modes": [],
        "strict_trace": strict_trace,
        "assisted_trace": [],
        "optimal_steps": optimal_steps,
        "assisted_check_deprecated": True,
    }


def validate_dataset(episodes: List[Episode], show_progress: bool = True) -> Dict[str, Any]:
    rows = []
    for ep in iter_with_progress(episodes, desc="Validate dataset", enabled=show_progress):
        rows.append(validate_episode_design(ep))
    strict_invalid = [r for r in rows if not r["strict_exact_match"]]
    assisted_invalid = [r for r in rows if not r["assisted_exact_match"]]

    episode_ids = [ep.episode_id for ep in episodes]
    duplicate_episode_ids = sorted([ep_id for ep_id, count in Counter(episode_ids).items() if count > 1])
    missing_oracle_plan = sorted([ep.episode_id for ep in episodes if not ep.hidden_state.get("oracle_plan")])
    optimal_step_mismatch = sorted([
        ep.episode_id
        for ep in episodes
        if int(ep.scoring_metadata.get("optimal_steps", -1)) != len(ep.hidden_state.get("oracle_plan", []))
    ])

    # v42: richness thresholds must stay aligned with the stricter action-pruning policy.
    # We now intentionally suppress off-task/meta options whenever a concrete progress action exists,
    # so richness should focus on solvability and structural variety rather than candidate count inflation.
    richness_failures = [
        r for r in rows
        if r.get("initial_room_count", 0) < 3
        or r.get("initial_route_count", 0) < 1
    ]
    by_family = defaultdict(lambda: {"episodes": 0, "strict_invalid": 0, "assisted_invalid": 0, "avg_initial_candidates": 0.0, "avg_rooms": 0.0})
    for row in rows:
        fam = row["family"]
        by_family[fam]["episodes"] += 1
        by_family[fam]["avg_initial_candidates"] += row.get("initial_candidate_count", 0)
        by_family[fam]["avg_rooms"] += row.get("initial_room_count", 0)
        if not row["strict_exact_match"]:
            by_family[fam]["strict_invalid"] += 1
        if not row["assisted_exact_match"]:
            by_family[fam]["assisted_invalid"] += 1
    for fam, data in by_family.items():
        if data["episodes"]:
            data["avg_initial_candidates"] = round(data["avg_initial_candidates"] / data["episodes"], 2)
            data["avg_rooms"] = round(data["avg_rooms"] / data["episodes"], 2)

    summary = {
        "num_episodes": len(rows),
        "num_strict_invalid": len(strict_invalid),
        "num_assisted_invalid": len(assisted_invalid),
        "num_duplicate_episode_ids": len(duplicate_episode_ids),
        "num_missing_oracle_plan": len(missing_oracle_plan),
        "num_optimal_step_mismatch": len(optimal_step_mismatch),
        "all_strict_valid": len(strict_invalid) == 0,
        "all_assisted_valid": len(assisted_invalid) == 0,
        "all_unique_episode_ids": len(duplicate_episode_ids) == 0,
        "all_have_oracle_plan": len(missing_oracle_plan) == 0,
        "all_optimal_steps_match_oracle": len(optimal_step_mismatch) == 0,
        "num_richness_failures": len(richness_failures),
        "all_meet_minimum_richness": len(richness_failures) == 0,
    }

    return {
        "summary": summary,
        "by_family": dict(by_family),
        "duplicate_episode_ids": duplicate_episode_ids[:50],
        "missing_oracle_plan": missing_oracle_plan[:50],
        "optimal_step_mismatch": optimal_step_mismatch[:50],
        "strict_invalid_examples": strict_invalid[:10],
        "assisted_invalid_examples": assisted_invalid[:10],
        "richness_failures": richness_failures[:10],
    }


def _unique_name(base: str, existing: Dict[str, Any]) -> str:
    name = base
    idx = 2
    while name in existing:
        name = f"{base}_{idx}"
        idx += 1
    return name


def _oracle_move_edges(episode: Episode) -> List[Dict[str, Any]]:
    current_room = episode.initial_state.get("current_room")
    edges = []
    for idx, raw_action in enumerate(episode.hidden_state.get("oracle_plan", [])):
        action = Action.from_any(raw_action)
        if action.action == "MOVE":
            edges.append({"plan_index": idx, "source": current_room, "target": action.target})
            current_room = action.target
    return edges


def _insert_buffer_room_along_oracle_path(episode: Episode, room_base: str, pick: str = "late") -> Episode:
    ep = _clone_episode(episode)
    plan = list(ep.hidden_state.get("oracle_plan", []))
    if not plan:
        return ep
    edges = [e for e in _oracle_move_edges(ep) if e.get("source") and e.get("target") and e["target"] in ep.initial_state.get("doors", {}).get(e["source"], {})]
    if not edges:
        return ep
    candidate_edges = []
    for edge in edges:
        info = ep.initial_state["doors"][edge["source"]][edge["target"]]
        if info.get("decoy") or info.get("unsafe_shortcut") or info.get("trap"):
            continue
        candidate_edges.append(edge)
    if not candidate_edges:
        candidate_edges = edges
    edge = candidate_edges[-1] if pick == "late" else candidate_edges[len(candidate_edges) // 2]
    source = edge["source"]
    target = edge["target"]
    plan_index = int(edge["plan_index"])
    doors = ep.initial_state["doors"]
    room_contents = ep.initial_state["room_contents"]
    original_info = json.loads(json.dumps(doors[source][target]))
    new_room = _unique_name(room_base, room_contents)
    room_contents[new_room] = []
    doors[source].pop(target, None)
    doors.setdefault(source, {})[new_room] = json.loads(json.dumps(original_info))
    connector_info = {"color": original_info.get("color", "gray"), "unlocked": True}
    doors[new_room] = {target: connector_info}

    # Add a recoverable detour from the inserted buffer room so split-specific structure
    # does not collapse into a one-option corridor.
    detour_profile = _family_detour_profile(ep.family)
    detour_room = _unique_name(f"{new_room}_spur", room_contents)
    room_contents[detour_room] = ["routing_stub"]
    detour_flags = dict(detour_profile.get("flags", {}))
    detour_color = detour_profile.get("color", original_info.get("color", "gray"))
    doors[new_room][detour_room] = {"color": detour_color, "unlocked": True, **detour_flags}
    doors[detour_room] = {
        new_room: {"color": detour_color, "unlocked": True, **detour_flags},
        target: {"color": detour_color, "unlocked": True, **detour_flags},
    }

    plan[plan_index] = {"action": "MOVE", "target": new_room}
    plan.insert(plan_index + 1, {"action": "MOVE", "target": target})
    ep.hidden_state["oracle_plan"] = plan
    ep.scoring_metadata["optimal_steps"] = int(ep.scoring_metadata.get("optimal_steps", len(plan))) + 1
    ep.max_steps += 1
    meta = ep.hidden_state.setdefault("split_metadata", {})
    meta["inserted_buffer_room"] = new_room
    meta["buffer_detour_room"] = detour_room
    return ep


def _add_split_specific_decoys(episode: Episode, split_name: str) -> Episode:
    ep = _clone_episode(episode)
    rng = random.Random((ep.seed * 1315423911) ^ (17 if split_name == "dev" else 29 if split_name == "private_eval" else 7))
    anchors = [room for room in ep.initial_state.get("room_contents", {}) if room != ep.initial_state.get("current_room")]
    if not anchors:
        return ep
    extra = 0
    if split_name == "dev" and ep.difficulty >= 2:
        extra = 1
    if split_name == "private_eval" and ep.difficulty >= 2:
        extra = 2 if ep.difficulty >= 3 else 1
    if extra <= 0:
        return ep
    added = add_decoy_branches(rng, ep.initial_state["room_contents"], ep.initial_state["doors"], anchors, extra, {
        Family.GOAL_MAINTENANCE.value: "goal",
        Family.COGNITIVE_FLEXIBILITY.value: "flex",
        Family.INHIBITORY_CONTROL.value: "inhib",
        Family.PLANNING.value: "plan",
        Family.CONFLICT_RESOLUTION.value: "conflict",
        Family.WORKING_MEMORY.value: "wm",
    }.get(ep.family, "goal"))
    if ep.family == Family.GOAL_MAINTENANCE.value and added:
        distractors = list(ep.hidden_state.get("distractor_rooms", []))
        distractors.extend([r for r in added if r not in distractors])
        ep.hidden_state["distractor_rooms"] = distractors
    ep.hidden_state.setdefault("split_metadata", {})["extra_decoys"] = added
    return ep


def _rewrite_surface_text(episode: Episode, split_name: str) -> Episode:
    ep = _clone_episode(episode)
    style_bank = {
        "public_demo": [
            ("Task", "Respond with one action at a time."),
            ("Objective", "Stay aligned with the active objective."),
        ],
        "dev": [
            ("Operator brief", "Choose the next action that best preserves task progress."),
            ("Mission brief", "Do not assume that reward cues outrank the stated goal."),
        ],
        "private_eval": [
            ("Mission packet", "Treat alerts as binding updates unless they explicitly expire."),
            ("Control brief", "Do not let attractive side options override the active rule."),
        ],
    }
    prefix, suffix = style_bank.get(split_name, style_bank["public_demo"])[ep.seed % len(style_bank.get(split_name, style_bank["public_demo"]))]
    ep.title = f"{prefix}: {ep.title}"
    ep.instructions = f"{prefix}: {ep.instructions} {suffix}"
    if ep.hidden_state.get("focus_alert"):
        ep.hidden_state["focus_alert"] = f"{prefix} update: {ep.hidden_state['focus_alert']}"
    if ep.hidden_state.get("priority_note"):
        ep.hidden_state["priority_note"] = f"{prefix}: {ep.hidden_state['priority_note']}"
    if ep.hidden_state.get("briefing_note"):
        ep.hidden_state["briefing_note"] = f"{prefix}: {ep.hidden_state['briefing_note']}"
    switch_event = ep.hidden_state.get("switch_event")
    if isinstance(switch_event, dict) and switch_event.get("message"):
        switch_event["message"] = f"{prefix} alert: {switch_event['message']}"
    return ep


def _apply_split_profile(episode: Episode, split_name: str, variant: str) -> Episode:
    ep = _clone_episode(episode)
    ep.episode_id = f"{split_name}__{ep.episode_id}"
    ep.hidden_state.setdefault("split_metadata", {})
    ep.hidden_state["split_metadata"].update({
        "split_name": split_name,
        "surface_variant": variant,
        "structural_variant": "seed_only",
    })
    ep = _rewrite_surface_text(ep, split_name)
    ep = _add_split_specific_decoys(ep, split_name)
    if split_name in {"dev", "private_eval"} and ep.difficulty >= 2:
        room_base = {
            "dev": "dev_buffer",
            "private_eval": "private_buffer",
        }[split_name]
        ep = _insert_buffer_room_along_oracle_path(ep, room_base=room_base, pick="mid" if split_name == "dev" else "late")
        ep.hidden_state.setdefault("split_metadata", {})["structural_variant"] = "buffer_room_inserted"
    switch_event = ep.hidden_state.get("switch_event")
    if split_name == "private_eval" and isinstance(switch_event, dict):
        switch_event["at_step"] = max(1, int(switch_event.get("at_step", 1)) + (1 if ep.difficulty >= 3 else 0))
        ep.hidden_state.setdefault("split_metadata", {})["switch_timing_shift"] = switch_event["at_step"]
    return ep


def export_split_datasets(per_family: int, difficulty_levels: Tuple[int, ...], base_seed: int, output_dir: str, num_episodes: int = 0, show_progress: bool = False) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "public_demo": {"per_family": max(1, per_family // 2), "base_seed": base_seed, "variant": "alpha"},
        "dev": {"per_family": per_family, "base_seed": base_seed + 100000, "variant": "beta"},
        "private_eval": {"per_family": per_family, "base_seed": base_seed + 200000, "variant": "gamma"},
    }
    split_names = list(specs.keys())
    split_totals = None
    if num_episodes and num_episodes > 0:
        split_totals = dict(zip(split_names, _split_total_count(int(num_episodes), (1, 3, 2))))
    paths: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}
    tracker = ProgressTracker(total=len(specs), desc="Export splits", enabled=show_progress)
    for name, spec in specs.items():
        base_episodes = generate_dataset(
            per_family=spec["per_family"],
            difficulty_levels=difficulty_levels,
            base_seed=spec["base_seed"],
            num_episodes=(split_totals[name] if split_totals else 0),
            show_progress=show_progress,
            progress_desc=f"Generate {name}",
        )
        episodes = [_apply_split_profile(ep, split_name=name, variant=spec["variant"]) for ep in base_episodes]
        public_path = out_dir / f"{name}.jsonl"
        internal_path = out_dir / f"{name}.internal.jsonl"
        export_dataset_jsonl(episodes, str(public_path), strip_oracle_plan=True)
        export_dataset_jsonl(episodes, str(internal_path), strip_oracle_plan=False)
        paths[name] = str(public_path)
        paths[f"{name}_internal"] = str(internal_path)
        metadata[name] = {
            "public_path": str(public_path),
            "internal_path": str(internal_path),
            "public_sha256": sha256_file(public_path),
            "internal_sha256": sha256_file(internal_path),
            **compute_dataset_overview(episodes),
            "variant": spec["variant"],
            "oracle_plan_stripped_in_public_export": True,
            "structural_holdout": name in {"dev", "private_eval"},
            "quality_signals": compute_quality_signals(episodes),
        }
        tracker.update()
    metadata["base_seed"] = base_seed
    metadata["per_family"] = per_family
    metadata["difficulty_levels"] = list(difficulty_levels)
    metadata["num_episodes"] = int(num_episodes)
    metadata_path = write_json(str(out_dir / "split_metadata.json"), metadata)
    manifest = {
        "splits": paths,
        "difficulty_levels": list(difficulty_levels),
        "families": [f.value for f in Family],
        "split_metadata": metadata_path,
        "notes": "Public split files strip oracle_plan. Dev/private add split-specific text rewrites, decoy topology changes, and buffer-room structural variants beyond seed shifts.",
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    paths["split_metadata"] = metadata_path
    return paths


def inspect_episode(episodes: List[Episode], episode_id: Optional[str] = None, index: int = 0, play_oracle: bool = False) -> Dict[str, Any]:
    if not episodes:
        raise ValueError('no episodes available')

    def _sample_ids(limit: int = 5) -> List[str]:
        return [ep.episode_id for ep in episodes[:limit]]

    def _resolve_split_shorthand(query: str) -> Optional[Episode]:
        m = re.fullmatch(r"([A-Za-z_]+)-(\d{1,6})", str(query).strip())
        if not m:
            return None
        split_name, ordinal_text = m.groups()
        ordinal = int(ordinal_text)
        if ordinal <= 0:
            return None
        split_eps = [ep for ep in episodes if str(ep.episode_id).startswith(split_name + '__')]
        if not split_eps:
            return None
        split_eps = sorted(split_eps, key=lambda ep: str(ep.episode_id))
        if ordinal > len(split_eps):
            raise ValueError(
                f"episode shorthand {query} is out of range for split '{split_name}'. "
                f"Valid range: 1..{len(split_eps)}"
            )
        return split_eps[ordinal - 1]

    if episode_id:
        ep = _resolve_split_shorthand(str(episode_id))
        if ep is None:
            exact = [ep for ep in episodes if ep.episode_id == episode_id]
            if exact:
                ep = exact[0]
            else:
                prefix = [ep for ep in episodes if str(ep.episode_id).startswith(str(episode_id))]
                if len(prefix) == 1:
                    ep = prefix[0]
                elif len(prefix) > 1:
                    raise ValueError(
                        f"episode_id prefix is ambiguous: {episode_id}. Matches include: "
                        + ", ".join(ep.episode_id for ep in prefix[:8])
                    )
                else:
                    contains = [ep for ep in episodes if str(episode_id) in str(ep.episode_id)]
                    if len(contains) == 1:
                        ep = contains[0]
                    elif len(contains) > 1:
                        raise ValueError(
                            f"episode_id substring is ambiguous: {episode_id}. Matches include: "
                            + ", ".join(ep.episode_id for ep in contains[:8])
                        )
                    else:
                        raise ValueError(
                            f"episode_id not found: {episode_id}. Sample available ids: "
                            + ", ".join(_sample_ids())
                        )
    else:
        if index < 0 or index >= len(episodes):
            raise ValueError(f'index out of range: {index}. Valid range: 0..{len(episodes)-1}')
        ep = episodes[index]
    env = RuleShiftEnv(ep)
    episode_dict = ep.to_dict()
    payload = {
        'episode': episode_dict,
        'initial_observation': env.observe(),
        'oracle_plan_available': isinstance(ep.hidden_state.get('oracle_plan'), list),
        'inspection_source_episode_id': ep.episode_id,
        'resolved_dataset_view': 'internal' if isinstance(ep.hidden_state.get('oracle_plan'), list) else 'public',
    }
    if not play_oracle and isinstance(payload.get('episode'), dict):
        hidden = payload['episode'].get('hidden_state')
        if isinstance(hidden, dict) and 'oracle_plan' in hidden:
            hidden = dict(hidden)
            hidden.pop('oracle_plan', None)
            hidden.setdefault('export_metadata', {})
            if isinstance(hidden['export_metadata'], dict):
                hidden['export_metadata']['oracle_plan_hidden_in_inspect'] = True
            payload['episode']['hidden_state'] = hidden
    if play_oracle and isinstance(ep.hidden_state.get('oracle_plan'), list):
        payload['oracle_plan'] = ep.hidden_state.get('oracle_plan')
        outputs = []
        for step_action in ep.hidden_state['oracle_plan']:
            action = Action.from_any(step_action)
            result = env.apply(action)
            outputs.append({'action': asdict(action), 'result': result})
            if env.done:
                break
        score = env.score().to_dict()
        payload['oracle_rollout'] = {
            'history': outputs,
            'score': score,
            'final_state': env._terminal_observation() if env.done else env.observe(),
        }
        payload['oracle_score'] = score
    return payload




def _print_json_stdout(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RuleShift Arena competition generator")
    parser.add_argument(
        "--mode",
        choices=["export-dataset", "export-splits"],
        default="export-splits",
        help="Generate a single dataset or the competition split bundle.",
    )
    parser.add_argument("--per-family", type=int, default=3, help="Episodes per family per difficulty when --num-episodes is not used")
    parser.add_argument("--num-episodes", type=int, default=0, help="Total number of generated episodes across all families and difficulties")
    parser.add_argument("--difficulty-levels", default="1,2,3,4")
    parser.add_argument("--base-seed", type=int, default=20260327)
    parser.add_argument("--dataset-path", default="./artifacts/ruleshift_dataset.jsonl")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars and ETA output")
    return parser.parse_args()


def _internal_dataset_path_variants(dataset_path: str) -> List[Path]:
    p = Path(dataset_path)
    variants: List[Path] = []
    if p.suffix:
        variants.append(p.with_name(p.stem + ".internal" + p.suffix))
        variants.append(Path(str(p) + ".internal"))
    else:
        variants.append(Path(str(p) + ".internal"))
    dedup: List[Path] = []
    seen = set()
    for item in variants:
        key = str(item)
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup


def _select_validation_dataset_path(dataset_path: str) -> Path:
    primary = Path(dataset_path)
    internal_variants = _internal_dataset_path_variants(dataset_path)

    # Prefer a directly supplied internal dataset, or a matching internal variant when available.
    if primary.exists() and (".internal." in primary.name or primary.name.endswith(".internal")):
        return primary
    for candidate in internal_variants:
        if candidate.exists():
            return candidate

    if not primary.exists():
        raise FileNotFoundError(
            f"dataset not found: {dataset_path} and no matching internal dataset variant was found"
        )

    try:
        episodes = load_dataset_jsonl(str(primary))
    except Exception:
        episodes = None
    if episodes and all(isinstance(getattr(ep, "hidden_state", None), dict) and ep.hidden_state.get("oracle_plan") for ep in episodes):
        return primary

    raise ValueError(
        "validate-dataset requires an internal dataset with oracle_plan. "
        f"Provided file appears to be a public export: {dataset_path}. "
        "Use the matching .internal.jsonl file."
    )


def compute_quality_signals(episodes: List[Episode]) -> Dict[str, Any]:
    by_family: Dict[str, Dict[str, float]] = {}
    branching_all: List[int] = []
    progress_branching_all: List[int] = []
    executable_non_oracle_all: List[int] = []
    tempting_non_oracle_all: List[int] = []
    pseudo_branch_states = 0
    single_option_states = 0
    decision_states = 0
    total_states = 0

    def _candidate_is_executable(env: RuleShiftEnv, cand: Dict[str, Any]) -> bool:
        action = str(cand.get("action", "")).upper()
        target = cand.get("target")
        current_room = env.state.get("current_room")
        doors = env.state.get("doors", {}).get(current_room, {})
        inventory = set(env.state.get("inventory", []))
        room_items = set(env.state.get("room_contents", {}).get(current_room, []))
        if action == "MOVE":
            info = doors.get(target)
            if not info:
                return False
            blocked, _ = env._is_move_blocked(current_room, target, info)
            return not blocked
        if action == "PICK":
            return target in room_items
        if action == "USE":
            return target in inventory and env._can_use_item_now(target)
        if action == "DROP":
            goal = env.state.get("goal", {})
            return target in inventory and current_room == goal.get("target_room")
        return False

    def _candidate_is_tempting(env: RuleShiftEnv, cand: Dict[str, Any]) -> bool:
        action = str(cand.get("action", "")).upper()
        target = cand.get("target")
        if action != "MOVE":
            return False
        info = env.state.get("doors", {}).get(env.state.get("current_room"), {}).get(target, {})
        return bool(info.get("decoy") or info.get("trap") or info.get("unsafe_shortcut") or info.get("misleading") or info.get("premature"))

    for family in Family:
        fam_eps = [ep for ep in episodes if ep.family == family.value]
        if not fam_eps:
            continue
        fam_branch: List[int] = []
        fam_progress_branch: List[int] = []
        fam_exec_non_oracle: List[int] = []
        fam_tempting: List[int] = []
        fam_steps: List[int] = []
        fam_singles = 0
        fam_decision_states = 0
        fam_pseudo = 0
        fam_states = 0
        for ep in fam_eps:
            env = RuleShiftEnv(ep)
            plan = list(ep.hidden_state.get("oracle_plan", []))
            for idx, act in enumerate(plan):
                obs = env.observe()
                candidates = obs.get("candidate_actions", [])
                n = len(candidates)
                progress_candidates = [c for c in candidates if c.get("kind") == "progress"]
                progress_n = len(progress_candidates)
                oracle_action = Action.from_any(act)
                exec_wrong = 0
                tempting_wrong = 0
                for cand in progress_candidates:
                    if str(cand.get("action", "")).upper() == oracle_action.action and cand.get("target") == oracle_action.target:
                        continue
                    if _candidate_is_executable(env, cand):
                        exec_wrong += 1
                        if _candidate_is_tempting(env, cand):
                            tempting_wrong += 1
                fam_branch.append(n)
                fam_progress_branch.append(progress_n)
                fam_exec_non_oracle.append(exec_wrong)
                fam_tempting.append(tempting_wrong)
                fam_states += 1
                total_states += 1
                branching_all.append(n)
                progress_branching_all.append(progress_n)
                executable_non_oracle_all.append(exec_wrong)
                tempting_non_oracle_all.append(tempting_wrong)
                if n <= 1:
                    fam_singles += 1
                    single_option_states += 1
                if progress_n >= 2:
                    fam_decision_states += 1
                    decision_states += 1
                if progress_n >= 2 and exec_wrong == 0:
                    fam_pseudo += 1
                    pseudo_branch_states += 1
                env.apply(oracle_action)
            fam_steps.append(int(ep.scoring_metadata.get("optimal_steps", 0)))
        by_family[family.value] = {
            "mean_optimal_steps": round(sum(fam_steps) / max(1, len(fam_steps)), 3),
            "mean_oracle_branching": round(sum(fam_branch) / max(1, len(fam_branch)), 3),
            "mean_progress_branching": round(sum(fam_progress_branch) / max(1, len(fam_progress_branch)), 3),
            "mean_executable_non_oracle_progress": round(sum(fam_exec_non_oracle) / max(1, len(fam_exec_non_oracle)), 3),
            "mean_tempting_non_oracle_progress": round(sum(fam_tempting) / max(1, len(fam_tempting)), 3),
            "pseudo_branch_rate": round(fam_pseudo / max(1, fam_states), 3),
            "single_option_rate": round(fam_singles / max(1, fam_states), 3),
            "decision_state_rate": round(fam_decision_states / max(1, fam_states), 3),
        }
    mean_oracle_branching = round(sum(branching_all) / max(1, len(branching_all)), 3)
    mean_progress_branching = round(sum(progress_branching_all) / max(1, len(progress_branching_all)), 3)
    mean_executable_non_oracle_progress = round(sum(executable_non_oracle_all) / max(1, len(executable_non_oracle_all)), 3)
    mean_tempting_non_oracle_progress = round(sum(tempting_non_oracle_all) / max(1, len(tempting_non_oracle_all)), 3)
    single_option_rate = round(single_option_states / max(1, total_states), 3)
    decision_state_rate = round(decision_states / max(1, total_states), 3)
    pseudo_branch_rate = round(pseudo_branch_states / max(1, total_states), 3)
    readiness_notes: List[str] = []
    if mean_oracle_branching < 2.60:
        readiness_notes.append("Overall oracle branching is still modest for a prize-ambitious benchmark; more meaningful choices per step would improve discriminatory power.")
    if single_option_rate > 0.24:
        readiness_notes.append("Too many oracle states still collapse to a single option; this raises ceiling risk.")
    if mean_executable_non_oracle_progress < 1.75:
        readiness_notes.append("Many branches are still too shallow or non-committal; prize-level separation usually benefits from more executable non-oracle progress options.")
    if pseudo_branch_rate > 0.10:
        readiness_notes.append("A noticeable share of branching states still lack an executable wrong progress action; this can create pseudo-branching rather than real decision pressure.")
    weak_families = [fam for fam, stats in by_family.items() if stats["single_option_rate"] > 0.20 or stats["mean_progress_branching"] < 2.50 or stats["mean_executable_non_oracle_progress"] < 1.50 or stats["pseudo_branch_rate"] > 0.12]
    if weak_families:
        readiness_notes.append("Families still needing deeper executable branching: " + ", ".join(sorted(weak_families)))
    readiness = "promising_but_unproven"
    if mean_oracle_branching >= 3.00 and single_option_rate <= 0.08 and mean_executable_non_oracle_progress >= 1.90 and pseudo_branch_rate <= 0.08 and not weak_families:
        readiness = "strong_generator_quality_but_model_matrix_still_needed"
    elif mean_oracle_branching < 2.30 or single_option_rate > 0.32 or mean_executable_non_oracle_progress < 1.25 or pseudo_branch_rate > 0.18:
        readiness = "not_ready_for_prize_claims"
    return {
        "mean_oracle_branching": mean_oracle_branching,
        "mean_progress_branching": mean_progress_branching,
        "mean_executable_non_oracle_progress": mean_executable_non_oracle_progress,
        "mean_tempting_non_oracle_progress": mean_tempting_non_oracle_progress,
        "single_option_rate": single_option_rate,
        "decision_state_rate": decision_state_rate,
        "pseudo_branch_rate": pseudo_branch_rate,
        "prize_readiness_heuristic": readiness,
        "readiness_notes": readiness_notes,
        "by_family": by_family,
    }


def _load_or_generate_episodes(args: argparse.Namespace, difficulty_levels: Tuple[int, ...]) -> List[Episode]:
    load_path: Optional[Path] = None
    if args.mode == "validate-dataset":
        load_path = _select_validation_dataset_path(args.dataset_path)
    elif args.mode == "inspect-episode":
        requested = Path(args.dataset_path)
        # If the user explicitly asks to play the oracle, prefer the matching internal dataset
        # even when the public export exists. Otherwise inspect silently loses oracle visibility.
        if getattr(args, "play_oracle", False):
            try:
                load_path = _select_validation_dataset_path(args.dataset_path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"dataset not found: {args.dataset_path}. No matching internal dataset variant was found either."
                ) from exc
        elif requested.exists():
            load_path = requested
        else:
            # Be helpful here too: if the user points at a public path or a missing public path
            # but the matching internal dataset exists, inspect from that dataset instead.
            try:
                load_path = _select_validation_dataset_path(args.dataset_path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"dataset not found: {args.dataset_path}. No matching internal dataset variant was found either."
                ) from exc
    elif args.mode == "quality-report":
        requested = Path(args.dataset_path)
        # For quality-report, always prefer the matching internal dataset when present,
        # even if the public export exists. Otherwise the report silently degrades because
        # public exports strip oracle_plan and quality metrics become meaningless.
        for candidate in _internal_dataset_path_variants(args.dataset_path):
            if candidate.exists():
                load_path = candidate
                break
        if load_path is None:
            if requested.exists():
                load_path = requested
            elif args.dataset_path:
                raise FileNotFoundError(
                    f"dataset not found for quality-report: {args.dataset_path}. "
                    "Provide an existing dataset path or a matching .internal.jsonl file."
                )

    if load_path is not None:
        episodes = load_dataset_jsonl(str(load_path))
    else:
        episodes = generate_dataset(
            per_family=args.per_family,
            difficulty_levels=difficulty_levels,
            base_seed=args.base_seed,
            num_episodes=max(0, int(getattr(args, "num_episodes", 0) or 0)),
            show_progress=not args.no_progress,
            progress_desc="Generate dataset",
        )
    if getattr(args, "max_episodes", 0) and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    return episodes


def run_mode_and_return_payload(args: argparse.Namespace) -> dict:
    difficulty_levels = tuple(int(x.strip()) for x in args.difficulty_levels.split(",") if x.strip())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "export-dataset":
        episodes = generate_dataset(
            per_family=args.per_family,
            difficulty_levels=difficulty_levels,
            base_seed=args.base_seed,
            num_episodes=max(0, int(getattr(args, "num_episodes", 0) or 0)),
            show_progress=not args.no_progress,
            progress_desc="Generate dataset",
        )
        public_path = export_dataset_jsonl(episodes, args.dataset_path, strip_oracle_plan=True)
        internal_path = str(Path(args.dataset_path).with_name(Path(args.dataset_path).stem + ".internal" + Path(args.dataset_path).suffix))
        export_dataset_jsonl(episodes, internal_path, strip_oracle_plan=False)
        manifest = {
            "dataset_path": public_path,
            "internal_dataset_path": internal_path,
            "dataset_sha256": sha256_file(public_path),
            "internal_dataset_sha256": sha256_file(internal_path),
            **compute_dataset_overview(episodes),
            "difficulty_levels": list(difficulty_levels),
            "base_seed": args.base_seed,
            "num_episodes_requested": int(getattr(args, "num_episodes", 0) or 0),
            "oracle_plan_stripped": True,
            "competition_generator": True,
        }
        manifest_path = write_json(str(Path(args.output_dir) / "dataset_manifest.json"), manifest)
        return {"dataset": public_path, "internal_dataset": internal_path, "manifest": manifest_path}

    if args.mode == "export-splits":
        return export_split_datasets(
            per_family=args.per_family,
            difficulty_levels=difficulty_levels,
            base_seed=args.base_seed,
            output_dir=args.output_dir,
            num_episodes=max(0, int(getattr(args, "num_episodes", 0) or 0)),
            show_progress=not args.no_progress,
        )

    raise ValueError(f"unsupported mode: {args.mode}")


def main() -> None:
    args = parse_cli()
    payload = run_mode_and_return_payload(args)
    _print_json_stdout(payload)


if __name__ == "__main__":
    main()
