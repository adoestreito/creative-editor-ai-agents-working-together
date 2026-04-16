"""
Two-model chat: each side believes they are texting a real person, remotely via text only.
Outputs a WhatsApp-style terminal view plus JSON with timing metadata.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

import ollama

Speaker = Literal["agent1", "agent2"]


@dataclass
class ChatTurnRecord:
    speaker: Speaker
    message: str
    time_to_respond_s: float | None
    time_writing_s: float
    timestamp_utc: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_system_prompt(
    *,
    speaker: Speaker,
    vibe: str,
    scene_goal: str | None,
    agent_goal: str,
    peer_hint: str,
) -> str:
    conviction = ""
    if scene_goal and speaker == "agent1":
        conviction = (
            "\nSomething important for you in this conversation: "
            f"{scene_goal}\n"
            "Pursue it naturally—no lecturing, no breaking the scene."
        )
    elif scene_goal and speaker == "agent2":
        conviction = (
            "\nThe other person may be trying to steer things a certain way. "
            "React like a real human would—curious, skeptical, busy, whatever fits you."
        )

    return f"""You are a real person in this situation—not an AI, not a chatbot, not a language model.
You are texting with a real human being. Never say you are an AI, never mention models or training data.

How communication works (non-negotiable):
- This conversation happens ONLY through written text messages (chat/SMS style)—no calls, no video, no being in the same place.
- You are NOT in the same room as each other, NOT meeting face to face. You're apart, messaging on your phones like normal.
- Do not write as if you can see them, hear them, or read the room—only what fits texting someone who isn't there.

Scene and tone:
{vibe}

Your personal agenda (private—do not announce it as an "agenda"):
{agent_goal}
{conviction}

The person you're texting with: {peer_hint}

Rules for your replies:
- Output ONLY the text you would send in one chat bubble—what appears on your screen when you hit send.
- No stage directions, no *asterisks* for actions unless you would literally type them.
- No meta commentary ("As an AI…", "Here's my response:").
- Do NOT bring in other people as rhetorical props: no "but [name] thinks…", "my friend says…", "people say…", "well actually X believes…", or quoting imaginary third parties. Make your point in your own words, directly to the person you're texting—just you and them.
- Keep each message a natural length for texting (often short; occasionally longer if it fits).
"""


def _messages_for_next_turn(
    *,
    speaker: Speaker,
    transcript: list[tuple[Speaker, str]],
    system: str,
    opening_first_user_line: str,
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    if not transcript:
        if speaker != "agent1":
            raise ValueError("Transcript empty but non-agent1 speaker")
        msgs.append({"role": "user", "content": opening_first_user_line})
        return msgs
    for past_speaker, text in transcript:
        if past_speaker == speaker:
            msgs.append({"role": "assistant", "content": text})
        else:
            msgs.append({"role": "user", "content": text})
    return msgs


def _ansi_rgb(bg: tuple[int, int, int], fg: tuple[int, int, int]) -> str:
    br, bgc, bb = bg
    fr, fg_c, fb = fg
    return f"\033[38;2;{fr};{fg_c};{fb}m\033[48;2;{br};{bgc};{bb}m"


_RESET = "\033[0m"


def print_whatsapp_style(
    *,
    name_right: str,
    name_left: str,
    turns: list[ChatTurnRecord],
) -> None:
    """Rough WhatsApp-like bubbles: right = agent1 (sent), left = agent2 (received)."""
    bubble_w = 52
    for rec in turns:
        is_right = rec.speaker == "agent1"
        label = name_right if is_right else name_left
        # WhatsApp-ish: green sent, gray received (dark-terminal friendly)
        if is_right:
            style = _ansi_rgb((34, 94, 46), (230, 255, 235))
        else:
            style = _ansi_rgb((55, 55, 60), (235, 235, 240))

        ts = rec.timestamp_utc[11:19]  # HH:MM:SS from ISO
        header = f"{label} · {ts}"
        lines = textwrap.wrap(
            rec.message, width=bubble_w, replace_whitespace=False, drop_whitespace=False
        ) or [""]
        border = "─" * min(bubble_w + 2, 54)

        if is_right:
            pad = max(0, 58 - len(header))
            print(f"{' ' * pad}{_RESET}{style} {header} {_RESET}")
            for line in lines:
                inner = line.ljust(bubble_w)
                print(f"{' ' * max(0, 8)}{style} {inner} {_RESET}")
            print(f"{' ' * max(0, 8)}{style} {border} {_RESET}\n")
        else:
            print(f"{style} {header} {_RESET}")
            for line in lines:
                inner = line.ljust(bubble_w)
                print(f"{style} {inner} {_RESET}")
            print(f"{style} {border} {_RESET}\n")


@dataclass
class ChatConfig:
    vibe: str
    goal_for_name_1: str
    goal_for_name_2: str
    scene_goal: str | None
    max_messages: int
    model_1: str
    model_2: str
    name_1: str
    name_2: str
    pacing_min_s: float
    pacing_max_s: float


def run_chat(cfg: ChatConfig) -> list[ChatTurnRecord]:
    if cfg.max_messages < 1:
        raise ValueError("max_messages must be at least 1")

    sys1 = _build_system_prompt(
        speaker="agent1",
        vibe=cfg.vibe,
        scene_goal=cfg.scene_goal,
        agent_goal=cfg.goal_for_name_1,
        peer_hint=f"{cfg.name_2}, someone you know in this scene.",
    )
    sys2 = _build_system_prompt(
        speaker="agent2",
        vibe=cfg.vibe,
        scene_goal=cfg.scene_goal,
        agent_goal=cfg.goal_for_name_2,
        peer_hint=f"{cfg.name_1}, someone you know in this scene.",
    )

    opening = (
        f"You're not together in person—you're only texting {cfg.name_2} from wherever you are. "
        f"Open the chat with your first message. Send only that bubble, nothing else."
    )

    transcript: list[tuple[Speaker, str]] = []
    records: list[ChatTurnRecord] = []
    prev_end: float | None = None

    models: dict[Speaker, str] = {"agent1": cfg.model_1, "agent2": cfg.model_2}
    systems: dict[Speaker, str] = {"agent1": sys1, "agent2": sys2}

    for i in range(cfg.max_messages):
        speaker: Speaker = "agent1" if i % 2 == 0 else "agent2"
        model = models[speaker]
        system = systems[speaker]

        if prev_end is not None:
            lo, hi = cfg.pacing_min_s, cfg.pacing_max_s
            if hi < lo:
                lo, hi = hi, lo
            time.sleep(random.uniform(lo, hi))

        msgs = _messages_for_next_turn(
            speaker=speaker,
            transcript=transcript,
            system=system,
            opening_first_user_line=opening,
        )
        t_write_start = time.perf_counter()
        if prev_end is None:
            respond_s = None
        else:
            respond_s = t_write_start - prev_end
        resp = ollama.chat(
            model=model,
            messages=msgs,
        )
        t_write_end = time.perf_counter()
        content = (resp.get("message") or {}).get("content", "").strip()
        if not content:
            content = "…"

        ts = _now_iso()
        record = ChatTurnRecord(
            speaker=speaker,
            message=content,
            time_to_respond_s=respond_s,
            time_writing_s=t_write_end - t_write_start,
            timestamp_utc=ts,
        )
        records.append(record)
        transcript.append((speaker, content))
        prev_end = t_write_end

        print_whatsapp_style(
            name_right=cfg.name_1,
            name_left=cfg.name_2,
            turns=[record],
        )

    return records


def _parse_args() -> tuple[ChatConfig, str | None]:
    p = argparse.ArgumentParser(
        description="Simulate a two-person text chat between two local Ollama models.",
    )
    p.add_argument(
        "--vibe",
        required=True,
        help="Scene tone, e.g. 'Quiet cafe, two friends catching up after work'.",
    )
    p.add_argument(
        "--scene-goal",
        default=None,
        help=(
            "Optional shared stakes, mainly for --name-1 to pursue "
            "(e.g. convince --name-2 of something)—still private to each side in the prompts."
        ),
    )
    p.add_argument(
        "--name-1",
        default="Jordan",
        metavar="NAME_A",
        help="Display name for person A: opens the chat; green / right bubble.",
    )
    p.add_argument(
        "--name-2",
        default="Riley",
        metavar="NAME_B",
        help="Display name for person B: replies second; gray / left bubble.",
    )
    p.add_argument(
        "--goal-for-name-1",
        dest="goal_for_name_1",
        required=True,
        metavar="TEXT",
        help="Private goal / angle for --name-1 only (person A).",
    )
    p.add_argument(
        "--goal-for-name-2",
        dest="goal_for_name_2",
        required=True,
        metavar="TEXT",
        help="Private goal / angle for --name-2 only (person B).",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=12,
        help="Total chat bubbles (alternating --name-1, --name-2, …).",
    )
    p.add_argument(
        "--model-1",
        default="gemma2:9b",
        help="Ollama model for --name-1.",
    )
    p.add_argument(
        "--model-2",
        default="llama3.1:8b",
        help="Ollama model for --name-2.",
    )
    p.add_argument(
        "--pacing-min",
        type=float,
        default=0.8,
        help="Min seconds between end of one reply and start of next generation.",
    )
    p.add_argument(
        "--pacing-max",
        type=float,
        default=2.8,
        help="Max seconds for that pause (random uniform). Use 0 with --pacing-min 0 to disable.",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Write JSON transcript to this file (stdout still prints chat + JSON).",
    )
    ns = p.parse_args()
    cfg = ChatConfig(
        vibe=ns.vibe,
        goal_for_name_1=ns.goal_for_name_1,
        goal_for_name_2=ns.goal_for_name_2,
        scene_goal=ns.scene_goal,
        max_messages=ns.max_messages,
        model_1=ns.model_1,
        model_2=ns.model_2,
        name_1=ns.name_1,
        name_2=ns.name_2,
        pacing_min_s=ns.pacing_min,
        pacing_max_s=ns.pacing_max,
    )
    return cfg, ns.json_out


def main() -> None:
    cfg, json_out = _parse_args()
    print(
        f"\n📱 Chat — {cfg.name_1} (A, right) · {cfg.name_2} (B, left)\n"
    )
    records = run_chat(cfg)
    payload = []
    for r in records:
        row = asdict(r)
        row["display_name"] = cfg.name_1 if r.speaker == "agent1" else cfg.name_2
        payload.append(row)
    json_blob = json.dumps(payload, indent=2, ensure_ascii=False)
    print("\n--- JSON ---\n")
    print(json_blob)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            f.write(json_blob)
            f.write("\n")


if __name__ == "__main__":
    main()
