# Creative agents with Ollama

Small Python experiments where **local LLMs** (via [Ollama](https://ollama.com/)) play different roles: a writer/editor loop for fiction, and a two-person **text-only chat** where each model thinks it is messaging a real human.

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running (`ollama serve` or the desktop app)

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pull the models used by the scripts (or change model names in the code / CLI):

```bash
ollama pull gemma2:9b
ollama pull llama3.1:8b
```

---

## 1. Creative engine (`creative_engine.py`)

A **writer** model drafts a piece; an **editor** model scores it and sends revision notes. The writer revises in a loop until the editor approves or you hit the iteration limit. The terminal shows each draft, full editor feedback, and each revision.

**Run:**

```bash
python creative_engine.py
```

**Customize** by editing the block at the bottom of the file (`genre`, `target_length`, `topic`) or import `agentic_creative_engine` from another script:

```python
from creative_engine import agentic_creative_engine

text = agentic_creative_engine(
    genre="Sci-Fi Horror",
    target_length=200,
    topic="A toaster that predicts the exact time of its owner's death",
)
print(text)
```

Default models in code: writer `gemma2:9b`, editor `llama3.1:8b`.

---

## 2. Two-model chat (`chat.py`)

Two models alternate as **person A** (`--name-1`, green / right) and **person B** (`--name-2`, gray / left). The scene is **remote text only** (no same room, no calls). Each side has its own private goal; optional `--scene-goal` adds shared stakes (mostly pushed toward person A in the prompts).

Output: **WhatsApp-style bubbles** in the terminal, then a **JSON** transcript with `message`, `time_to_respond_s`, `time_writing_s`, `timestamp_utc`, and `display_name`.

### Example: friends texting after work

```bash
python chat.py \
  --vibe "Evening, both on the couch, half-watching TV, texting like friends do." \
  --name-1 Jordan \
  --name-2 Riley \
  --goal-for-name-1 "Vent about your new boss without sounding whiny." \
  --goal-for-name-2 "Gently steer toward what actually matters in life." \
  --max-messages 10
```

### Example: optional scene stakes + save JSON

```bash
python chat.py \
  --vibe "High-pressure day; you're both texting between meetings, terse but real." \
  --scene-goal "Jordan wants Riley to agree to cover one weekend shift." \
  --name-1 Jordan \
  --name-2 Riley \
  --goal-for-name-1 "You need that coverage but can't sound desperate." \
  --goal-for-name-2 "You're overloaded and skeptical of last-minute asks." \
  --max-messages 12 \
  --json-out transcript.json
```

### Example: different models per person

```bash
python chat.py \
  --vibe "Quiet, slow texting; late night." \
  --name-1 Alex \
  --name-2 Sam \
  --goal-for-name-1 "You're lonely and dance around saying it." \
  --goal-for-name-2 "You're supportive but exhausted." \
  --model-1 gemma2:9b \
  --model-2 llama3.1:8b \
  --max-messages 8
```

### Useful flags

| Flag | Meaning |
|------|--------|
| `--vibe` | Scene and tone (required) |
| `--name-1` / `--name-2` | Display names; **A** opens, **B** replies second |
| `--goal-for-name-1` / `--goal-for-name-2` | Private angle for each person |
| `--scene-goal` | Optional; mainly for person A to pursue |
| `--max-messages` | Total bubbles (alternating A, B, …) |
| `--model-1` / `--model-2` | Ollama model per person |
| `--pacing-min` / `--pacing-max` | Random pause between replies (seconds); set both `0` for no pause |
| `--json-out` | Write the JSON transcript to a file |

---

## License

Use and modify however you like for your own projects.
