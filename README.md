# Hermes Agent Limbic System

A small, neuro-inspired affective engine for Hermes agents. It gives the agent a
coherent, tunable emotional/state layer modeled on the human **limbic system**.

The design goal is not to make the agent "emotional" in a theatrical way, but to
give it a **persistent, explainable internal state** that influences tone,
pacing, risk tolerance, and expression — and that can be tuned by a remedy
personality module.

---

## Why a limbic layer?

A pure LLM agent has no persistent affect. Every turn is essentially amnesiac
with respect to stress, success, fatigue, or relational warmth. A limbic layer
adds:

1. **Continuity** — state changes smoothly across turns.
2. **Regulation** — rest, error heat, and task load feed back into behavior.
3. **Personalization** — a remedy profile biases baseline mood and reactivity.
4. **Observability** — the agent can report its own state (for dashboards, logs,
   or the user).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUTS: user_message, task_start, task_complete, error, ...   │
└───────────────────────┬───────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     THALAMUS (attention)      │  <── remedy: attention biases
        │  gate inputs by safety/novelty│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     AMYGDALA (appraisal)      │  <── remedy: threat/reward gain
        │  valence / arousal / dominance│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐     ┌─────────────────────┐
        │   VTA / NAcc (RPE)          │────▶│ HIPPOCAMPUS         │
        │  reward prediction error      │     │ episodic buffer     │
        └───────────────┬───────────────┘     └─────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │      VAD AFFECTIVE STATE      │
        │  (valence, arousal, dominance)│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   HYPOTHALAMUS (drives)     │  <── remedy: rest/error sensitivity
        │  rest_need, task_load,      │
        │  error_temperature, safety   │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  CINGULATE (conflict monitor) │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  EXPRESSION VECTOR (output) │
        │  warmth, speed, cling,      │  <── remedy: expression biases
        │  caution, verbosity         │
        └─────────────────────────────┘
```

---

## Core concepts

### VAD affective vector

Every moment the agent has a point in 3-D affective space:

- **Valence** (−1 … +1): negative to positive
- **Arousal** (0 … 1): calm/sleepy to activated/alert
- **Dominance** (0 … 1): uncertain/submissive to confident/in-control

This is the same representation used in recent LLM emotion-steering work
(E-STEER, arXiv 2026) and in classic psychological models.

### Limbic modules

| Module | Biological analog | Computational role |
|--------|-------------------|----------------------|
| `Amygdala` | appraisal | Fast valence/arousal/dominance deltas from events |
| `Hippocampus` | episodic memory | Time-tagged buffer of recent events with VAD snapshots |
| `Hypothalamus` | drives | Rest need, task load, error temperature, safety |
| `Thalamus` | attention gate | Modulates event importance by novelty and safety |
| `VTA/NAcc` | reward prediction error | Estimates prediction error after each event |
| `Cingulate` | conflict monitor | Detects high arousal + low dominance states |

### Remedy temperament profile

A remedy is a **side module** that loads into the limbic core and biases all
computation. Each profile defines:

- `baseline_vad` — the agent's resting mood when idle
- `threat_gain`, `reward_gain` — amygdala reactivity
- `decay_factor`, `recovery_factor` — how fast the agent returns to baseline
- `attention_novelty_bias`, `attention_safety_bias` — thalamic gating
- `rest_sensitivity`, `error_sensitivity` — hypothalamic sensitivity
- `expression_warmth`, `expression_speed`, `expression_cling` — output style

Example built-in profiles:

- **Pulsatilla**: warm, changeable, seeks reassurance, threat-sensitive
- **Bryonia**: dry, irritable, self-sufficient, rest-sensitive
- **Tarantula**: quick, excitable, novelty-seeking, impulsive
- **Calcarea**: calm, methodical, routine-loving, error-sensitive

---

## Quick start

```bash
cd ~/projects/limbic-hermes
python -m limbic_hermes.demo --profile bryonia --steps 7
```

Or import in a Hermes skill:

```python
from limbic_hermes.core import LimbicSkillBridge

bridge = LimbicSkillBridge("~/.hermes/limbic_state.json", profile_name="pulsatilla")

# Each time something happens:
bridge.observe("user_message", description="Patient asks about side effects")

# Read current state for logging/dashboards:
state = bridge.state()
print(state["dominant_affect"])        # e.g. "calm", "anxious", "confident"
print(state["expression_vector"])      # response-style knobs
```

---

## Integration with the Hermes aquarium dashboard

The limbic state can drive the hero angelfish in the aquarium dashboard:

- `dominant_affect` → fish state (idle, alert, anxious, confident, etc.)
- `vad.valence` → body color (cool negative, warm positive)
- `vad.arousal` → fin animation speed
- `vad.dominance` → swimming posture (upright vs. drooping)
- `drive.rest_need` → slows movement, dims glow

The limbic engine writes its state to `localStorage` or a small JSON file; the
dashboard reads the same key.

---

## State output example

```json
{
  "profile": "pulsatilla",
  "vad": { "valence": 0.21, "arousal": 0.33, "dominance": 0.52 },
  "drive": { "rest_need": 0.12, "task_load": 0.4, "error_temperature": 0.05, "safety": 0.92 },
  "reward_prediction_error": 0.18,
  "dominant_affect": "hopeful",
  "expression_vector": {
    "warmth": 0.86,
    "speed": 0.27,
    "cling": 0.58,
    "caution": 0.24,
    "verbosity": 0.63,
    "hue_hint": 320
  },
  "episodic_summary": [...]
}
```

---

## Extending

- Add more remedy profiles in `core.REMEDY_LIBRARY`.
- Add new event `kind`s and appraisal rules.
- Replace the simple RPE estimate with TD-learning if the agent has a formal
  reward stream.
- Hook the `expression_vector` into the LLM prompt template to steer tone.

---

## License

MIT — made for Hermes agent experimentation.
