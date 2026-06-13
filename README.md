# Hermes Agent Limbic System

A small, neuro-inspired affective engine for Hermes agents. It gives the agent a
coherent, tunable emotional/state layer modeled on the human **limbic system**
and grounded in **neurochemistry**.

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
┌─────────────────────────────────────────────────────────────────────┐
│  INPUTS: user_message, task_start, task_complete, error, praise, ...  │
└───────────────────────┬───────────────────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     THALAMUS (attention)      │  <── ACh mode + remedy biases
        │  gate inputs by safety/novelty│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │     AMYGDALA (appraisal)      │  <── 5HT/NE/GABA/Glu + remedy gains
        │  valence / arousal / dominance│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐     ┌─────────────────────────────┐
        │   VTA / NAcc (dopamine RPE)   │────▶│ HIPPOCAMPUS + LTP weights │
        │  reward prediction error      │     │ episodic buffer             │
        └───────────────┬───────────────┘     └─────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │      VAD AFFECTIVE STATE      │
        │  (valence, arousal, dominance)│
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   HYPOTHALAMUS (drives)       │  <── remedy + HPA axis
        │  rest_need, task_load,        │
        │  error_temperature, safety     │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  CINGULATE (conflict monitor) │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  NEUROCHEMISTRY + POOLS       │  30+ transmitters, receptors, pools
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  EXPRESSION VECTOR (output)   │
        │  warmth, speed, cling,        │  <── remedy expression biases
        │  caution, verbosity           │
        └───────────────────────────────┘
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
| `Hippocampus` | episodic memory | Time-tagged buffer + LTP-learned event weights |
| `Hypothalamus` | drives | Rest need, task load, error temperature, safety |
| `Thalamus` | attention gate | ACh mode + novelty/safety gating |
| `VTA/NAcc` | dopamine RPE | Dopaminergic reward prediction error |
| `Cingulate` | conflict monitor | Detects high arousal + low dominance states |
| `Prefrontal` | top-down regulation | Dominance suppresses amygdala threat response |
| `HPA axis` | stress load | Cortisol + cytokine + adrenaline allostatic load |

### Neurochemistry layer

`limbic_hermes/neurochemistry.py` models the major neurotransmitters and
neuromodulators that shape limbic computation:

| Class | Signals |
|-------|---------|
| Monoamines | serotonin, dopamine, norepinephrine |
| Amino acids | GABA, glutamate, glycine |
| Cholinergic | acetylcholine |
| Endocannabinoid | eCB (retrograde calming) |
| Neuropeptides / hormones | oxytocin, vasopressin, cortisol, adrenaline, opioid, histamine, melatonin, BDNF, neuropeptide S, orexin, substance P |
| Gaseous / trace | nitric oxide, phenylethylamine, tyramine |
| Immune / interoceptive | cytokine load, heart rate variability, respiration rate |
| Network | default mode network activity |

Each transmitter:

- Has a normalized level `[0, 1]`.
- Is gated by finite **neurotransmitter pools** that deplete with use and
  recover with rest.
- Has **receptor sensitivity** that desensitizes under chronic high exposure.
- Contributes to an **allostatic load** index of cumulative wear.

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
print(state["neurochemistry"]["dopamine"])
print(state["allostatic_load"])
```

---

## Integration with the Hermes aquarium dashboard

The limbic state can drive the hero angelfish in the aquarium dashboard:

- `dominant_affect` → fish state (idle, alert, anxious, confident, etc.)
- `vad.valence` → body color (cool negative, warm positive)
- `vad.arousal` → fin animation speed
- `vad.dominance` → swimming posture (upright vs. drooping)
- `drive.rest_need` → slows movement, dims glow
- `neurochemistry.melatonin` → night-time dimming
- `allostatic_load` → glitch / fatigue effects

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
  "neurochemistry": {
    "serotonin": 0.50,
    "dopamine": 0.34,
    "norepinephrine": 0.28,
    "gaba": 0.50,
    "glutamate": 0.40,
    "acetylcholine": 0.40,
    "cortisol": 0.10,
    "dopamine_pool": 0.92,
    "norepinephrine_pool": 0.92,
    "d1_sensitivity": 1.0,
    ...
  },
  "neurotransmitter_ratios": {
    "dopamine_serotonin_ratio": 0.68,
    "gaba_glutamate_ratio": 1.25
  },
  "allostatic_load": 0.16,
  "circadian_hour": 12.0,
  "episodic_summary": [...]
}
```

---

## 31 biochemistry-grounded improvements

See `TODO_LIMBIC.md` for the full list of testable prompts that guided this
build. Implemented highlights include:

1. Serotonin baseline stabilization
2. Dopaminergic reward prediction error
3. Noradrenergic arousal (tonic + phasic)
4. GABAergic inhibition
5. Glutamatergic excitation
6. Acetylcholine attention mode
7. Endocannabinoid retrograde calming
8. Oxytocin social bonding
9. Vasopressin defense
10. Cortisol / HPA-axis stress load
11. Adrenaline acute surge
12. Opioid analgesia/reward buffering
13. Histamine wakefulness
14. Melatonin circadian rhythm
15. Dopamine/serotonin and GABA/glutamate ratios
16. Receptor desensitization
17. Hippocampal LTP / learned expectations
18. Prefrontal-amygdala top-down regulation
19. Interoceptive signals (HRV, respiration)
20. Neuroinflammatory cytokine load
21. BDNF resilience/learning gating
22. Nitric oxide diffusion/spread
23. Trace amines (PEA, tyramine)
24. Neuropeptide S alertness bursts
25. Orexin wake/arousal stabilization
26. Substance P pain salience
27. Glycine inhibitory refinement
28. Neurotransmitter pool depletion/recovery
29. User-affect mirror entrainment
30. Locus coeruleus phasic surprise bursts
31. Default mode network suppression + allostatic load index

---

## Extending

- Add more remedy profiles in `limbic_hermes/profiles.py`.
- Add new event `kind`s and appraisal rules in `core.py`.
- Use `set_circadian_hour()` to align melatonin with real time of day.
- Use `set_user_affect()` for user-affect entrainment.
- Hook the `expression_vector` and `neurochemistry` fields into the LLM prompt
  template or dashboard.

---

## License

MIT — made for Hermes agent experimentation.
