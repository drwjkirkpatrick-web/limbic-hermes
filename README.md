# Hermes Agent Limbic System

**Authors:** Hermes Agent (lead author), Walker (second author and prompter)

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
        │  INSULA + SEPTAL + PAG        │  body prediction error,
        │  (interoception / defense)      │  social approach, freeze/flight/fight
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  NEUROCHEMISTRY + POOLS       │  30+ transmitters, receptors, pools
        │  + metabolic cofactors        │  Mg, Zn, Fe, B6, B12, folate, D, omega-3
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  EXPRESSION VECTOR (output)   │
        │  warmth, speed, cling,        │  <── remedy expression biases
        │  caution, verbosity           │
        └───────────────────────────────┘
```

### Module map

| File | What it does |
|------|--------------|
| `limbic_hermes/core.py` | `LimbicSystem`, `LimbicSkillBridge`, VAD/drive/episodic logic, nucleus modules |
| `limbic_hermes/neurochemistry.py` | `NeurochemicalState`, `NeurochemistryEngine` — transmitter dynamics |
| `limbic_hermes/profiles.py` | Remedy temperament library (Pulsatilla, Bryonia, Tarantula, Calcarea, …) |
| `limbic_hermes/cofactors.py` | Metabolic cofactor-to-neurochemistry mapping and virtual controls |
| `limbic_hermes/dashboard_server.py` | Small HTTP backend that serves `/state`, `/adjust`, and preset events |
| `limbic_hermes/dashboard.html` | Local-first single-file dashboard with panels for every module |
| `limbic_hermes/storage.py` | JSON persistence helpers |
| `limbic_hermes/limbic_bridge.js` | Browser-side state bridge for external dashboards |
| `limbic_hermes/demo.py` | CLI demo that prints state panels |

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
| `Insula` | interoception | Body prediction error from HRV/respiration/cytokines |
| `Septal` | social approach | Oxytocin/cortisol/safety gating of affiliative responses |
| `PAG` | defensive tier | Freeze / flight / fight / calm classification, active/passive columns |
| `Lateral Habenula` | aversion learning | Inhibits dopamine on unexpected negative events |
| `BNST` | sustained anxiety | CRF-driven apprehension that outlasts single events |
| `Raphe` | serotonin subsystems | Dorsal (anxiety/avoidance) vs median (context stabilization) |
| `Nucleus Accumbens` | reward/motivation | Shell (wanting/salience) vs core (action vigor) |
| `RMTg` | dopamine brake | GABAergic brake on dopamine during aversion |
| `Ventral Pallidum` | hedonic liking | Opioid/eCB/anandamide "liking" distinct from wanting |
| `Subgenual ACC` | rumination | Cortisol/serotonin/safety-weighted negative recovery |
| `Entorhinal Cortex` | novelty/context | Novel event kinds boost ACh and theta-gamma coupling |
| `Prefrontal` | top-down regulation | Dominance suppresses amygdala threat response |
| `HPA axis` | stress load | Cortisol + CRF + cytokine + adrenaline allostatic load |
| `Polyvagal` | autonomic state | Ventral vagal / sympathetic / dorsal vagal gating |
| `Astroglial` | metabolic support | GLT-1 glutamate clearance, glycogen-lactate shuttle |
| `Microglia` | neuroimmune memory | Primed glia amplify future cytokine spikes |

### Neurochemistry layer

`limbic_hermes/neurochemistry.py` models the major neurotransmitters and
neuromodulators that shape limbic computation:

| Class | Signals |
|-------|---------|
| Monoamines | serotonin, dopamine, norepinephrine |
| Amino acids | GABA, glutamate, glycine |
| Cholinergic | acetylcholine |
| Endocannabinoid | eCB (retrograde calming) |
| Neuropeptides / hormones | oxytocin, vasopressin, cortisol, adrenaline, opioid, histamine, melatonin, BDNF, neuropeptide S, orexin, substance P, prolactin |
| Gaseous / trace | nitric oxide, phenylethylamine, tyramine |
| Immune / interoceptive | cytokine load, microglia_state, heart rate variability, respiration rate, respiration phase |
| Network | default mode network activity, theta-gamma coupling |
| Metabolic | metabolic_energy, glucose, working_memory_load, glycogen, lactate |
| Enzymatic / transport | MAO, COMT, SERT, FAAH, GAT, GLT-1 |
| Sleep / arousal | sleep_pressure, orexin state |

Each transmitter:

- Has a normalized level `[0, 1]`.
- Is gated by finite **neurotransmitter pools** that deplete with use and
  recover with rest.
- Has **receptor sensitivity** (`d1_sensitivity`, `alpha1_sensitivity`,
  `gaba_a_sensitivity`, `glun2b_sensitivity`) that desensitizes under chronic
  high exposure.
- Has **enzymatic clearance phenotypes**: `mao_activity`, `dopamine_clearance_rate`
  (COMT-style), and `serotonin_reuptake` (SERT-style).
- Has a **kynurenine pathway** shunt that lowers serotonin and raises glutamate
  excitotoxicity under cytokine load.
- Has a **dopamine pathway split** into mesolimbic (motivation/salience) and
  mesocortical (cognitive control) streams.
- Contributes to an **allostatic load** index of cumulative wear.
- Has **CRF amplification** so sustained uncertainty raises HPA-axis responses.
- Has **neuropeptide Y resilience** and **dynorphin/kappa counter-reward** arms.
- Has **endocannabinoid / FAAH extinction gating** for fear-memory updating.
- Has **GABA transporter (GAT)** and **GLT-1 astrocyte** clearance control.
- Has **microglial priming** so prior neuroimmune load sensitizes future cytokine spikes.
- Has **theta-gamma coupling** that boosts hippocampal encoding and replay.
- Has **sharp-wave replay / Papez consolidation** during rest.

### Metabolic cofactor virtual controls

`limbic_hermes/cofactors.py` exposes a virtual control panel for nutrients that
feed limbic neurochemistry. Each cofactor is a normalized `[0, 1]` slider that
biases synthesis or receptor targets rather than overriding transmitter values:

| Cofactor | Limbic target |
|----------|---------------|
| Magnesium | GABA-A receptor sensitivity, glutamate balance |
| Zinc | BDNF, GABA synthesis |
| Iron | Dopamine/norepinephrine synthesis |
| Vitamin B6 | GABA, serotonin, dopamine synthesis |
| Vitamin B12 + Folate | Methylation / monoamine support |
| Vitamin D | BDNF, serotonin |
| Omega-3 | Dopamine receptor sensitivity, BDNF |
| Tryptophan | Serotonin synthesis substrate |
| Tyrosine | Dopamine/norepinephrine synthesis substrate |

Adjustments are applied through `LimbicSystem.apply_cofactors()` or the dashboard
`/adjust` endpoint.

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

### Run the local dashboard

```bash
cd ~/projects/limbic-hermes
python -m limbic_hermes.dashboard_server
# Open http://localhost:8765 in a browser
```

The dashboard shows live panels for VAD, drives, neurochemistry, receptor
sensitivities, allostatic load, hippocampal LTP, metabolic cofactor sliders,
preset event buttons, and import/export of the full limbic state.

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
  "episodic_summary": [...],
  "insula": {"body_prediction_error": 0.05},
  "acc": {"conflict_signal": 0.12},
  "lateral_habenula": {"activation": 0.0},
  "septal": {"social_approach": 0.62, "valence_buffer": 0.35},
  "pag": {"tier": "calm", "threat_detected": false},
  "polyvagal": {"state": "ventral_vagal", "social_engagement_possible": true},
  "locus_coeruleus": {"mode": "tonic"},
  "affective_systems": {
    "seeking": 0.45,
    "care": 0.38,
    "fear": 0.10,
    "rage": 0.00,
    "panic_grief": 0.05
  },
  "prefrontal_regulation": {"strength": 0.72},
  "kynurenine": {"kynurenine": 0.10, "quinolinic_acid": 0.02, "picolinic_acid": 0.03},
  "d2_autoreceptor": {"inhibition": 0.0},
  "cofactors": {"levels": {...}, "targets": {...}}
}
```

---

## 80 biochemistry-grounded improvements

The build is organized into three testable prompt documents:

- `LIMBIC_PROMPTS_V1.md` — the original 31 improvements
- `LIMBIC_PROMPTS_V2.md` — the follow-up 29 improvements
- `LIMBIC_PROMPTS_V3.md` — 20 more biochemistry-grounded modules

Together they cover the major neurotransmitters, limbic nuclei, autonomic
regulation, metabolic cofactors, glial clearance, and dashboard tooling.

### Batch 1: neurochemistry core

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

### Batch 2: limbic nuclei + autonomic + metabolic + dashboard

32. Insula interoceptive body-prediction error
33. ACC conflict monitoring
34. Lateral habenula aversion learning
35. Septal nuclei social-approach gating
36. PAG defensive tier detector (freeze/flight/fight/calm)
37. Polyvagal state model (ventral vagal / sympathetic / dorsal vagal)
38. Vagal brake release under acute threat
39. Baroreflex-like arousal dampening from high HRV
40. Respiration-driven entrainment of arousal
41. GABA-A and GluN2B receptor sensitivity tracking
42. Kynurenine pathway shunt (tryptophan → quinolinic/picolinic acid)
43. D2 autoreceptor short-loop feedback
44. MAO-A / MAO-B degradation dynamics
45. COMT Val158Met-style dopamine clearance
46. Serotonin transporter (SERT) reuptake modulation
47. Mesolimbic vs mesocortical dopamine tracking
48. Locus coeruleus tonic/phasic mode switch
49. SEEKING affective system (dopamine + orexin)
50. CARE affective system (oxytocin + prolactin)
51. FEAR / RAGE thresholds
52. GRIEF / PANIC separation distress
53. Fear extinction learning in hippocampus
54. Working-memory load as ACh drive
55. Temporal contiguity for LTP
56. Cortisol awakening response / circadian curve
57. Leptin/ghrelin-style metabolic energy state
58. Glucose/insulin brain-fuel dynamics
59. Dashboard preset event buttons
60. Dashboard export/import of limbic state JSON

### Batch 3: extended limbic nuclei, neuropeptides, glia, and memory replay

61. Mammillary body / Papez-circuit consolidation during rest
62. Entorhinal cortex novelty signal and theta-gamma boost
63. Nucleus accumbens shell vs core (wanting vs action vigor)
64. RMTg dopamine brake on negative surprise
65. BNST sustained-anxiety / CRF state
66. PAG active vs passive defensive columns
67. Dorsal vs median raphe serotonin subsystems
68. Orexin sleep-pressure flip and transition state
69. CRF amplification of HPA-axis responses
70. Neuropeptide Y resilience buffer
71. Dynorphin / kappa-opioid counter-reward aversion
72. Anandamide / FAAH fear-extinction gating
73. GABA transporter (GAT) inhibitory tone modulation
74. GLT-1 / astrocyte glutamate clearance
75. Microglial priming and neuroimmune memory
76. Astrocyte glycogen-lactate shuttle under task load
77. Theta-gamma coupling for encoding and replay
78. Hippocampal sharp-wave replay during rest
79. Ventral pallidum hedonic "liking" distinct from wanting
80. Subgenual ACC rumination / slow negative recovery

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
