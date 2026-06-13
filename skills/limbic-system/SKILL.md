---
name: limbic-system
description: Use when driving or inspecting the Hermes agent's affective limbic state, remedy temperament profiles, or affect-aware prompt prefixes.
version: 1.0.0
author: Walker Kirkpatrick
license: MIT
metadata:
  hermes:
    tags: [limbic, affect, emotion, remedy, personality, agent-state]
    related_skills: [hermes-agent]
---

# Limbic System Skill

This skill gives a Hermes agent a persistent, neuro-inspired affective state
layer and lets it be tuned by homeopathic remedy personalities.

## What it provides

- `limbic_observe(kind, ...)` — feed events into the affective engine
- `limbic_state(profile)` — read current VAD, drives, expression vector
- `limbic_prompt_prefix(profile, intensity)` — render a prompt prefix
- `limbic_set_profile(profile)` — switch remedy personality
- `limbic_rest(duration_sec)` — apply a rest pulse
- `limbic_task_start/complete/error(...)` — shorthand event helpers

## Quick recipes

### Report that a task started
```
limbic_task_start(description="Researching drug interactions")
```

### Report success
```
limbic_task_complete(description="Wrote summary", success=True)
```

### Check how the agent is feeling
```
limbic_state()
```

### Switch temperament to Pulsatilla
```
limbic_set_profile("pulsatilla")
```

### Render an affect-aware system prompt prefix
```
limbic_prompt_prefix(intensity=0.7)
```

## Integration with aquarium dashboard

The limbic state is saved as JSON in `~/.hermes/limbic_state/default.json`.
The dashboard reads the same file (or localStorage mirror) to drive the hero
angelfish's color, animation, and posture.

## Available remedy profiles

- default
- pulsatilla
- bryonia
- tarantula
- calcarea
- arsenicum
- lycopodium
- natrum-muriaticum
- sulphur
- phosphorus
- nux-vomica
- hepar-sulphuris
- sepia
- gelsemium

## Pitfalls

- This skill is a **steering layer**, not a clinical decision-maker. It must not
  override diagnosis, prescription, or safety requirements.
- The `limbic_prompt_prefix` intensity should stay <= 0.8 for clinical work to
  avoid over-steering tone.
- State persists across sessions; switching profile resets the state file.
