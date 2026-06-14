"""
attention_salience.py
=====================
LC phasic mode gates salient events (NE burst).
Pulvinar amplifies relevant stimuli.
Claustrum controls awareness gating.
"""
from typing import Dict


def compute_attention_salience(state) -> Dict[str, float]:
    """
    Returns:
        attention_focus: ACh + phasic NE + low sleep_pressure
        distractibility: high tonic NE + low ACh + high novelty
        awareness_level: claustrum + pulvinar
    """
    ach = state.acetylcholine
    ne = state.norepinephrine
    lc_phasic = state.lc_mode
    sleep_p = state.sleep_pressure
    pulvinar = state.pulvinar
    claustrum = state.claustrum

    focus = min(1.0, ach * 0.4 + lc_phasic * 0.3 + (1.0 - sleep_p) * 0.3)
    distract = min(1.0, (1.0 - lc_phasic) * ne * 0.5 + (1.0 - ach) * 0.3 + sleep_p * 0.2)
    awareness = min(1.0, claustrum * 0.5 + pulvinar * 0.5)

    return {
        "attention_focus": round(focus, 3),
        "distractibility": round(distract, 3),
        "awareness_level": round(awareness, 3),
    }
