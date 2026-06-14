"""
gut_brain_stress.py
====================
Vagal afferents carry gut signals to brainstem (NTS).
Butyrate (SCFA) enhances GABA and reduces anxiety.
Gut inflammation (cytokines) activates HPA axis.
"""
from typing import Dict


def compute_gut_brain_stress(state) -> Dict[str, float]:
    """
    Returns:
        gut_resilience: butyrate + vagal tone - cytokines
        gut_stress: cytokines + low butyrate + cortisol
        microbial_mood: vagal + butyrate effects on anxiety
    """
    vagal = state.vagal_afferent
    butyrate = state.butyrate
    cytokines = state.cytokine_load
    cort = state.cortisol

    resilience = min(1.0, max(0.0, vagal * 0.4 + butyrate * 0.3 - cytokines * 0.3))
    stress = min(1.0, cytokines * 0.4 + (1.0 - butyrate) * 0.3 + cort * 0.3)
    mood = min(1.0, max(0.0, vagal * 0.3 + butyrate * 0.4 - cort * 0.3))

    return {
        "gut_resilience": round(resilience, 3),
        "gut_stress": round(stress, 3),
        "microbial_mood": round(mood, 3),
    }
