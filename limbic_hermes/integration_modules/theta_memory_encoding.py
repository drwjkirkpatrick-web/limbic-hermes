"""
theta_memory_encoding.py
=========================
Theta-gamma coupling gates memory encoding.
Medial septum paces theta; high theta + ACh = strong encoding.
"""
from typing import Dict


def compute_theta_memory_state(state) -> Dict[str, float]:
    """
    Returns:
        encoding_strength: theta-gamma * ACh * low cortisol
        retrieval_state: theta + BDNF + low NE
        consolidation_quality: NREM delta + BDNF
    """
    tgc = state.theta_gamma_coupling
    ach = state.acetylcholine
    cort = state.cortisol
    theta = state.hippocampal_theta
    bdnf = state.bdnf
    ne = state.norepinephrine
    nrem = state.nrem_slow_wave

    encoding = min(1.0, tgc * 0.4 + ach * 0.3 + (1.0 - cort) * 0.3)
    retrieval = min(1.0, theta * 0.4 + bdnf * 0.3 + (1.0 - ne) * 0.3)
    consolidation = min(1.0, nrem * 0.5 + bdnf * 0.5)

    return {
        "encoding_strength": round(encoding, 3),
        "retrieval_state": round(retrieval, 3),
        "consolidation_quality": round(consolidation, 3),
    }
