"""
circadian_metabolic_coupling.py
================================
SCN phase controls cortisol peak (AM) and melatonin (night).
Metabolic rate varies with circadian (higher during day).
Orexin peaks in wake, drops at night.
"""
import math
from typing import Dict


def compute_circadian_metabolic(state, circadian_hour: float = 12.0) -> Dict[str, float]:
    """
    Returns:
        metabolic_phase: day vs night energy state
        circadian_alignment: SCN phase vs actual time match
        energy_rhythm: orexin + histamine - melatonin
    """
    scn = state.scn_phase
    orexin = state.orexin
    histamine = state.histamine
    melatonin = state.melatonin
    cortisol = state.cortisol

    expected_scn = (circadian_hour % 24.0) / 24.0
    alignment = 1.0 - min(abs(scn - expected_scn), 1.0 - abs(scn - expected_scn)) * 2

    phase = min(1.0, orexin * 0.4 + histamine * 0.3 + cortisol * 0.3)
    rhythm = min(1.0, max(0.0, orexin * 0.4 + histamine * 0.3 - melatonin * 0.3))

    return {
        "metabolic_phase": round(phase, 3),
        "circadian_alignment": round(max(0.0, alignment), 3),
        "energy_rhythm": round(rhythm, 3),
    }
