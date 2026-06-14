"""
sleep_homeostasis.py
====================
Homeostatic sleep pressure (adenosine) + circadian gating (SCN/melatonin).
NREM delta clears toxins (glymphatics). REM theta processes emotions.
"""
from typing import Dict


def compute_sleep_homeostasis(state, circadian_hour: float = 12.0) -> Dict[str, float]:
    """
    Returns:
        sleep_need: adenosine + sleep_pressure + melatonin
        sleep_efficiency: glymphatic flow / amyloid clearance
        recovery_state: NREM delta + REM theta + low NE
    """
    adenosine = state.adenosine
    sleep_pressure = state.sleep_pressure
    melatonin = state.melatonin
    nrem = state.nrem_slow_wave
    rem = state.rem_theta
    ne = state.norepinephrine
    glymphatic = state.glymphatic_flow
    amyloid = state.amyloid_beta

    sleep_need = min(1.0, adenosine * 0.35 + sleep_pressure * 0.35 + melatonin * 0.3)
    efficiency = min(1.0, glymphatic * 0.7 + (1.0 - amyloid) * 0.3)
    recovery = min(1.0, nrem * 0.4 + rem * 0.3 + (1.0 - ne) * 0.3)

    return {
        "sleep_need": round(sleep_need, 3),
        "sleep_efficiency": round(efficiency, 3),
        "recovery_state": round(recovery, 3),
    }
