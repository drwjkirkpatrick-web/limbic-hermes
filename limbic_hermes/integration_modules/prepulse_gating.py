"""
prepulse_gating.py
==================
Prepulse inhibition (PPI) gates startle response.
GABA and striatal signaling mediate PPI.
"""
from typing import Dict


def compute_prepulse_gating(state) -> Dict[str, float]:
    """
    Returns:
        startle_magnitude: NE + low safety - PPI
        gating_efficiency: PPI level
        sensorimotor_gating: PPI * GABA
    """
    ne = state.norepinephrine
    safety = 1.0 - state.cortisol
    ppi = state.prepulse_inhibition
    gaba = state.gaba

    startle = min(1.0, max(0.0, ne * 0.5 + (1.0 - safety) * 0.3 - ppi * 0.2))
    gating = min(1.0, ppi * 0.6 + gaba * 0.4)
    smg = min(1.0, ppi * 0.5 + gaba * 0.5)

    return {
        "startle_magnitude": round(startle, 3),
        "gating_efficiency": round(gating, 3),
        "sensorimotor_gating": round(smg, 3),
    }
