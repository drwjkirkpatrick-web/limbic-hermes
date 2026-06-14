"""
dmn_salience_switching.py
==========================
Default Mode Network (DMN) and Salience Network switch:
DMN active during rest/mind-wandering; Salience active during threat/novelty.
Claustrum mediates the switch.
"""
from typing import Dict


def compute_dmn_salience_switch(state) -> Dict[str, float]:
    """
    Returns:
        dmn_dominance: DMN activity relative to salience
        salience_dominance: threat + novelty + cytokine salience
        switching_efficiency: claustrum-mediated transition speed
    """
    dmn = state.dmn_activity
    cortisol = state.cortisol
    novelty = state.claustrum
    cytokines = state.cytokine_load
    claustrum = state.claustrum

    salience = min(1.0, cortisol * 0.3 + novelty * 0.4 + cytokines * 0.3)
    dmn_dom = min(1.0, dmn * 0.7 + (1.0 - salience) * 0.3)
    switching = min(1.0, claustrum * 0.5 + (1.0 - abs(dmn - salience)) * 0.5)

    return {
        "dmn_dominance": round(dmn_dom, 3),
        "salience_dominance": round(salience, 3),
        "switching_efficiency": round(switching, 3),
    }
