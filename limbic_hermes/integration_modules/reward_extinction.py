"""
reward_extinction.py
====================
Dopamine PE drives reward learning (BLA → NAcc).
Anandamide gates extinction via vmPFC/IL.
Dynorphin (kappa opioid) blocks reward seeking.
"""
from typing import Dict


def compute_reward_extinction_state(state) -> Dict[str, float]:
    """
    Returns:
        reward_learning: dopamine PE + mesolimbic activation
        extinction_state: anandamide + IL activity - dynorphin
        aversion_strength: dynorphin + CRF
    """
    da = state.dopamine
    mesolimbic = state.dopamine_mesolimbic
    anandamide = state.anandamide
    il_act = state.il_activity
    dynorphin = state.dynorphin
    crf = state.crf
    bdnf = state.bdnf

    reward_learn = min(1.0, da * 0.5 + mesolimbic * 0.3 + bdnf * 0.2)
    extinction = min(1.0, max(0.0, anandamide * 0.4 + il_act * 0.4 - dynorphin * 0.3))
    aversion = min(1.0, dynorphin * 0.6 + crf * 0.4)

    return {
        "reward_learning": round(reward_learn, 3),
        "extinction_state": round(extinction, 3),
        "aversion_strength": round(aversion, 3),
    }
