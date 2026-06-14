"""
hormonal_mood.py
================
Estrogen enhances serotonin + BDNF.
Progesterone -> allopregnanolone -> GABA-A enhancement (calming).
Testosterone modulates dominance and aggression.
"""
from typing import Dict


def compute_hormonal_mood(state) -> Dict[str, float]:
    """
    Returns:
        estrogenic_mood: serotonin + BDNF boost
        progestogenic_calm: GABA-A sensitivity boost
        androgenic_drive: testosterone + dopamine
    """
    estrogen = state.estrogen
    prog = state.progesterone
    allo = state.allopregnanolone
    test = state.testosterone
    ser = state.serotonin
    bdnf = state.bdnf
    gaba_sens = state.gaba_a_sensitivity
    da = state.dopamine

    est_mood = min(1.0, ser * 0.3 + bdnf * 0.3 + estrogen * 0.4)
    prog_calm = min(1.0, gaba_sens * 0.4 + allo * 0.3 + prog * 0.3)
    androgen = min(1.0, test * 0.5 + da * 0.3 + (1.0 - state.cortisol) * 0.2)

    return {
        "estrogenic_mood": round(est_mood, 3),
        "progestogenic_calm": round(prog_calm, 3),
        "androgenic_drive": round(androgen, 3),
    }
