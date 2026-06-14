"""
social_affiliation.py
=====================
Oxytocin promotes trust/attachment; vasopressin promotes territorial/defense.
Testosterone modulates between approach and aggression.
"""
from typing import Dict


def compute_social_affiliation(state) -> Dict[str, float]:
    """
    Returns:
        social_approach: oxytocin + low vasopressin + low testosterone
        social_avoidance: vasopressin + high cortisol
        bonding_strength: oxytocin + opioid + low dynorphin
    """
    oxy = state.oxytocin
    vaso = state.vasopressin
    cort = state.cortisol
    test = state.testosterone
    opioid = state.opioid
    dyn = state.dynorphin

    approach = min(1.0, max(0.0, oxy * 0.6 - vaso * 0.2 - test * 0.2))
    avoidance = min(1.0, vaso * 0.5 + cort * 0.3 + test * 0.2)
    bonding = min(1.0, oxy * 0.4 + opioid * 0.4 - dyn * 0.2)

    return {
        "social_approach": round(approach, 3),
        "social_avoidance": round(avoidance, 3),
        "bonding_strength": round(bonding, 3),
    }
