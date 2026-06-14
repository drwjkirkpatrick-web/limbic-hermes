"""
hpa_feedback.py
===============
Cortisol -> suppresses CRF (GR-mediated negative feedback).
Impaired feedback (chronic stress) -> sustained CRF.
"""
from typing import Dict


def compute_hpa_feedback(state) -> Dict[str, float]:
    """
    Returns:
        feedback_integrity: cortisol's ability to suppress CRF
        sustained_stress: high CRF despite cortisol
        recovery_capacity: intact feedback + low cytokines
    """
    cort = state.cortisol
    crf = state.crf
    pvn = state.pvn_crf
    cytokines = state.cytokine_load
    bdnf = state.bdnf

    expected_crf_suppression = cort * 0.8
    feedback_integrity = min(1.0, max(0.0, expected_crf_suppression / max(crf, 0.01)))
    sustained = min(1.0, crf * 0.5 + pvn * 0.3 + cytokines * 0.2)
    recovery = min(1.0, feedback_integrity * 0.6 + bdnf * 0.2 + (1.0 - cytokines) * 0.2)

    return {
        "feedback_integrity": round(feedback_integrity, 3),
        "sustained_stress": round(sustained, 3),
        "recovery_capacity": round(recovery, 3),
    }
