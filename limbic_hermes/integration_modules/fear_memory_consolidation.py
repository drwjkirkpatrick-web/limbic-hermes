"""
fear_memory_consolidation.py
=============================
Strong BLA + high NE → fear encoding.
NREM sleep consolidates fear memory.
IL extinction during wake (BDNF + anandamide).
"""
from typing import Dict


def compute_fear_memory_state(state) -> Dict[str, float]:
    """
    Returns:
        fear_memory_strength: BLA * NE * (1 - GABA)
        extinction_level: IL + anandamide - CRF
        reconsolidation_risk: high BDNF + recent retrieval + high NE
    """
    bla = state.bla
    ne = state.norepinephrine
    gaba = state.gaba
    il = state.il_activity
    anand = state.anandamide
    crf = state.crf
    bdnf = state.bdnf

    fear = min(1.0, bla * 0.4 + ne * 0.3 + (1.0 - gaba) * 0.3)
    extinction = min(1.0, max(0.0, il * 0.4 + anand * 0.3 - crf * 0.3))
    reconsolidation = min(1.0, bdnf * 0.5 + ne * 0.3)

    return {
        "fear_memory_strength": round(fear, 3),
        "extinction_level": round(extinction, 3),
        "reconsolidation_risk": round(reconsolidation, 3),
    }
