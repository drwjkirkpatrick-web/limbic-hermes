"""
pain_modulation.py
==================
Substance P (pain signal).
Spinal opioid gates pain (mu-opioid).
Descending PAG-RVMM control.
Abeta touch fibers inhibit pain.
"""
from typing import Dict


def compute_pain_modulation(state) -> Dict[str, float]:
    """
    Returns:
        pain_level: substance P - spinal_opioid - Abeta fiber
        analgesia_strength: spinal_opioid + PAG activation
        gate_status: net inhibitory control
    """
    sp = state.substance_p
    spinal_op = state.spinal_opioid
    ab = state.ab_fiber
    rvmm = state.rvmm_activity
    pag = state.pag_periaqueductal_gray

    pain = min(1.0, max(0.0, sp * 0.5 - spinal_op * 0.3 - ab * 0.2))
    analgesia = min(1.0, spinal_op * 0.4 + rvmm * 0.3 + pag * 0.3)
    gate = min(1.0, analgesia + ab * 0.3)

    return {
        "pain_level": round(pain, 3),
        "analgesia_strength": round(analgesia, 3),
        "gate_status": round(gate, 3),
    }
