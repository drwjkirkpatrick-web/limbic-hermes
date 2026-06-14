"""
excitotoxicity_protection.py
=============================
Glutamate excess + low GLT-1 + low ATP -> excitotoxicity.
GABA and glycine provide protection.
Adenosine suppresses glutamate.
"""
from typing import Dict


def compute_excitotoxicity_protection(state) -> Dict[str, float]:
    """
    Returns:
        excitotoxicity_risk: glutamate + quinolinic + low GLT-1 + low ATP
        protection_strength: GABA + glycine + adenosine + GLT-1
        net_risk: risk - protection
    """
    glu = state.glutamate
    quin = state.quinolinic_acid
    glt1 = state.glt1_activity
    atp = state.mitochondrial_atp
    gaba = state.gaba
    glycine = state.glycine
    adenosine = state.adenosine

    risk = min(1.0, glu * 0.3 + quin * 0.25 + (1.0 - glt1) * 0.2 + (1.0 - atp) * 0.25)
    protection = min(1.0, gaba * 0.25 + glycine * 0.2 + adenosine * 0.3 + glt1 * 0.25)
    net = max(0.0, risk - protection * 0.8)

    return {
        "excitotoxicity_risk": round(risk, 3),
        "protection_strength": round(protection, 3),
        "net_risk": round(net, 3),
    }
