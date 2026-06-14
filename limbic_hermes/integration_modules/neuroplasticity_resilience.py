"""
neuroplasticity_resilience.py
==============================
BDNF + neurogenesis + LTP threshold resilience.
Chronic stress erodes plasticity; enrichment restores it.
"""
from typing import Dict


def compute_resilience_state(state) -> Dict[str, float]:
    """
    Returns:
        plasticity_index: BDNF + neurogenesis + low dynorphin
        resilience_capacity: serotonin + NPY + low cortisol
        vulnerability: high cytokines + low BDNF + low serotonin
    """
    bdnf = state.bdnf
    neurogenesis = state.neurogenesis_rate
    dynorphin = state.dynorphin
    serotonin = state.serotonin
    npy = state.neuropeptide_y
    cort = state.cortisol
    cytokines = state.cytokine_load

    plasticity = min(1.0, bdnf * 0.4 + neurogenesis * 0.3 + (1.0 - dynorphin) * 0.3)
    resilience = min(1.0, serotonin * 0.3 + npy * 0.4 + (1.0 - cort) * 0.3)
    vuln = min(1.0, cytokines * 0.4 + (1.0 - bdnf) * 0.3 + (1.0 - serotonin) * 0.3)

    return {
        "plasticity_index": round(plasticity, 3),
        "resilience_capacity": round(resilience, 3),
        "vulnerability": round(vuln, 3),
    }
