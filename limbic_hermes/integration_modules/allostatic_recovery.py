"""
allostatic_recovery.py
========================
Allostatic load recovery depends on rest, sleep, BDNF, and
HPA-axis negative feedback restoration.
"""
from typing import Dict


def compute_allostatic_recovery(state) -> Dict[str, float]:
    """
    Returns:
        recovery_potential: rest_need + sleep + low cortisol
        load_vs_capacity: allostatic load / recovery capacity
        restoration_rate: BDNF + sleep + vagal tone
    """
    cort = state.cortisol
    bdnf = state.bdnf
    sleep_p = state.sleep_pressure
    nrem = state.nrem_slow_wave
    vagal = state.vagal_afferent
    ne = state.norepinephrine
    cytokines = state.cytokine_load

    # Estimate allostatic load (same formula as neurochemistry)
    load = min(1.0, cort * 0.3 + cytokines * 0.25 + (1.0 - state.heart_rate_variability) * 0.1)

    potential = min(1.0, (1.0 - cort) * 0.3 + sleep_p * 0.3 + nrem * 0.2 + vagal * 0.2)
    capacity = max(0.01, potential)
    load_ratio = min(1.0, load / capacity)
    restoration = min(1.0, bdnf * 0.3 + nrem * 0.3 + vagal * 0.2 + (1.0 - ne) * 0.2)

    return {
        "recovery_potential": round(potential, 3),
        "load_vs_capacity": round(load_ratio, 3),
        "restoration_rate": round(restoration, 3),
    }
