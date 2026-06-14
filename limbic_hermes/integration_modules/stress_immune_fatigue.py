"""
stress_immune_fatigue.py
=========================
Chronic stress elevates cytokines via microglial priming,
which shunts tryptophan to kynurenine, lowering serotonin
and producing fatigue. Fatigue suppresses NE, which further
primes microglia in a positive feedback loop.
"""
from typing import Dict


def compute_stress_immune_fatigue(state) -> Dict[str, float]:
    """
    Returns:
        chronic_stress_index: composite of cortisol + CRF + microglia
        fatigue_level: driven by low serotonin + high cytokines + low ATP
        immune_resilience: inverse of priming (BDNF + NPY protect)
    """
    cortisol = state.cortisol
    crf = state.crf
    microglia = state.microglia_state
    cytokines = state.cytokine_load
    serotonin = state.serotonin
    ne = state.norepinephrine
    bdnf = state.bdnf
    npy = state.neuropeptide_y
    atp = state.mitochondrial_atp

    chronic_stress = min(1.0, (cortisol * 0.4 + crf * 0.3 + microglia * 0.3))
    fatigue = min(1.0, (1.0 - serotonin) * 0.3 + cytokines * 0.3 + (1.0 - ne) * 0.2 + (1.0 - atp) * 0.2)
    resilience = min(1.0, bdnf * 0.5 + npy * 0.3 + (1.0 - microglia) * 0.2)

    return {
        "chronic_stress_index": round(chronic_stress, 3),
        "fatigue_level": round(fatigue, 3),
        "immune_resilience": round(resilience, 3),
    }
