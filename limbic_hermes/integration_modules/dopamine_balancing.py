"""
dopamine_balancing.py
=====================
Mesolimbic (wanting) vs mesocortical (cognitive) balance.
D2 autoreceptor inhibits excessive dopamine.
VTA GABA brake prevents over-seeking.
"""
from typing import Dict


def compute_dopamine_balance(state) -> Dict[str, float]:
    """
    Returns:
        mesolimbic_bias: wanting vs thinking
        mesocortical_strength: cognitive control
        dopamine_stability: balance - autoreceptor - VTA GABA
    """
    mesolimbic = state.dopamine_mesolimbic
    mesocortical = state.dopamine_mesocortical
    d2_auto = state.d2_autoreceptor_inhibition
    vta_gaba = state.vta_gaba
    dopamine = state.dopamine

    limbic_bias = min(1.0, mesolimbic / max(0.01, mesolimbic + mesocortical))
    cortical = min(1.0, mesocortical * 0.6 + dopamine * 0.4)
    stability = min(1.0, max(0.0, 1.0 - d2_auto * 0.4 - vta_gaba * 0.4))

    return {
        "mesolimbic_bias": round(limbic_bias, 3),
        "mesocortical_strength": round(cortical, 3),
        "dopamine_stability": round(stability, 3),
    }
