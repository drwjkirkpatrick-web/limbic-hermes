"""
thermogenesis_arousal.py
=========================
Brown adipose thermogenesis (NE-driven) raises body temperature.
Preoptic warmth suppresses orexin/histamine (sleep-promoting).
"""
from typing import Dict


def compute_thermogenesis_arousal(state) -> Dict[str, float]:
    """
    Returns:
        thermogenesis_level: brown adipose activity
        warmth_suppression: preoptic warmth -> wake suppression
        arousal_thermal_index: NE + BAT - PO warmth
    """
    bat = state.brown_adipose_activity
    po = state.preoptic_warmth
    ne = state.norepinephrine
    temp = state.body_temperature

    thermo = min(1.0, bat * 0.6 + ne * 0.4)
    suppression = min(1.0, po * 0.7 + temp * 0.3)
    arousal = min(1.0, max(0.0, ne * 0.4 + bat * 0.3 - po * 0.3))

    return {
        "thermogenesis_level": round(thermo, 3),
        "warmth_suppression": round(suppression, 3),
        "arousal_thermal_index": round(arousal, 3),
    }
