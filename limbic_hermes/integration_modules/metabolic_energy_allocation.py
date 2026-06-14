"""
metabolic_energy_allocation.py
===============================
Brain glucose + ATP → cognitive resources.
Task load depletes glycogen → lactate.
Astrocyte glycogen repletion during rest.
"""
from typing import Dict


def compute_metabolic_allocation(state, drive_task_load: float = 0.0, drive_rest_need: float = 0.0) -> Dict[str, float]:
    """
    Returns:
        cognitive_reserve: ATP - task_load
        energy_status: glycogen + lactate shuttle
        glycogen_depletion: inverse of glycogen reserve
    """
    atp = state.mitochondrial_atp
    glycogen = state.glycogen
    lactate = state.lactate

    reserve = min(1.0, max(0.0, atp * 0.6 - drive_task_load * 0.4))
    status = min(1.0, glycogen * 0.5 + lactate * 0.3 + atp * 0.2)
    depletion = min(1.0, 1.0 - glycogen)

    return {
        "cognitive_reserve": round(reserve, 3),
        "energy_status": round(status, 3),
        "glycogen_depletion": round(depletion, 3),
    }
