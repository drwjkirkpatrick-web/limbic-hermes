"""
limbic_hermes/cofactors.py
==========================
Metabolic cofactors (vitamins, minerals, amino-acid precursors) that influence
limbic neurochemistry.

This module maps cofactor levels to adjustments in neurotransmitter synthesis,
receptor sensitivity, and allostatic load. It is intentionally simplified and
educational — not a prescription system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Cofactor:
    """One metabolic cofactor with its target neurochemical effects."""
    name: str
    category: str  # vitamin, mineral, amino_acid, methyl_donor, etc.
    default_level: float = 0.5  # 0..1
    min_level: float = 0.0
    max_level: float = 1.0
    # Mapping: neurochemical key -> (slope, target ceiling/floor hint)
    # Positive slope increases the target; negative decreases it.
    effects: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Cofactor library built from human-metabolic-optimization references
# ---------------------------------------------------------------------------

COFACTOR_LIBRARY: Dict[str, Cofactor] = {
    "magnesium": Cofactor(
        name="Magnesium",
        category="mineral",
        default_level=0.5,
        effects={
            "gaba": (0.25, 1.0),
            "serotonin": (0.15, 1.0),
            "dopamine": (0.10, 1.0),
            "norepinephrine": (-0.10, 0.0),  # calming sympathetic tone
        },
        description="COMT cofactor; supports GABA tone and calms catecholamine surges.",
    ),
    "zinc": Cofactor(
        name="Zinc",
        category="mineral",
        default_level=0.5,
        effects={
            "gaba": (0.20, 1.0),
            "glutamate": (-0.15, 0.0),  # zinc modulates NMDA/glutamatergic excitation
            "bdnf": (0.15, 1.0),
        },
        description="Modulates GABA and glutamate balance; supports neuroplasticity.",
    ),
    "iron": Cofactor(
        name="Iron",
        category="mineral",
        default_level=0.5,
        effects={
            "dopamine": (0.25, 1.0),
            "norepinephrine": (0.15, 1.0),
            "gaba": (-0.10, 0.0),
        },
        description="Tyrosine hydroxylase cofactor; needed for dopamine/NE synthesis.",
    ),
    "vitamin_b6": Cofactor(
        name="Vitamin B6 (P5P)",
        category="vitamin",
        default_level=0.5,
        effects={
            "dopamine": (0.20, 1.0),
            "serotonin": (0.20, 1.0),
            "gaba": (0.20, 1.0),
            "histamine": (-0.15, 0.0),  # DAO requires B6
        },
        description="Cofactor for dopamine, serotonin, and GABA synthesis/degradation.",
    ),
    "vitamin_b12": Cofactor(
        name="Vitamin B12 (methylcobalamin)",
        category="vitamin",
        default_level=0.5,
        effects={
            "dopamine": (0.10, 1.0),
            "norepinephrine": (0.10, 1.0),
            "bdnf": (0.10, 1.0),
            "cytokine_load": (-0.10, 0.0),
        },
        description="Methylation cofactor; supports myelin, mood, and neurotrophic factors.",
    ),
    "folate": Cofactor(
        name="Folate / L-methylfolate",
        category="vitamin",
        default_level=0.5,
        effects={
            "serotonin": (0.20, 1.0),
            "dopamine": (0.10, 1.0),
            "bdnf": (0.10, 1.0),
        },
        description="Methyl donor for monoamine synthesis; supports mood stability.",
    ),
    "vitamin_d": Cofactor(
        name="Vitamin D",
        category="vitamin",
        default_level=0.5,
        effects={
            "serotonin": (0.15, 1.0),
            "dopamine": (0.10, 1.0),
            "bdnf": (0.15, 1.0),
            "cytokine_load": (-0.15, 0.0),
        },
        description="Supports monoamine synthesis, BDNF, and neuroimmune tone.",
    ),
    "s_adenosyl_methionine": Cofactor(
        name="SAMe",
        category="methyl_donor",
        default_level=0.3,
        effects={
            "dopamine": (-0.15, 0.0),  # methylates dopamine for degradation
            "norepinephrine": (-0.10, 0.0),
            "serotonin": (0.10, 1.0),
        },
        description="Methyl donor; increases catecholamine degradation, supports serotonin.",
    ),
    "l_tyrosine": Cofactor(
        name="L-Tyrosine",
        category="amino_acid",
        default_level=0.4,
        effects={
            "dopamine": (0.25, 1.0),
            "norepinephrine": (0.15, 1.0),
            "adrenaline": (0.10, 1.0),
        },
        description="Dopamine/NE substrate support; rate-limited by tyrosine hydroxylase.",
    ),
    "l_tryptophan": Cofactor(
        name="L-Tryptophan / 5-HTP",
        category="amino_acid",
        default_level=0.4,
        effects={
            "serotonin": (0.25, 1.0),
            "melatonin": (0.15, 1.0),
        },
        description="Serotonin/melatonin precursor; competes with tyrosine for transport.",
    ),
    "taurine": Cofactor(
        name="Taurine",
        category="amino_acid",
        default_level=0.4,
        effects={
            "gaba": (0.20, 1.0),
            "glycine": (0.15, 1.0),
            "glutamate": (-0.10, 0.0),
        },
        description="GABA-glycine modulator; calming inhibitory support.",
    ),
    "omega3": Cofactor(
        name="Omega-3 fatty acids",
        category="lipid",
        default_level=0.4,
        effects={
            "bdnf": (0.20, 1.0),
            "dopamine": (0.10, 1.0),
            "cytokine_load": (-0.15, 0.0),
        },
        description="Supports membrane fluidity, neuroplasticity, and anti-inflammatory tone.",
    ),
    "zinc_copper_balance": Cofactor(
        name="Zinc:Copper balance",
        category="mineral_ratio",
        default_level=0.5,
        effects={
            "dopamine": (0.15, 1.0),
            "norepinephrine": (0.10, 1.0),
            "histamine": (-0.15, 0.0),
        },
        description="Copper is dopamine-beta-hydroxylase cofactor; zinc modulates histamine.",
    ),
}


def list_cofactors() -> List[Dict]:
    """Return cofactor metadata for UI rendering."""
    return [
        {
            "id": key,
            "name": c.name,
            "category": c.category,
            "default_level": c.default_level,
            "min_level": c.min_level,
            "max_level": c.max_level,
            "description": c.description,
        }
        for key, c in COFACTOR_LIBRARY.items()
    ]


def compute_cofactor_targets(cofactor_levels: Dict[str, float]) -> Dict[str, float]:
    """
    Given a map of cofactor id -> level [0,1], compute target offsets for each
    neurochemical. Returns a dict of suggested target adjustments.
    """
    targets: Dict[str, float] = {}
    for key, level in cofactor_levels.items():
        cofactor = COFACTOR_LIBRARY.get(key)
        if not cofactor:
            continue
        # Normalize: 0.5 = baseline, no effect; <0.5 deficit, >0.5 surplus
        deviation = level - cofactor.default_level
        for nt, (slope, _) in cofactor.effects.items():
            targets[nt] = targets.get(nt, 0.0) + deviation * slope
    return targets


def apply_cofactors_to_neurochemistry(
    neurochemistry_state,
    cofactor_levels: Dict[str, float],
    dt: float = 1.0,
) -> None:
    """Apply cofactor-derived targets to a NeurochemicalState object in place."""
    targets = compute_cofactor_targets(cofactor_levels)
    s = neurochemistry_state

    def toward(current: float, target_offset: float, amount: float, clamp: Tuple[float, float] = (0.0, 1.0)):
        target = max(clamp[0], min(clamp[1], current + target_offset))
        diff = target - current
        if abs(diff) <= amount:
            return target
        return current + (amount if diff > 0 else -amount)

    if "serotonin" in targets:
        s.serotonin = toward(s.serotonin, targets["serotonin"], 0.03 * dt)
    if "dopamine" in targets:
        s.dopamine = toward(s.dopamine, targets["dopamine"], 0.03 * dt)
    if "norepinephrine" in targets:
        s.norepinephrine = toward(s.norepinephrine, targets["norepinephrine"], 0.03 * dt)
    if "gaba" in targets:
        s.gaba = toward(s.gaba, targets["gaba"], 0.03 * dt)
    if "glutamate" in targets:
        s.glutamate = toward(s.glutamate, targets["glutamate"], 0.03 * dt)
    if "glycine" in targets:
        s.glycine = toward(s.glycine, targets["glycine"], 0.02 * dt)
    if "bdnf" in targets:
        s.bdnf = toward(s.bdnf, targets["bdnf"], 0.02 * dt)
    if "cytokine_load" in targets:
        s.cytokine_load = toward(s.cytokine_load, targets["cytokine_load"], 0.02 * dt)
    if "histamine" in targets:
        s.histamine = toward(s.histamine, targets["histamine"], 0.02 * dt)
    if "melatonin" in targets:
        s.melatonin = toward(s.melatonin, targets["melatonin"], 0.02 * dt)
    if "adrenaline" in targets:
        s.adrenaline = toward(s.adrenaline, targets["adrenaline"], 0.02 * dt)

    s.clamp()
