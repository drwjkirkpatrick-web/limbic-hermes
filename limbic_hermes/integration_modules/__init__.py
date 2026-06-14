"""
limbic_hermes/integration_modules/__init__.py
===============================================

Circuit-level integration modules for limbic-hermes.
Each module ties together multiple neurochemical signals into
functional circuits with physiologically-grounded interactions.
"""

from .stress_immune_fatigue import compute_stress_immune_fatigue
from .reward_extinction import compute_reward_extinction_state
from .sleep_homeostasis import compute_sleep_homeostasis
from .social_affiliation import compute_social_affiliation
from .fear_memory_consolidation import compute_fear_memory_state
from .attention_salience import compute_attention_salience
from .metabolic_energy_allocation import compute_metabolic_allocation
from .pain_modulation import compute_pain_modulation
from .hpa_feedback import compute_hpa_feedback
from .circadian_metabolic_coupling import compute_circadian_metabolic
from .dmn_salience_switching import compute_dmn_salience_switch
from .neuroplasticity_resilience import compute_resilience_state
from .gut_brain_stress import compute_gut_brain_stress
from .hormonal_mood import compute_hormonal_mood
from .excitotoxicity_protection import compute_excitotoxicity_protection
from .prepulse_gating import compute_prepulse_gating
from .thermogenesis_arousal import compute_thermogenesis_arousal
from .dopamine_balancing import compute_dopamine_balance
from .theta_memory_encoding import compute_theta_memory_state
from .allostatic_recovery import compute_allostatic_recovery

__all__ = [
    "compute_stress_immune_fatigue",
    "compute_reward_extinction_state",
    "compute_sleep_homeostasis",
    "compute_social_affiliation",
    "compute_fear_memory_state",
    "compute_attention_salience",
    "compute_metabolic_allocation",
    "compute_pain_modulation",
    "compute_hpa_feedback",
    "compute_circadian_metabolic",
    "compute_dmn_salience_switch",
    "compute_resilience_state",
    "compute_gut_brain_stress",
    "compute_hormonal_mood",
    "compute_excitotoxicity_protection",
    "compute_prepulse_gating",
    "compute_thermogenesis_arousal",
    "compute_dopamine_balance",
    "compute_theta_memory_state",
    "compute_allostatic_recovery",
]
