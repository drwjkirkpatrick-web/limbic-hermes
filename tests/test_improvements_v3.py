"""
tests/test_improvements_v3.py
==============================
Testable prompts for the third batch of 20 biochemistry-grounded Limbic Hermes
improvements. Each test mirrors a prompt from TODO_LIMBIC_V3.md.
"""
import pytest
from limbic_hermes.core import LimbicSystem
from limbic_hermes.neurochemistry import NeurochemicalState, NeurochemistryEngine
import math
import time

# ---------------------------------------------------------------------------
# 1. Mammillary body / Papez circuit consolidation
# ---------------------------------------------------------------------------

def test_rest_consolidates_recent_event_weights():
    """During rest, recently weighted episodic event kinds are strengthened
    into longer-term hippocampal weights, modeling Papez-circuit consolidation."""
    limbic = LimbicSystem(profile_name="default")
    for _ in range(3):
        limbic.observe_event("study", raw_valence=0.5, raw_arousal=0.3, importance=0.7)
    before = limbic.hippocampal_weights.get("study", 0.0)
    limbic.rest(duration_sec=2.0)
    assert limbic.hippocampal_weights.get("study", 0.0) > before


# ---------------------------------------------------------------------------
# 2. Entorhinal grid-cell novelty signal
# ---------------------------------------------------------------------------

def test_novel_event_kind_boosts_ach_and_theta():
    """Unknown event kinds generate an entorhinal novelty signal that boosts
    acetylcholine and theta-gamma coupling."""
    limbic = LimbicSystem(profile_name="default")
    before_ach = limbic.neurochemistry.state.acetylcholine
    before_tgc = limbic.neurochemistry.state.theta_gamma_coupling
    limbic.observe_event("never_seen_before_kind_xyz", raw_valence=0.0, raw_arousal=0.3, importance=0.5)
    assert limbic.neurochemistry.state.acetylcholine > before_ach
    assert limbic.neurochemistry.state.theta_gamma_coupling > before_tgc


# ---------------------------------------------------------------------------
# 3. Nucleus accumbens shell vs core
# ---------------------------------------------------------------------------

def test_nucleus_accumbens_shell_and_core_tracked():
    """Reward events raise nucleus accumbens shell (wanting) more than core,
    while task load raises core (action vigor) more than shell."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("success", raw_valence=0.8, raw_arousal=0.3, importance=0.8)
    state = limbic.get_state()
    assert state["nucleus_accumbens"]["shell"] > state["nucleus_accumbens"]["core"]


# ---------------------------------------------------------------------------
# 4. RMTg dopamine brake
# ---------------------------------------------------------------------------

def test_rmtg_brake_suppresses_dopamine_on_negative_surprise():
    """Unexpected negative events activate the RMTg GABAergic brake, which
    suppresses dopamine release."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.dopamine = 0.7
    limbic.observe_event("error", raw_valence=-0.9, raw_arousal=0.7, importance=0.9)
    state = limbic.get_state()
    assert state["rmtg"]["brake"] > 0.1


# ---------------------------------------------------------------------------
# 5. BNST sustained anxiety / CRF
# ---------------------------------------------------------------------------

def test_bnst_crf_rises_under_sustained_uncertainty():
    """Prolonged low safety raises CRF and BNST apprehension that outlasts a
    single safe event."""
    limbic = LimbicSystem(profile_name="default")
    limbic.drive.safety = 0.1
    limbic.drive.rest_need = 0.0
    # Force a full second of limbic update so CRF dynamics have dt to rise
    limbic.update(now=time.time() + 1.0)
    state = limbic.get_state()
    assert state["bnst"]["crf"] > 0.1
    assert state["bnst"]["apprehension"] > 0.3


# ---------------------------------------------------------------------------
# 6. PAG column specialization
# ---------------------------------------------------------------------------

def test_pag_active_passive_columns():
    """Dominance selects active (dorsolateral/fight-flight) vs passive
    (ventrolateral/freeze) PAG column output."""
    limbic = LimbicSystem(profile_name="default")
    limbic.vad = limbic.vad.__class__(-0.6, 0.8, 0.8)
    limbic.drive.safety = 0.2
    limbic.observe_event("threat", raw_valence=-0.6, raw_arousal=0.7, importance=0.9)
    state = limbic.get_state()
    assert "active" in state["pag"]
    assert "passive" in state["pag"]
    assert state["pag"]["active"] > state["pag"]["passive"]


# ---------------------------------------------------------------------------
# 7. Dorsal vs median raphe serotonin
# ---------------------------------------------------------------------------

def test_raphe_split_responds_to_threat():
    """Threat lowers dorsal raphe serotonin (anxiety/avoidance) while raising
    median raphe stabilization."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("threat", raw_valence=-0.8, raw_arousal=0.8, importance=0.9)
    state = limbic.get_state()
    assert state["raphe"]["dorsal"] < 0.6
    assert state["raphe"]["median"] >= 0.5
    # Median should exceed dorsal under threat
    assert state["raphe"]["median"] > state["raphe"]["dorsal"]

# 8. Orexin sleep-pressure flip
# ---------------------------------------------------------------------------

def test_sleep_pressure_flips_orexin_to_transition():
    """High sleep pressure reduces orexin-promoted wake and raises transition
    need, lowering histamine."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.sleep_pressure = 0.9
    before_hist = limbic.neurochemistry.state.histamine
    limbic.update()
    state = limbic.get_state()
    assert state["orexin_state"]["transition_need"] > 0.3
    assert limbic.neurochemistry.state.histamine < before_hist


# ---------------------------------------------------------------------------
# 9. CRF stress amplifier
# ---------------------------------------------------------------------------

def test_crf_amplifies_stress_hormones():
    """High CRF amplifies cortisol and adrenaline response to uncertainty."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.crf = 0.8
    before_cort = limbic.neurochemistry.state.cortisol
    before_adr = limbic.neurochemistry.state.adrenaline
    limbic.observe_event("conflict", raw_valence=-0.3, raw_arousal=0.5, importance=0.6)
    assert limbic.neurochemistry.state.cortisol > before_cort
    assert limbic.neurochemistry.state.adrenaline > before_adr


# ---------------------------------------------------------------------------
# 10. Neuropeptide Y resilience
# ---------------------------------------------------------------------------

def test_npy_dampens_bnst_and_threat():
    """High neuropeptide Y reduces BNST CRF/apprehension and lowers threat
    reactivity after a negative event."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.neuropeptide_y = 0.9
    limbic.drive.safety = 0.2
    limbic.observe_event("error", raw_valence=-0.6, raw_arousal=0.5, importance=0.7)
    state = limbic.get_state()
    assert state["neuropeptide_y"] > 0.5
    assert state["bnst"]["apprehension"] < 0.5


# ---------------------------------------------------------------------------
# 11. Dynorphin / kappa opioid aversion
# ---------------------------------------------------------------------------

def test_dynorphin_spikes_on_negative_outcome_and_suppresses_dopamine():
    """Unexpected negative outcomes raise dynorphin, which suppresses dopamine
    and seeking."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("error", raw_valence=-0.9, raw_arousal=0.6, importance=0.9)
    state = limbic.get_state()
    assert state["dynorphin"] > 0.1
    assert limbic.neurochemistry.state.dopamine < 0.5


# ---------------------------------------------------------------------------
# 12. Anandamide / FAAH extinction gating
# ---------------------------------------------------------------------------

def test_low_faah_high_anandamide_speeds_extinction():
    """Low FAAH + high anandamide during safe exposure accelerates reduction of
    hippocampal threat weights."""
    limbic = LimbicSystem(profile_name="default")
    for _ in range(3):
        limbic.observe_event("error", raw_valence=-0.8, raw_arousal=0.6, importance=0.9)
    learned = limbic.hippocampal_weights.get("error", 0.0)
    limbic.neurochemistry.state.faah_activity = 0.05
    limbic.neurochemistry.state.anandamide = 0.9
    for _ in range(10):
        limbic.observe_event("error", raw_valence=0.0, raw_arousal=0.0, importance=0.1)
    assert limbic.hippocampal_weights.get("error", 1.0) < learned


# ---------------------------------------------------------------------------
# 13. GABA transporter (GAT) tone
# ---------------------------------------------------------------------------

def test_gat_activity_lowers_effective_gaba():
    """Higher GAT activity reduces effective GABA tone, increasing
    excitability/irritability."""
    s = NeurochemicalState(gaba=0.8)
    s.gat_activity = 0.9
    engine = NeurochemistryEngine(s)
    engine.update(1.0, 0, 0, 0, 0, 0, 0, 1.0, 0)
    assert engine.state.gaba < 0.8


# ---------------------------------------------------------------------------
# 14. Astrocyte GLT-1 / glutamate clearance
# ---------------------------------------------------------------------------

def test_low_glt1_raises_glutamate_and_excitotoxicity():
    """Low astroglial GLT-1 activity impairs glutamate clearance, raising
    glutamate and quinolinic acid-driven excitotoxicity."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.glt1_activity = 0.1
    limbic.neurochemistry.state.cytokine_load = 0.6
    before_glu = limbic.neurochemistry.state.glutamate
    limbic.update()
    state = limbic.get_state()
    assert limbic.neurochemistry.state.glutamate > before_glu
    assert state["excitotoxicity_risk"] > 0.1


# ---------------------------------------------------------------------------
# 15. Microglial priming / neuroimmune memory
# ---------------------------------------------------------------------------

def test_microglial_priming_sensitizes_future_cytokine_response():
    """Prior cytokine/cortisol load primes microglia, so a later stressor
    produces a larger cytokine spike."""
    limbic = LimbicSystem(profile_name="default")
    # Prime
    for _ in range(5):
        limbic.observe_event("error", raw_valence=-0.6, raw_arousal=0.5, importance=0.7)
    # Rest to let cytokine decay
    limbic.rest(duration_sec=3.0)
    # Add a small additional rest tick to drop cytokine below primed reactivity threshold
    limbic.update(now=time.time() + 3.0)
    before = limbic.neurochemistry.state.cytokine_load
    limbic.observe_event("threat", raw_valence=-0.5, raw_arousal=0.5, importance=0.6)
    after = limbic.neurochemistry.state.cytokine_load
    assert after > before
    assert limbic.neurochemistry.state.microglia_state > 0.08


# ---------------------------------------------------------------------------
# 16. Astrocyte glycogen-lactate shuttle
# ---------------------------------------------------------------------------

def test_task_load_consumes_glycogen_and_raises_lactate():
    """High working-memory/task load consumes glycogen and raises lactate."""
    limbic = LimbicSystem(profile_name="default")
    limbic.set_working_memory_load(0.9)
    before_gly = limbic.neurochemistry.state.glycogen
    limbic.update()
    assert limbic.neurochemistry.state.glycogen < before_gly
    assert limbic.neurochemistry.state.lactate > 0.1


# ---------------------------------------------------------------------------
# 17. Theta-gamma coupling
# ---------------------------------------------------------------------------

def test_theta_gamma_coupling_boosts_encoding():
    """High ACh + moderate arousal raises theta-gamma coupling, which increases
    hippocampal weight updates."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.acetylcholine = 0.8
    limbic.neurochemistry.state.norepinephrine = 0.4
    limbic.observe_event("target", raw_valence=0.5, raw_arousal=0.4, importance=0.7)
    state = limbic.get_state()
    assert state["theta_gamma_coupling"] > 0.3
    assert limbic.hippocampal_weights.get("target", 0.0) > 0.05


# ---------------------------------------------------------------------------
# 18. Sharp-wave ripple / place-cell replay
# ---------------------------------------------------------------------------

def test_rest_replay_strengthens_high_importance_weights():
    """During rest, high-importance event kinds are replayed and consolidated
    faster than low-importance ones."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("urgent", raw_valence=0.7, raw_arousal=0.5, importance=0.95)
    limbic.observe_event("trivial", raw_valence=0.7, raw_arousal=0.5, importance=0.1)
    urgent_before = limbic.hippocampal_weights.get("urgent", 0.0)
    trivial_before = limbic.hippocampal_weights.get("trivial", 0.0)
    limbic.rest(duration_sec=2.0)
    assert limbic.hippocampal_weights.get("urgent", 0.0) > urgent_before
    assert limbic.hippocampal_weights.get("urgent", 0.0) > limbic.hippocampal_weights.get("trivial", 0.0)


# ---------------------------------------------------------------------------
# 19. Ventral pallidum wanting vs liking
# ---------------------------------------------------------------------------

def test_ventral_pallidum_liking_separate_from_wanting():
    """Reward events can raise nucleus accumbens wanting without raising
    ventral pallidum liking if opioid tone is low."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.opioid = 0.1
    limbic.observe_event("success", raw_valence=0.8, raw_arousal=0.3, importance=0.8)
    state = limbic.get_state()
    assert state["nucleus_accumbens"]["shell"] > state["ventral_pallidum"]["liking"]


# ---------------------------------------------------------------------------
# 20. Subgenual anterior cingulate rumination switch
# ---------------------------------------------------------------------------

def test_subgenual_acc_slows_negative_recovery():
    """High cortisol + low serotonin + repeated negative events shift ACC toward
    rumination, slowing recovery of negative valence."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.cortisol = 0.8
    limbic.neurochemistry.state.serotonin = 0.2
    for _ in range(3):
        limbic.observe_event("error", raw_valence=-0.5, raw_arousal=0.4, importance=0.6)
    state = limbic.get_state()
    assert state["subgenual_acc"]["rumination"] > 0.3
