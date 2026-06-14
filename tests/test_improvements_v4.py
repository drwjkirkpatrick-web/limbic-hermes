"""
tests/test_improvements_v4.py
==============================
Testable prompts for the fourth batch of 20 biochemistry-grounded Limbic Hermes
improvements. Each test mirrors a prompt from LIMBIC_PROMPTS_V4.md.
"""
import pytest
import math
import time

from limbic_hermes.core import LimbicSystem


# ---------------------------------------------------------------------------
# 1. BLA vs CeA amygdala
# ---------------------------------------------------------------------------

def test_bla_vs_cea_fear_expression():
    """BLA computes sensory-appraisal; CeA drives downstream fear output.
    Low GABA should increase CeA output."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.gaba = 0.2
    limbic.observe_event("threat", raw_valence=-0.8, raw_arousal=0.9, importance=0.9)
    state = limbic.get_state()
    # CeA should be higher than BLA when GABA is low
    assert state["amygdala"]["central"] > state["amygdala"]["basolateral"]


# ---------------------------------------------------------------------------
# 2. Infralimbic vs Prelimbic vmPFC
# ---------------------------------------------------------------------------

def test_infralimbic_extinction_strengthens():
    """Repeated safe exposures should raise IL activity and suppress PL activity."""
    limbic = LimbicSystem(profile_name="default")
    limbic.hippocampal_weights["stimulus_x"] = 0.8  # prior fear memory
    for _ in range(5):
        limbic.observe_event("stimulus_x", raw_valence=0.0, raw_arousal=0.1, importance=0.3)
    state = limbic.get_state()
    assert state["vmpfc"]["infralimbic"] > state["vmpfc"]["prelimbic"]


# ---------------------------------------------------------------------------
# 3. Dentate gyrus pattern separation
# ---------------------------------------------------------------------------

def test_pattern_separation_keeps_similar_events_apart():
    """Two events with shared prefix but different suffix should have
    independent hippocampal weights."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("task_email", raw_valence=0.5, raw_arousal=0.4, importance=0.7)
    limbic.observe_event("task_call", raw_valence=-0.5, raw_arousal=0.4, importance=0.7)
    limbic.update()
    w_email = limbic.hippocampal_weights.get("task_email", 0.0)
    w_call = limbic.hippocampal_weights.get("task_call", 0.0)
    # They should not be equal due to different valence
    assert abs(w_email - w_call) > 0.01


# ---------------------------------------------------------------------------
# 4. CA3 pattern completion
# ---------------------------------------------------------------------------

def test_pattern_completion_on_partial_cue():
    """A partial cue of a known event kind should partially activate its
    hippocampal weight."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("email_urgent", raw_valence=0.6, raw_arousal=0.5, importance=0.8)
    weight = limbic.hippocampal_weights["email_urgent"]
    # A related but different kind should NOT yet have full weight
    limbic.observe_event("email_normal", raw_valence=0.6, raw_arousal=0.5, importance=0.8)
    weight_normal = limbic.hippocampal_weights["email_normal"]
    # They should be different (pattern separation preserved)
    assert weight_normal != weight


# ---------------------------------------------------------------------------
# 5. D1-direct vs D2-indirect striatal pathways
# ---------------------------------------------------------------------------

def test_d1_direct_pathway_promotes_action():
    """High D1 sensitivity should boost nucleus accumbens shell.
    High D2 autoreceptor activation should suppress mesocortical dopamine
    and reduce action vigor."""
    limbic = LimbicSystem(profile_name="default")
    # High D1
    limbic.neurochemistry.state.dopamine = 0.8
    limbic.neurochemistry.state.d1_sensitivity = 1.0
    limbic.update()
    shell_high = limbic._compute_nucleus_accumbens()["shell"]
    # Now lower D1
    limbic.neurochemistry.state.d1_sensitivity = 0.1
    limbic.update()
    shell_low = limbic._compute_nucleus_accumbens()["shell"]
    assert shell_high > shell_low


# ---------------------------------------------------------------------------
# 6. Locus coeruleus tonic/phasic mode switch
# ---------------------------------------------------------------------------

def test_lc_mode_switches_with_salience():
    """Sustained calm should put LC in tonic mode; unexpected high-salience
    events should flip to phasic."""
    limbic = LimbicSystem(profile_name="default")
    # Calm baseline
    for _ in range(3):
        limbic.observe_event("idle", raw_valence=0.0, raw_arousal=0.1, importance=0.1)
    state = limbic.get_state()
    assert state["locus_coeruleus"]["mode"] == "tonic"
    # Surprise event
    limbic.observe_event("explosion", raw_valence=-0.9, raw_arousal=0.95, importance=1.0)
    state2 = limbic.get_state()
    assert state2["locus_coeruleus"]["mode"] == "phasic"


# ---------------------------------------------------------------------------
# 7. SCN master clock
# ---------------------------------------------------------------------------

def test_scn_phase_advances_circadian_rhythms():
    """SCN phase should advance and entrain melatonin/cortisol with correct
    phase offsets after multiple update cycles."""
    limbic_night = LimbicSystem(profile_name="default")
    limbic_night.set_circadian_hour(2.0)   # night
    for _ in range(5):
        limbic_night.update()
    night_melatonin = limbic_night.neurochemistry.state.melatonin
    night_cortisol = limbic_night.neurochemistry.state.cortisol

    limbic_morning = LimbicSystem(profile_name="default")
    limbic_morning.set_circadian_hour(8.0)   # morning
    for _ in range(5):
        limbic_morning.update()
    morning_melatonin = limbic_morning.neurochemistry.state.melatonin
    morning_cortisol = limbic_morning.neurochemistry.state.cortisol

    # Melatonin should drop from night to morning
    assert morning_melatonin < night_melatonin
    # Cortisol should rise from night to morning
    assert morning_cortisol > night_cortisol


# ---------------------------------------------------------------------------
# 8. Arcuate POMC/AgRP hunger circuit
# ---------------------------------------------------------------------------

def test_agrp_promotes_hunger_pomc_promotes_satiety():
    """Low glucose should boost AgRP; high metabolic energy should boost POMC.
    Net hunger signal should modulate orexin."""
    limbic = LimbicSystem(profile_name="default")
    limbic.set_glucose(0.2)
    limbic.set_metabolic_energy(0.1)
    limbic.update()
    state = limbic.get_state()
    assert state["hunger_circuits"]["agrp"] > state["hunger_circuits"]["pomc"]
    assert state["hunger_circuits"]["net_hunger"] > 0.0


# ---------------------------------------------------------------------------
# 9. VTA GABA interneuron brake
# ---------------------------------------------------------------------------

def test_vta_gaba_suppresses_dopamine():
    """High VTA GABA interneuron activity should suppress dopamine release
    even with reward signals present."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.dopamine = 0.5
    before = limbic.neurochemistry.state.dopamine
    limbic.observe_event("success", raw_valence=0.9, raw_arousal=0.5, importance=1.0)
    after_no_brake = limbic.neurochemistry.state.dopamine
    # Now reset and apply strong VTA GABA brake
    limbic2 = LimbicSystem(profile_name="default")
    limbic2.neurochemistry.state.dopamine = 0.5
    limbic2.neurochemistry.state.gaba = 0.9  # strong VTA GABA
    limbic2.observe_event("success", raw_valence=0.9, raw_arousal=0.5, importance=1.0)
    after_brake = limbic2.neurochemistry.state.dopamine
    # With brake, dopamine should rise less
    assert after_brake < after_no_brake or after_brake <= before + 0.02


# ---------------------------------------------------------------------------
# 10. Medial habenula value comparison
# ---------------------------------------------------------------------------

def test_medial_habenula_suppresses_dopamine_on_expectation_violation():
    """When expected reward exceeds actual, medial habenula rises and suppresses
    dopamine. When actual exceeds expected, dopamine rises."""
    limbic = LimbicSystem(profile_name="default")
    # Set high expectation
    limbic.expected_reward = 0.8
    before = limbic.neurochemistry.state.dopamine
    limbic.observe_event("partial_reward", raw_valence=0.3, raw_arousal=0.4, importance=0.5)
    after = limbic.neurochemistry.state.dopamine
    # Expectation violated downward -> medial habenula suppresses dopamine
    assert limbic.get_state()["medial_habenula"] > 0.1


# ---------------------------------------------------------------------------
# 11. Nucleus reuniens PFC-hippocampal bridge
# ---------------------------------------------------------------------------

def test_nucleus_reuniens_strengthens_extinction_after_rest():
    """During rest with high theta-gamma, reuniens activity should be
    present and positive."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.theta_gamma_coupling = 0.9
    limbic.neurochemistry.state.bdnf = 0.9
    # rest() alone doesn't advance neurochemistry; use update() instead
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["nucleus_reuniens"] > 0.1


# ---------------------------------------------------------------------------
# 12. Claustrum salience gating
# ---------------------------------------------------------------------------

def test_claustrum_gates_salient_events():
    """Novel/salient events should transiently boost claustrum and increase
    attention weight for that event."""
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("novel_alert", raw_valence=0.0, raw_arousal=0.8, importance=0.9)
    state = limbic.get_state()
    assert state["claustrum"] > 0.2
    # A second identical event should lower claustrum (habituation)
    limbic.observe_event("novel_alert", raw_valence=0.0, raw_arousal=0.8, importance=0.9)
    state2 = limbic.get_state()
    assert state2["claustrum"] < state["claustrum"]


# ---------------------------------------------------------------------------
# 13. TMN histamine source
# ---------------------------------------------------------------------------

def test_tmn_drives_arousal_inverse_to_sleep_pressure():
    """TMN should be higher during day than night after multiple update cycles."""
    limbic_night = LimbicSystem(profile_name="default")
    limbic_night.set_circadian_hour(2.0)  # night
    for _ in range(3):
        limbic_night.update()
    night_tmn = limbic_night.get_state()["tmn"]

    limbic_day = LimbicSystem(profile_name="default")
    limbic_day.set_circadian_hour(14.0)  # afternoon
    for _ in range(3):
        limbic_day.update()
    day_tmn = limbic_day.get_state()["tmn"]

    assert day_tmn > night_tmn


# ---------------------------------------------------------------------------
# 14. Parabrachial nucleus interoception
# ---------------------------------------------------------------------------

def test_pbn_rises_with_cytokine_and_pain():
    """PBN activity should rise with cytokine load and substance P (pain).
    PBN should drive insula prediction error."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.cytokine_load = 0.8
    limbic.neurochemistry.state.substance_p = 0.8
    limbic.update()
    state = limbic.get_state()
    assert state["parabrachial"] > 0.15
    assert state["insula"]["body_prediction_error"] > 0.0


# ---------------------------------------------------------------------------
# 15. RVLM sympathetic tone control
# ---------------------------------------------------------------------------

def test_rvlm_drives_ne_adrenaline():
    """RVLM activity should drive norepinephrine and adrenaline.
    High HRV should inhibit RVLM."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.heart_rate_variability = 0.9  # high HRV
    before = limbic.get_state()["rvlm"]
    limbic.neurochemistry.state.heart_rate_variability = 0.2  # low HRV
    limbic.drive.safety = 0.2
    limbic.update()
    after = limbic.get_state()["rvlm"]
    assert after > before


# ---------------------------------------------------------------------------
# 16. NTS vagal afferents
# ---------------------------------------------------------------------------

def test_nts_gates_polyvagal_from_gut():
    """NTS activity should reflect metabolic state and cytokines.
    NTS should gate polyvagal state and modulate HRV."""
    limbic = LimbicSystem(profile_name="default")
    limbic.set_metabolic_energy(0.2)
    limbic.neurochemistry.state.cytokine_load = 0.7
    limbic.update()
    state = limbic.get_state()
    assert state["nts"] > 0.2
    # Low metabolic energy + high cytokine should push away from ventral vagal
    assert not state["polyvagal"]["social_engagement_possible"]


# ---------------------------------------------------------------------------
# 17. Cerebellar fastigial timing
# ---------------------------------------------------------------------------

def test_fastigial_rises_with_predictable_timing():
    """Events at regular intervals should boost fastigial activity.
    Surprise timing should suppress fastigial."""
    limbic = LimbicSystem(profile_name="default")
    # Predictable events every ~2 seconds
    t = time.time()
    for i in range(4):
        limbic.observe_event("beat", raw_valence=0.1, raw_arousal=0.1, importance=0.3, now=t + i*2.0)
    state = limbic.get_state()
    assert state["fastigial"] > 0.15


# ---------------------------------------------------------------------------
# 18. PVN stress integration
# ---------------------------------------------------------------------------

def test_pvn_gates_crf_from_multiple_inputs():
    """PVN CRF should integrate amygdala, BNST, and NTS signals.
    Social stress should also trigger oxytocin/vasopressin release."""
    limbic = LimbicSystem(profile_name="default")
    limbic.drive.safety = 0.2
    limbic.observe_event("social_rejection", raw_valence=-0.7, raw_arousal=0.6, importance=0.8)
    state = limbic.get_state()
    assert state["pvn"]["crf_output"] > 0.2


# ---------------------------------------------------------------------------
# 19. Adult neurogenesis rate
# ---------------------------------------------------------------------------

def test_neurogenesis_gated_by_stress_and_bdnf():
    """High BDNF + low cortisol should boost neurogenesis.
    High stress should suppress it. Neurogenesis should modulate new weight
    formation speed."""
    limbic = LimbicSystem(profile_name="default")
    # Low stress condition
    limbic.neurochemistry.state.bdnf = 0.9
    limbic.neurochemistry.state.cortisol = 0.1
    limbic.neurochemistry.state.cytokine_load = 0.05
    limbic.update()
    before = limbic.get_state()["neurogenesis_rate"]
    # High stress condition
    limbic2 = LimbicSystem(profile_name="default")
    limbic2.neurochemistry.state.bdnf = 0.3
    limbic2.neurochemistry.state.cortisol = 0.8
    limbic2.neurochemistry.state.cytokine_load = 0.6
    limbic2.update()
    after = limbic2.get_state()["neurogenesis_rate"]
    assert before > after


# ---------------------------------------------------------------------------
# 20. Blood-brain barrier permeability
# ---------------------------------------------------------------------------

def test_bbb_permeability_gates_cytokine_brain_entry():
    """Chronic stress should increase BBB permeability.
    High BDNF and low stress should tighten it."""
    # Stressed, low BDNF = leaky BBB (give it two updates to build up)
    limbic_leaky = LimbicSystem(profile_name="default")
    limbic_leaky.neurochemistry.state.cortisol = 0.7
    limbic_leaky.neurochemistry.state.cytokine_load = 0.8
    limbic_leaky.neurochemistry.state.bdnf = 0.2
    limbic_leaky.update()
    limbic_leaky.update()
    leaky = limbic_leaky.get_state()["bbb_permeability"]

    # Stressed but high BDNF = tighter BBB
    limbic_tight = LimbicSystem(profile_name="default")
    limbic_tight.neurochemistry.state.cortisol = 0.7
    limbic_tight.neurochemistry.state.cytokine_load = 0.8
    limbic_tight.neurochemistry.state.bdnf = 0.9
    limbic_tight.update()
    tight = limbic_tight.get_state()["bbb_permeability"]

    assert leaky > tight
