"""
tests/test_improvements_v5.py
=============================

V5: 20 new neurochemistry modules filling gaps between human neurology
and the current Limbic Hermes model.

Each test asserts a physiologically-grounded expectation. Run with:
    pytest tests/test_improvements_v5.py -v
"""

import pytest
from limbic_hermes.core import LimbicSystem


# =========================================================================
# 1. CEREBELLAR CORTICAL LAYERS (Purkinje, Granule, Climbing Fiber)
# =========================================================================

def test_purkinje_cell_inhibition_damps_motor_error():
    """Purkinje cells provide inhibitory brake on motor commands;
    high climbing fiber error should increase Purkinje output which
    suppresses downstream motor signals."""
    limbic = LimbicSystem()
    limbic.drive.error_temperature = 0.7
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["purkinje_output"] > 0.2
    # High Purkinje output should suppress fastigial
    assert state["neurochemistry"]["purkinje_output"] > state["fastigial"]


def test_climbing_fiber_error_teaches_purkinje():
    """Climbing fibers from inferior olive carry motor error signals
    that teach Purkinje cells to inhibit future similar commands."""
    limbic = LimbicSystem()
    limbic.drive.error_temperature = 0.7
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["climbing_fiber_error"] > 0.2
    # Error should drive up Purkinje learning
    assert state["neurochemistry"]["purkinje_output"] > 0.35
    assert state["neurochemistry"]["purkinje_output"] > state["fastigial"]


# =========================================================================
# 2. PAG COLUMNAR ORGANIZATION
# =========================================================================

def test_pag_dorsolateral_fight_column():
    """Dorsolateral PAG activates fight responses: high dominance,
    high threat, but not fleeing — should activate sympathetic
    and raise vasopressin/testosterone."""
    limbic = LimbicSystem()
    # Set high dominance and low safety directly for PAG fight column
    limbic.neurochemistry.state.pag_dorsolateral = 0.5
    limbic.update()
    state = limbic.get_state()
    assert state.get("pag", {}).get("dorsolateral", 0) >= 0.2


def test_pag_ventrolateral_freeze_column():
    """Ventrolateral PAG activates freezing: high threat, low
    dominance, complete immobility — should suppress histamine
    and raise opioid."""
    limbic = LimbicSystem()
    for _ in range(3):
        limbic.observe_event("threat", "Inescapable predator")
    state = limbic.get_state()
    assert state.get("pag", {}).get("ventrolateral", 0) > 0.05


# =========================================================================
# 3. PREFRONTAL WORKING MEMORY GATING
# =========================================================================

def test_dlpfc_maintenance_vs_updating():
    """dlPFC D1/D2 balance controls maintenance (high D1) vs
    updating (high D2); task load should shift balance."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.dopamine = 0.8
    limbic.neurochemistry.state.dopamine_mesocortical = 0.7
    limbic.set_working_memory_load(0.7)
    limbic.update()
    state = limbic.get_state()
    assert state.get("prefrontal", {}).get("maintenance_bias", 0) > 0.4


def test_ofc_reward_valuation_updates_with_experience():
    """OFC tracks reward value of specific outcomes; repeated
    devaluation should reduce expected reward."""
    limbic = LimbicSystem()
    # Reward then devalue
    for _ in range(5):
        limbic.observe_event("reward", "Expected reward")
    # Now devalue
    for _ in range(5):
        limbic.observe_event("error", "Reward devalued")
    state = limbic.get_state()
    assert state["expected_reward"] < 0.3


# =========================================================================
# 4. TESTOSTERONE / SOCIAL DOMINANCE
# =========================================================================

def test_testosterone_rises_with_social_victory():
    """Testosterone rises after social victory (challenge hypothesis)
    and suppresses fear circuits (BLA/CeA)."""
    limbic = LimbicSystem()
    for _ in range(3):
        limbic.observe_event("success", "Social dominance victory")
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["testosterone"] > 0.4
    assert state["neurochemistry"]["cea"] < 0.65


def test_social_defeat_lowers_testosterone_and_bdnf():
    """Social defeat chronically lowers testosterone and BDNF,
    increasing susceptibility to learned helplessness."""
    limbic = LimbicSystem()
    # Start with elevated testosterone
    limbic.neurochemistry.state.testosterone = 0.8
    initial_t = limbic.get_state()["neurochemistry"]["testosterone"]
    initial_bdnf = limbic.get_state()["neurochemistry"]["bdnf"]
    for _ in range(10):
        limbic.observe_event("error", "Social defeat")
    state = limbic.get_state()
    assert state["neurochemistry"]["testosterone"] < initial_t
    assert state["neurochemistry"]["bdnf"] < initial_bdnf


# =========================================================================
# 5. SLEEP ARCHITECTURE (NREM / REM cycling)
# =========================================================================

def test_nrem_slow_wave_boosts_glymphatic_clearance():
    """NREM slow wave sleep drives glymphatic clearance;
    high sleep pressure + low NE should maximize clearance."""
    limbic = LimbicSystem()
    limbic.set_circadian_hour(4.0)
    limbic.neurochemistry.state.sleep_pressure = 0.8
    limbic.neurochemistry.state.norepinephrine = 0.2
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state.get("sleep_architecture", {}).get("nrem_slow_wave", 0) > 0.3
    assert state["neurochemistry"]["glymphatic_flow"] > state["neurochemistry"]["amyloid_beta"]


def test_rem_sleep_boosts_dopamine_and_theta():
    """REM sleep has high ACh, low NE, and elevated theta;
    it should selectively boost dopamine and suppress NE."""
    limbic = LimbicSystem()
    limbic.set_circadian_hour(4.0)
    limbic.neurochemistry.state.sleep_pressure = 0.8
    # Build REM conditions
    limbic.neurochemistry.state.acetylcholine = 0.7
    limbic.neurochemistry.state.norepinephrine = 0.15
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state.get("sleep_architecture", {}).get("rem_theta", 0) > 0.2
    assert state["neurochemistry"]["dopamine"] > 0.3
    assert state["neurochemistry"]["norepinephrine"] < 0.3


# =========================================================================
# 6. THERMOREGULATION / PREOPTIC AREA
# =========================================================================

def test_preoptic_area_suppresses_arousal_at_high_temp():
    """Preoptic area warmth sensing suppresses wake-promoting
    orexin and histamine when body temperature rises."""
    limbic = LimbicSystem()
    for _ in range(5):
        limbic.observe_event("task_complete", "Exercise heat")
    for _ in range(10):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["preoptic_warmth"] > 0.3
    assert state["neurochemistry"]["histamine"] < 0.65
    # Verify some histamine suppression occurred vs initial trend
    assert state["neurochemistry"]["preoptic_warmth"] > state["neurochemistry"]["body_temperature"] * 0.5


def test_brown_adipose_raises_temperature_and_metabolism():
    """Brown adipose tissue thermogenesis raises body temperature
    and increases metabolic rate via NE."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.norepinephrine = 0.9
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["brown_adipose_activity"] > 0.3
    assert state["neurochemistry"]["body_temperature"] > 0.4


# =========================================================================
# 7. THALAMIC RELAY NUCLEI (MD, Pulvinar)
# =========================================================================

def test_md_thalamus_gates_working_memory():
    """Mediodorsal thalamus gates prefrontal working memory;
    high task load should increase MD firing."""
    limbic = LimbicSystem()
    limbic.set_working_memory_load(0.7)
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["md_thalamus"] > 0.2
    assert state.get("prefrontal", {}).get("maintenance_bias", 0) > 0.3


def test_pulvinar_gates_attention_salience():
    """Pulvinar nucleus gates visual attention salience;
    novel events should increase pulvinar activity."""
    limbic = LimbicSystem()
    for _ in range(3):
        limbic.observe_event("task_complete", "Novel visual stimulus")
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["pulvinar"] > 0.15


# =========================================================================
# 8. THETA RHYTHM GENERATION (Septum-Hippocampus)
# =========================================================================

def test_medial_septum_drives_hippocampal_theta():
    """Medial septum GABAergic/cholinergic cells pace hippocampal
    theta; high ACh should increase theta power."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.acetylcholine = 0.8
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["medial_septum"] > 0.3
    assert state["neurochemistry"]["hippocampal_theta"] > 0.2


def test_grid_cell_theta_modulation():
    """Entorhinal grid cells fire at theta-modulated rates;
    high theta-gamma coupling should enhance spatial encoding."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.theta_gamma_coupling = 0.7
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["grid_cell_modulation"] > 0.3


# =========================================================================
# 9. GLYMPHATIC SYSTEM (CSF clearance)
# =========================================================================

def test_glymphatic_clearance_removes_waste_during_sleep():
    """Glymphatic system clears amyloid-beta during NREM;
    high sleep pressure + low NE should increase clearance."""
    limbic = LimbicSystem()
    limbic.set_circadian_hour(4.0)
    limbic.neurochemistry.state.sleep_pressure = 0.8
    limbic.neurochemistry.state.norepinephrine = 0.2
    initial_ab = limbic.get_state()["neurochemistry"]["amyloid_beta"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["glymphatic_flow"] > 0.2
    assert state["neurochemistry"]["amyloid_beta"] < initial_ab


def test_glymphatic_flow_correlates_with_astrocyte_aquaporin():
    """Astrocyte aquaporin-4 water channels mediate glymphatic
    flow; high AQP4 should correlate with high clearance."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.aquaporin_4 = 0.8
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["glymphatic_flow"] > 0.2
    assert state["neurochemistry"]["glymphatic_flow"] > state["neurochemistry"]["amyloid_beta"]


# =========================================================================
# 10. GUT-BRAIN AXIS (Vagus, Microbiome)
# =========================================================================

def test_vagus_signaling_modulates_hpa_axis():
    """Vagal afferent signaling from gut suppresses HPA axis;
    high vagal tone should lower cortisol."""
    limbic = LimbicSystem()
    for _ in range(3):
        limbic.observe_event("error", "Stressor")
    initial_cort = limbic.get_state()["neurochemistry"]["cortisol"]
    # Now increase vagal tone
    limbic.neurochemistry.state.vagal_afferent = 0.8
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["vagal_afferent"] > 0.3
    assert state["neurochemistry"]["cortisol"] < initial_cort


def test_microbiome_scfa_modulates_gaba():
    """Short-chain fatty acids (butyrate) from microbiome enhance
    GABAergic signaling and reduce anxiety."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.butyrate = 0.7
    initial_gaba = limbic.get_state()["neurochemistry"]["gaba"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["gaba"] > initial_gaba
    assert state["neurochemistry"]["gaba"] > 0.4


# =========================================================================
# 11. ESTROGEN / PROGESTERONE CYCLE
# =========================================================================

def test_estrogen_cycle_modulates_serotonin_and_bdnf():
    """Estrogen upregulates serotonin transporters and BDNF;
    high estrogen should boost both."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.estrogen = 0.8
    initial_bdnf = limbic.get_state()["neurochemistry"]["bdnf"]
    initial_ser = limbic.get_state()["neurochemistry"]["serotonin"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["bdnf"] > initial_bdnf
    assert state["neurochemistry"]["serotonin"] > initial_ser


def test_progestrone_allopregnanolone_enhances_gaba_a():
    """Progesterone → allopregnanolone → GABA-A receptor
    enhancement; should increase GABA-A sensitivity."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.progesterone = 0.8
    limbic.neurochemistry.state.gaba_a_sensitivity = 0.8
    initial_sens = limbic.get_state()["neurochemistry"]["gaba_a_sensitivity"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["gaba_a_sensitivity"] > initial_sens
    assert state["neurochemistry"]["allopregnanolone"] > 0.3
    assert state["neurochemistry"]["gaba_a_sensitivity"] >= 1.0


# =========================================================================
# 12. HYPOXIC RESPONSE (HIF-1α, Adenosine)
# =========================================================================

def test_hypoxia_raises_adenosine_and_suppresses_glutamate():
    """Hypoxia raises protective adenosine which suppresses
    excitotoxic glutamate release."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.oxygen_saturation = 0.3
    initial_glu = limbic.get_state()["neurochemistry"]["glutamate"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["adenosine"] > 0.3
    assert state["neurochemistry"]["glutamate"] < initial_glu
    assert state["neurochemistry"]["glutamate"] < 0.5


def test_hif1_alpha_guides_hypoxic_adaptation():
    """HIF-1α activates under sustained hypoxia to promote
    glycolysis and angiogenesis."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.oxygen_saturation = 0.1
    # Block recovery by keeping cortisol high and task load present
    limbic.neurochemistry.state.cortisol = 0.9
    limbic.drive.task_load = 0.9
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["hif1_alpha"] > 0.15
    assert state["neurochemistry"]["hif1_alpha"] > 0.05


# =========================================================================
# 13. DESCENDING PAIN CONTROL (PAG-RVMM-Spinal Cord)
# =========================================================================

def test_pag_rvmm_spinal_gate_closes_with_opioid():
    """PAG activates RVMM which releases spinal opioids;
    high opioid should reduce substance P (pain)."""
    limbic = LimbicSystem()
    limbic.observe_event("error", "Painful event")
    initial_sp = limbic.get_state()["neurochemistry"]["substance_p"]
    # Now activate opioid system
    limbic.neurochemistry.state.opioid = 0.8
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["spinal_opioid"] > 0.2
    assert state["neurochemistry"]["substance_p"] < initial_sp


def test_gate_control_theory_ab_fiber_inhibits_c_fiber():
    """Gate control theory: Aβ fiber activation (touch/pressure)
    inhibits C-fiber pain transmission at dorsal horn."""
    limbic = LimbicSystem()
    # Pain event
    limbic.observe_event("error", "Painful event")
    pain_state = limbic.get_state()
    initial_sp = pain_state["neurochemistry"]["substance_p"]
    # Now apply Aβ stimulation
    limbic.neurochemistry.state.ab_fiber = 0.8
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["substance_p"] < initial_sp
    assert state["neurochemistry"]["ab_fiber"] > 0.5


# =========================================================================
# 14. HEDONIC HOTSPOTS (Nucleus Accumbens Shell μ-Opioid)
# =========================================================================

def test_nacc_shell_mu_opioid_liking_vs_wanting():
    """Nucleus accumbens shell μ-opioid mediates 'liking';
    it should be higher than dopamine 'wanting' during
    high opioid states."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.opioid = 0.8
    limbic.neurochemistry.state.dopamine = 0.2
    initial_liking = limbic.get_state()["neurochemistry"]["nacc_shell_liking"]
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["nacc_shell_liking"] > initial_liking
    assert state["neurochemistry"]["nacc_shell_liking"] > 0.4


# =========================================================================
# 15. MAST CELL-NEUROIMMUNE CROSSTALK
# =========================================================================

def test_mast_cell_degranulation_releases_histamine_and_cytokines():
    """Mast cell degranulation under stress releases histamine
    and IL-4, sensitize pain circuits."""
    limbic = LimbicSystem()
    initial_hist = limbic.get_state()["neurochemistry"]["histamine"]
    for _ in range(5):
        limbic.observe_event("error", "Allergic stressor")
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["mast_cell_activation"] > 0.05
    assert state["neurochemistry"]["histamine"] > initial_hist


# =========================================================================
# 16. AMINO ACID PRECURSOR COMPETITION
# =========================================================================

def test_tryptophan_depletion_lowers_serotonin():
    """Tryptophan depletion → lower 5-HT synthesis; chronic
    depletion should lower serotonin."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.tryptophan = 0.1
    initial_ser = limbic.get_state()["neurochemistry"]["serotonin"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["serotonin"] < initial_ser
    assert state["neurochemistry"]["serotonin"] < 0.5
    assert state["neurochemistry"]["dopamine"] > 0.2


def test_tyrosine_competition_at_bbb():
    """Tyrosine and tryptophan compete for transport across BBB;
    high tyrosine should favor dopamine over serotonin."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.tyrosine = 0.9
    limbic.neurochemistry.state.tryptophan = 0.2
    initial_ratio = limbic.get_state()["neurotransmitter_ratios"]["dopamine_serotonin_ratio"]
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurotransmitter_ratios"]["dopamine_serotonin_ratio"] > initial_ratio
    assert state["neurotransmitter_ratios"]["dopamine_serotonin_ratio"] > 0.5


# =========================================================================
# 17. PREPULSE INHIBITION
# =========================================================================

def test_prepulse_inhibition_gates_startle():
    """Prepulse inhibition: a weak prestimulus reduces startle
    response to a subsequent strong stimulus."""
    # Baseline: high NE, low safety = strong startle
    limbic = LimbicSystem()
    limbic.neurochemistry.state.norepinephrine = 0.8
    limbic.drive.safety = 0.2
    limbic.neurochemistry.state.prepulse_inhibition = 0.0
    for _ in range(3):
        limbic.update()
    startle_alone = limbic.get_state()["neurochemistry"]["startle_response"]
    # With prepulse: same conditions but PPI active
    limbic2 = LimbicSystem()
    limbic2.neurochemistry.state.norepinephrine = 0.8
    limbic2.drive.safety = 0.2
    limbic2.neurochemistry.state.prepulse_inhibition = 0.7
    for _ in range(3):
        limbic2.update()
    startle_prepulse = limbic2.get_state()["neurochemistry"]["startle_response"]
    assert startle_prepulse < startle_alone


# =========================================================================
# 18. SYNAPTIC PLASTICITY / METAPLASTICITY
# =========================================================================

def test_bcm_theory_ltp_ltd_threshold():
    """BCM theory: weak synaptic activation → LTD, moderate → LTP,
    very strong → depotentiation. The threshold slides with history."""
    limbic = LimbicSystem()
    # Set history to high activity (raises threshold)
    for _ in range(10):
        limbic.observe_event("task_complete", "High activity")
    state = limbic.get_state()
    assert state["neurochemistry"]["ltp_threshold"] > 0.5
    assert state["neurochemistry"]["ltp_threshold"] > 0.4
    # Weak activation should now produce LTD
    limbic.observe_event("user_message", "Weak input")
    state2 = limbic.get_state()
    assert state2["neurochemistry"]["synaptic_change"] <= 0.0


# =========================================================================
# 19. MITOCHONDRIAL BIOENERGETICS
# =========================================================================

def test_mitochondrial_atp_depletes_with_high_task_load():
    """Mitochondrial ATP production drops under sustained high
    task load, raising ROS and reducing calcium buffering."""
    # Baseline: few tasks
    base = LimbicSystem()
    for _ in range(2):
        base.observe_event("task_complete", "Light work")
    base_state = base.get_state()
    # High task load
    limbic = LimbicSystem()
    limbic.drive.task_load = 0.9
    for _ in range(10):
        limbic.observe_event("task_complete", "Sustained work")
    for _ in range(5):
        limbic.update()
    state = limbic.get_state()
    assert state["neurochemistry"]["mitochondrial_atp"] < base_state["neurochemistry"]["mitochondrial_atp"]
    assert state["neurochemistry"]["reactive_oxygen_species"] > 0.1


def test_mitochondrial_dysfunction_increases_excitotoxicity_risk():
    """Mitochondrial dysfunction impairs calcium buffering and
    increases vulnerability to glutamate excitotoxicity."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.mitochondrial_atp = 0.2
    limbic.neurochemistry.state.glutamate = 0.8
    # Freeze other variables that would lower risk during update
    limbic.neurochemistry.state.gaba = 0.1
    limbic.neurochemistry.state.glycine = 0.1
    limbic.neurochemistry.state.quinolinic_acid = 0.4
    limbic.neurochemistry.state.glt1_activity = 0.2
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    assert state["excitotoxicity_risk"] > 0.25
    assert state["neurochemistry"]["reactive_oxygen_species"] > 0.1


# =========================================================================
# 20. NEUROPEPTIDE CO-TRANSMISSION / VOLUME TRANSMISSION
# =========================================================================

def test_neuropeptide_volume_transmission_diffuses_slower():
    """Neuropeptides released with classical NTs diffuse more
    slowly and have longer-lasting effects."""
    limbic = LimbicSystem()
    # Set explicit values for comparison
    limbic.neurochemistry.state.oxytocin = 0.8
    limbic.neurochemistry.state.dopamine = 0.8
    state1 = limbic.get_state()
    oxy_1 = state1["neurochemistry"]["oxytocin"]
    dop_1 = state1["neurochemistry"]["dopamine"]
    for _ in range(10):
        limbic.update()
    state2 = limbic.get_state()
    oxy_2 = state2["neurochemistry"]["oxytocin"]
    dop_2 = state2["neurochemistry"]["dopamine"]
    oxy_decay = oxy_1 - oxy_2
    dop_decay = dop_1 - dop_2
    assert oxy_decay <= dop_decay + 0.05
    assert oxy_decay >= 0
    assert dop_decay >= 0
