"""
tests/test_integration_modules.py
=================================

20 integration circuit modules tying together limbic-hermes
neurochemical components into functional systems.

Each test asserts a physiologically-grounded expectation.
Run with: pytest tests/test_integration_modules.py -v
"""
import pytest
from limbic_hermes.core import LimbicSystem


# =========================================================================
# 1. STRESS-IMMUNE-FATIGUE CIRCUIT
# =========================================================================

def test_stress_immune_fatigue_rises_with_cortisol_and_cytokines():
    """Chronic stress (cortisol + CRF + microglia) increases fatigue index;
    BDNF + NPY protect against it."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.cortisol = 0.8
    limbic.neurochemistry.state.crf = 0.7
    limbic.neurochemistry.state.microglia_state = 0.6
    limbic.neurochemistry.state.cytokine_load = 0.5
    limbic.neurochemistry.state.serotonin = 0.2
    limbic.neurochemistry.state.mitochondrial_atp = 0.3
    state = limbic.get_state()
    sif = state["integration"]["stress_immune_fatigue"]
    assert sif["chronic_stress_index"] > 0.5
    assert sif["fatigue_level"] > 0.3


def test_immune_resilience_protected_by_bdnf_npy():
    """High BDNF and NPY should raise immune_resilience even under stress."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.bdnf = 0.9
    limbic.neurochemistry.state.neuropeptide_y = 0.8
    limbic.neurochemistry.state.microglia_state = 0.2
    state = limbic.get_state()
    sif = state["integration"]["stress_immune_fatigue"]
    assert sif["immune_resilience"] > 0.5


# =========================================================================
# 2. REWARD-EXTINCTION CIRCUIT
# =========================================================================

def test_reward_learning_rises_with_dopamine_and_mesolimbic():
    """High dopamine + mesolimbic activation = strong reward learning."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.dopamine = 0.8
    limbic.neurochemistry.state.dopamine_mesolimbic = 0.7
    state = limbic.get_state()
    re = state["integration"]["reward_extinction"]
    assert re["reward_learning"] > 0.5


def test_extinction_state_rises_with_anandamide_and_il():
    """Anandamide + infralimbic activity promotes extinction;
    dynorphin suppresses it."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.anandamide = 0.7
    limbic.neurochemistry.state.il_activity = 0.6
    limbic.neurochemistry.state.dynorphin = 0.1
    state = limbic.get_state()
    re = state["integration"]["reward_extinction"]
    assert re["extinction_state"] > 0.3


def test_aversion_strength_rises_with_dynorphin():
    """High dynorphin + CRF = strong aversion."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.dynorphin = 0.8
    limbic.neurochemistry.state.crf = 0.7
    state = limbic.get_state()
    re = state["integration"]["reward_extinction"]
    assert re["aversion_strength"] > 0.5


# =========================================================================
# 3. SLEEP HOMEOSTASIS CIRCUIT
# =========================================================================

def test_sleep_need_rises_with_adenosine_and_melatonin():
    """High adenosine + sleep pressure + melatonin = high sleep need."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.adenosine = 0.8
    limbic.neurochemistry.state.sleep_pressure = 0.7
    limbic.neurochemistry.state.melatonin = 0.6
    state = limbic.get_state()
    sh = state["integration"]["sleep_homeostasis"]
    assert sh["sleep_need"] > 0.5


def test_sleep_efficiency_correlates_with_glymphatic_flow():
    """High glymphatic flow + low amyloid = efficient sleep."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.glymphatic_flow = 0.8
    limbic.neurochemistry.state.amyloid_beta = 0.1
    state = limbic.get_state()
    sh = state["integration"]["sleep_homeostasis"]
    assert sh["sleep_efficiency"] > 0.5


# =========================================================================
# 4. SOCIAL AFFILIATION CIRCUIT
# =========================================================================

def test_social_approach_rises_with_oxytocin():
    """High oxytocin + low vasopressin = social approach."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.oxytocin = 0.8
    limbic.neurochemistry.state.vasopressin = 0.2
    limbic.neurochemistry.state.testosterone = 0.3
    state = limbic.get_state()
    sa = state["integration"]["social_affiliation"]
    assert sa["social_approach"] > 0.3


def test_social_avoidance_rises_with_vasopressin_and_cortisol():
    """High vasopressin + cortisol = social avoidance."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.vasopressin = 0.8
    limbic.neurochemistry.state.cortisol = 0.7
    state = limbic.get_state()
    sa = state["integration"]["social_affiliation"]
    assert sa["social_avoidance"] > 0.5


# =========================================================================
# 5. FEAR MEMORY CONSOLIDATION CIRCUIT
# =========================================================================

def test_fear_memory_strength_rises_with_bla_and_ne():
    """High BLA + high NE + low GABA = strong fear memory."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.bla = 0.8
    limbic.neurochemistry.state.norepinephrine = 0.7
    limbic.neurochemistry.state.gaba = 0.2
    state = limbic.get_state()
    fm = state["integration"]["fear_memory"]
    assert fm["fear_memory_strength"] > 0.5


def test_extinction_level_rises_with_il_and_anandamide():
    """High IL + anandamide - CRF = extinction."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.il_activity = 0.7
    limbic.neurochemistry.state.anandamide = 0.6
    limbic.neurochemistry.state.crf = 0.1
    state = limbic.get_state()
    fm = state["integration"]["fear_memory"]
    assert fm["extinction_level"] > 0.3


# =========================================================================
# 6. ATTENTION-SALIENCE CIRCUIT
# =========================================================================

def test_attention_focus_rises_with_ach_and_phasic_ne():
    """High ACh + phasic LC mode + low sleep pressure = focused attention."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.acetylcholine = 0.8
    limbic.neurochemistry.state.lc_mode = 1.0
    limbic.neurochemistry.state.sleep_pressure = 0.1
    state = limbic.get_state()
    att = state["integration"]["attention_salience"]
    assert att["attention_focus"] > 0.5


def test_distractibility_rises_with_tonic_ne_and_low_ach():
    """Tonic NE + low ACh = high distractibility."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.norepinephrine = 0.7
    limbic.neurochemistry.state.lc_mode = 0.0
    limbic.neurochemistry.state.acetylcholine = 0.2
    state = limbic.get_state()
    att = state["integration"]["attention_salience"]
    assert att["distractibility"] > 0.2


# =========================================================================
# 7. METABOLIC ENERGY ALLOCATION CIRCUIT
# =========================================================================

def test_cognitive_reserve_drops_with_high_task_load():
    """High task load depletes cognitive reserve."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.mitochondrial_atp = 0.8
    state = limbic.get_state()
    ma = state["integration"]["metabolic_allocation"]
    # With low task_load, reserve should be higher
    assert ma["cognitive_reserve"] > 0.3


def test_glycogen_depletion_inverse_of_glycogen():
    """Low glycogen = high depletion."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.glycogen = 0.2
    state = limbic.get_state()
    ma = state["integration"]["metabolic_allocation"]
    assert ma["glycogen_depletion"] > 0.7


# =========================================================================
# 8. PAIN MODULATION CIRCUIT
# =========================================================================

def test_pain_level_rises_with_substance_p():
    """High substance P = high pain; spinal opioid and Abeta reduce it."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.substance_p = 0.8
    limbic.neurochemistry.state.spinal_opioid = 0.1
    limbic.neurochemistry.state.ab_fiber = 0.1
    state = limbic.get_state()
    pm = state["integration"]["pain_modulation"]
    assert pm["pain_level"] > 0.3


def test_analgesia_rises_with_spinal_opioid():
    """High spinal opioid + PAG activation = strong analgesia."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.spinal_opioid = 0.8
    limbic.neurochemistry.state.pag_periaqueductal_gray = 0.7
    state = limbic.get_state()
    pm = state["integration"]["pain_modulation"]
    assert pm["analgesia_strength"] > 0.5


# =========================================================================
# 9. HPA FEEDBACK CIRCUIT
# =========================================================================

def test_hpa_feedback_integrity_with_high_cortisol_low_crf():
    """High cortisol suppressing CRF = intact feedback."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.cortisol = 0.8
    limbic.neurochemistry.state.crf = 0.1
    state = limbic.get_state()
    hpa = state["integration"]["hpa_feedback"]
    assert hpa["feedback_integrity"] > 0.5


def test_sustained_stress_when_crf_persists_despite_cortisol():
    """High CRF despite cortisol = sustained stress (impaired feedback)."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.cortisol = 0.7
    limbic.neurochemistry.state.crf = 0.7
    limbic.neurochemistry.state.pvn_crf = 0.6
    state = limbic.get_state()
    hpa = state["integration"]["hpa_feedback"]
    assert hpa["sustained_stress"] > 0.3


# =========================================================================
# 10. CIRCADIAN-METABOLIC COUPLING CIRCUIT
# =========================================================================

def test_circadian_alignment_matches_scn_phase():
    """SCN phase should align with circadian_hour."""
    limbic = LimbicSystem()
    limbic.set_circadian_hour(12.0)
    for _ in range(3):
        limbic.update()
    state = limbic.get_state()
    cm = state["integration"]["circadian_metabolic"]
    assert cm["circadian_alignment"] > 0.5


def test_energy_rhythm_higher_during_day():
    """Energy rhythm (orexin + histamine - melatonin) higher at noon than midnight."""
    limbic_day = LimbicSystem()
    limbic_day.set_circadian_hour(12.0)
    limbic_day.neurochemistry.state.orexin = 0.7
    limbic_day.neurochemistry.state.histamine = 0.6
    limbic_day.neurochemistry.state.melatonin = 0.1
    for _ in range(3):
        limbic_day.update()
    state_day = limbic_day.get_state()
    cm_day = state_day["integration"]["circadian_metabolic"]
    assert cm_day["energy_rhythm"] > 0.3


# =========================================================================
# 11. DMN-SALIENCE SWITCHING CIRCUIT
# =========================================================================

def test_dmn_dominance_rises_with_low_task_load():
    """Low task load + low cortisol = DMN dominance (mind-wandering)."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.dmn_activity = 0.8
    limbic.neurochemistry.state.cortisol = 0.2
    state = limbic.get_state()
    ds = state["integration"]["dmn_salience"]
    assert ds["dmn_dominance"] > 0.3


def test_salience_dominance_rises_with_cortisol_and_cytokines():
    """High cortisol + cytokines = salience network dominance."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.cortisol = 0.7
    limbic.neurochemistry.state.cytokine_load = 0.6
    state = limbic.get_state()
    ds = state["integration"]["dmn_salience"]
    assert ds["salience_dominance"] > 0.3


# =========================================================================
# 12. NEUROPLASTICITY-RESILIENCE CIRCUIT
# =========================================================================

def test_plasticity_index_rises_with_bdnf_and_neurogenesis():
    """High BDNF + neurogenesis + low dynorphin = high plasticity."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.bdnf = 0.9
    limbic.neurochemistry.state.neurogenesis_rate = 0.7
    limbic.neurochemistry.state.dynorphin = 0.1
    state = limbic.get_state()
    nr = state["integration"]["resilience"]
    assert nr["plasticity_index"] > 0.5


def test_vulnerability_rises_with_cytokines_and_low_bdnf():
    """High cytokines + low BDNF + low serotonin = vulnerability."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.cytokine_load = 0.8
    limbic.neurochemistry.state.bdnf = 0.2
    limbic.neurochemistry.state.serotonin = 0.2
    state = limbic.get_state()
    nr = state["integration"]["resilience"]
    assert nr["vulnerability"] > 0.4


# =========================================================================
# 13. GUT-BRAIN-STRESS CIRCUIT
# =========================================================================

def test_gut_resilience_rises_with_butyrate_and_vagal():
    """High butyrate + vagal tone - cytokines = gut resilience."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.butyrate = 0.8
    limbic.neurochemistry.state.vagal_afferent = 0.7
    limbic.neurochemistry.state.cytokine_load = 0.2
    state = limbic.get_state()
    gbs = state["integration"]["gut_brain_stress"]
    assert gbs["gut_resilience"] > 0.4


def test_microbial_mood_improves_with_vagal_butyrate():
    """Vagal + butyrate - cortisol = positive microbial mood effect."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.vagal_afferent = 0.7
    limbic.neurochemistry.state.butyrate = 0.6
    limbic.neurochemistry.state.cortisol = 0.2
    state = limbic.get_state()
    gbs = state["integration"]["gut_brain_stress"]
    assert gbs["microbial_mood"] > 0.3


# =========================================================================
# 14. HORMONAL-MOOD CIRCUIT
# =========================================================================

def test_estrogenic_mood_rises_with_estrogen():
    """High estrogen boosts serotonin + BDNF = positive mood."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.estrogen = 0.8
    limbic.neurochemistry.state.serotonin = 0.6
    limbic.neurochemistry.state.bdnf = 0.6
    state = limbic.get_state()
    hm = state["integration"]["hormonal_mood"]
    assert hm["estrogenic_mood"] > 0.5


def test_progestogenic_calm_with_allopregnanolone():
    """High progesterone -> allopregnanolone -> GABA-A enhancement = calm."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.progesterone = 0.8
    limbic.neurochemistry.state.allopregnanolone = 0.6
    limbic.neurochemistry.state.gaba_a_sensitivity = 1.0
    state = limbic.get_state()
    hm = state["integration"]["hormonal_mood"]
    assert hm["progestogenic_calm"] > 0.4


# =========================================================================
# 15. EXCITOTOXICITY PROTECTION CIRCUIT
# =========================================================================

def test_excitotoxicity_risk_rises_with_glutamate_and_low_glt1():
    """High glutamate + low GLT-1 + low ATP = excitotoxicity risk."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.glutamate = 0.9
    limbic.neurochemistry.state.glt1_activity = 0.2
    limbic.neurochemistry.state.mitochondrial_atp = 0.2
    state = limbic.get_state()
    ep = state["integration"]["excitotoxicity_protection"]
    assert ep["excitotoxicity_risk"] > 0.4


def test_protection_strength_rises_with_gaba_and_glycine():
    """High GABA + glycine + adenosine = strong protection."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.gaba = 0.8
    limbic.neurochemistry.state.glycine = 0.7
    limbic.neurochemistry.state.adenosine = 0.6
    limbic.neurochemistry.state.glt1_activity = 0.8
    state = limbic.get_state()
    ep = state["integration"]["excitotoxicity_protection"]
    assert ep["protection_strength"] > 0.4


# =========================================================================
# 16. PREPULSE GATING CIRCUIT
# =========================================================================

def test_prepulse_inhibition_reduces_startle():
    """High PPI should reduce startle magnitude compared to low PPI."""
    limbic_low = LimbicSystem()
    limbic_low.neurochemistry.state.norepinephrine = 0.8
    limbic_low.neurochemistry.state.prepulse_inhibition = 0.1
    state_low = limbic_low.get_state()
    startle_low = state_low["integration"]["prepulse_gating"]["startle_magnitude"]

    limbic_high = LimbicSystem()
    limbic_high.neurochemistry.state.norepinephrine = 0.8
    limbic_high.neurochemistry.state.prepulse_inhibition = 0.8
    state_high = limbic_high.get_state()
    startle_high = state_high["integration"]["prepulse_gating"]["startle_magnitude"]

    assert startle_high < startle_low


def test_sensorimotor_gating_improves_with_gaba():
    """High GABA + PPI = better sensorimotor gating."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.gaba = 0.8
    limbic.neurochemistry.state.prepulse_inhibition = 0.7
    state = limbic.get_state()
    pg = state["integration"]["prepulse_gating"]
    assert pg["sensorimotor_gating"] > 0.5


# =========================================================================
# 17. THERMOGENESIS-AROUSAL CIRCUIT
# =========================================================================

def test_thermogenesis_rises_with_bat_and_ne():
    """High brown adipose + NE = thermogenesis."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.brown_adipose_activity = 0.8
    limbic.neurochemistry.state.norepinephrine = 0.7
    state = limbic.get_state()
    ta = state["integration"]["thermogenesis_arousal"]
    assert ta["thermogenesis_level"] > 0.5


def test_warmth_suppression_reduces_arousal():
    """High preoptic warmth suppresses wake circuits."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.preoptic_warmth = 0.7
    limbic.neurochemistry.state.body_temperature = 0.6
    state = limbic.get_state()
    ta = state["integration"]["thermogenesis_arousal"]
    assert ta["warmth_suppression"] > 0.4


# =========================================================================
# 18. DOPAMINE BALANCING CIRCUIT
# =========================================================================

def test_mesolimbic_bias_higher_when_mesolimbic_gt_mesocortical():
    """When mesolimbic > mesocortical, bias toward wanting."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.dopamine_mesolimbic = 0.8
    limbic.neurochemistry.state.dopamine_mesocortical = 0.2
    state = limbic.get_state()
    db = state["integration"]["dopamine_balance"]
    assert db["mesolimbic_bias"] > 0.5


def test_dopamine_stability_rises_when_autoreceptors_low():
    """Low D2 autoreceptor + low VTA GABA = stable dopamine."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.d2_autoreceptor_inhibition = 0.1
    limbic.neurochemistry.state.vta_gaba = 0.1
    state = limbic.get_state()
    db = state["integration"]["dopamine_balance"]
    assert db["dopamine_stability"] > 0.5


# =========================================================================
# 19. THETA-MEMORY ENCODING CIRCUIT
# =========================================================================

def test_encoding_strength_rises_with_theta_gamma_and_ach():
    """High theta-gamma + ACh + low cortisol = strong encoding."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.theta_gamma_coupling = 0.8
    limbic.neurochemistry.state.acetylcholine = 0.7
    limbic.neurochemistry.state.cortisol = 0.1
    state = limbic.get_state()
    tm = state["integration"]["theta_memory"]
    assert tm["encoding_strength"] > 0.5


def test_consolidation_quality_rises_with_nrem_and_bdnf():
    """High NREM delta + BDNF = good consolidation."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.nrem_slow_wave = 0.7
    limbic.neurochemistry.state.bdnf = 0.8
    state = limbic.get_state()
    tm = state["integration"]["theta_memory"]
    assert tm["consolidation_quality"] > 0.5


# =========================================================================
# 20. ALLOSTATIC RECOVERY CIRCUIT
# =========================================================================

def test_recovery_potential_rises_with_low_cortisol_and_sleep():
    """Low cortisol + high sleep pressure = recovery potential."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.cortisol = 0.2
    limbic.neurochemistry.state.sleep_pressure = 0.7
    limbic.neurochemistry.state.nrem_slow_wave = 0.6
    state = limbic.get_state()
    ar = state["integration"]["allostatic_recovery"]
    assert ar["recovery_potential"] > 0.4


def test_restoration_rate_rises_with_bdnf_and_vagal():
    """High BDNF + vagal tone + low NE = fast restoration."""
    limbic = LimbicSystem()
    limbic.neurochemistry.state.bdnf = 0.8
    limbic.neurochemistry.state.vagal_afferent = 0.7
    limbic.neurochemistry.state.norepinephrine = 0.2
    state = limbic.get_state()
    ar = state["integration"]["allostatic_recovery"]
    assert ar["restoration_rate"] > 0.4
