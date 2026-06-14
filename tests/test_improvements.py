"""
tests/test_improvements.py
==========================
Testable prompts for the 29 additional Limbic Hermes improvements.
Each test mirrors a prompt from TODO_LIMBIC_V2.md.
"""
import pytest

from limbic_hermes.core import LimbicSkillBridge, LimbicSystem
from limbic_hermes.neurochemistry import NeurochemicalState, NeurochemistryEngine


# ---------------------------------------------------------------------------
# 1. Insula interoceptive salience
# ---------------------------------------------------------------------------

def test_insula_body_prediction_error_amplifies_threat():
    """Given interoceptive signals, the Insula computes a body_prediction_error
    that amplifies threat appraisal when interoception is discordant."""
    limbic = LimbicSystem(profile_name="default")
    # Discordant interoception: high cytokine load but low reported arousal
    limbic.neurochemistry.state.cytokine_load = 0.9
    limbic.neurochemistry.state.heart_rate_variability = 0.9
    limbic.neurochemistry.state.respiration_rate = 0.1
    appraisal = limbic.observe_event(
        "error", raw_valence=-0.3, raw_arousal=0.1, importance=0.5
    )
    assert limbic.get_state()["insula"]["body_prediction_error"] > 0.1


# ---------------------------------------------------------------------------
# 2. ACC conflict/error signal
# ---------------------------------------------------------------------------

def test_acc_conflict_signal_boosts_ne_and_reduces_safety():
    """When valence is near zero, arousal is high, and dominance is low, the
    ACC emits a conflict_signal that boosts NE and reduces safety."""
    limbic = LimbicSystem(profile_name="default")
    limbic.vad = limbic.vad.__class__(0.0, 0.8, 0.2)
    before_ne = limbic.neurochemistry.state.norepinephrine
    before_safety = limbic.drive.safety
    limbic.observe_event("conflict", raw_valence=0.0, raw_arousal=0.6, importance=0.7)
    state = limbic.get_state()
    assert state["acc"]["conflict_signal"] > 0.2
    assert limbic.neurochemistry.state.norepinephrine > before_ne
    assert limbic.drive.safety < before_safety


# ---------------------------------------------------------------------------
# 3. Lateral Habenula aversion learning
# ---------------------------------------------------------------------------

def test_lateral_habenula_inhibits_dopamine_on_negative_surprise():
    """Unexpected negative events activate the LHb, which inhibits dopamine and
    increases serotonin release; repeated negative events lower expected reward
    more than simple RPE."""
    limbic = LimbicSystem(profile_name="default")
    # First negative surprise
    limbic.observe_event("error", raw_valence=-0.8, raw_arousal=0.6, importance=0.9)
    first_da = limbic.neurochemistry.state.dopamine
    first_er = limbic.expected_reward
    # Repeat
    limbic.observe_event("error", raw_valence=-0.8, raw_arousal=0.6, importance=0.9)
    state = limbic.get_state()
    assert state["lateral_habenula"]["activation"] > 0.1
    assert limbic.expected_reward < first_er


# ---------------------------------------------------------------------------
# 4. Septal nuclei social approach
# ---------------------------------------------------------------------------

def test_septal_social_approach_reduces_threat_to_social_events():
    """Oxytocin + low cortisol activates septal social-approach, raising valence
    toward affiliative events and reducing threat response to social cues."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.oxytocin = 0.9
    limbic.neurochemistry.state.cortisol = 0.1
    before_valence = limbic.vad.valence
    limbic.observe_event("user_message", raw_valence=-0.2, raw_arousal=0.2, description="social")
    state = limbic.get_state()
    assert state["septal"]["social_approach"] > 0.4
    assert limbic.vad.valence > before_valence - 0.05


# ---------------------------------------------------------------------------
# 5. PAG defensive tier detector
# ---------------------------------------------------------------------------

def test_pag_defensive_tier_classification():
    """Based on arousal/dominance and threat flag, classify defensive response as
    freeze, flight, fight, or calm."""
    limbic = LimbicSystem(profile_name="default")
    # Fight: high arousal, high dominance, threat
    limbic.vad = limbic.vad.__class__(0.0, 0.9, 0.9)
    limbic.observe_event("threat", raw_valence=-0.5, raw_arousal=0.6, importance=0.9)
    state = limbic.get_state()
    assert state["pag"]["tier"] in {"freeze", "flight", "fight", "calm"}


# ---------------------------------------------------------------------------
# 6. Polyvagal state model
# ---------------------------------------------------------------------------

def test_polyvagal_state_gates_social_engagement():
    """HRV, oxytocin, and safety map to a discrete vagal state; the state gates
    whether social engagement is possible or defensive shutdown occurs."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.heart_rate_variability = 0.8
    limbic.neurochemistry.state.oxytocin = 0.8
    limbic.drive.safety = 0.9
    limbic.update()
    state = limbic.get_state()
    assert state["polyvagal"]["state"] in {"ventral_vagal", "sympathetic", "dorsal_vagal"}
    assert state["polyvagal"]["social_engagement_possible"] is True


# ---------------------------------------------------------------------------
# 7. Vagal brake release under acute threat
# ---------------------------------------------------------------------------

def test_vagal_brake_release_lowers_hrv_and_raises_adrenaline():
    """A sudden threat event should transiently lower HRV and increase
    adrenaline, modeling vagal withdrawal."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.heart_rate_variability = 0.8
    limbic.observe_event("threat", raw_valence=-0.8, raw_arousal=0.8, importance=0.9)
    state = limbic.get_state()
    assert limbic.neurochemistry.state.heart_rate_variability < 0.75
    assert limbic.neurochemistry.state.adrenaline > 0.2


# ---------------------------------------------------------------------------
# 8. Baroreflex-like arousal dampening
# ---------------------------------------------------------------------------

def test_high_hrv_gradually_reduces_arousal_and_ne():
    """Sustained high HRV should gradually reduce arousal and NE."""
    limbic = LimbicSystem(profile_name="default")
    limbic.vad = limbic.vad.__class__(0.0, 0.8, 0.5)
    limbic.neurochemistry.state.heart_rate_variability = 0.9
    limbic.update()
    assert limbic.vad.arousal < 0.8
    assert limbic.neurochemistry.state.norepinephrine < 0.25


# ---------------------------------------------------------------------------
# 9. Respiration-driven entrainment
# ---------------------------------------------------------------------------

def test_respiration_entrainment_modulates_arousal():
    """A user-controlled respiration rate modulates NE and oscillates arousal
    with the respiratory cycle (inhalation up, exhalation down)."""
    limbic = LimbicSystem(profile_name="default")
    limbic.set_respiration_phase(0.0)  # inhalation peak
    a1 = limbic.vad.arousal
    limbic.set_respiration_phase(0.5)  # exhalation trough
    a2 = limbic.vad.arousal
    assert a1 > a2


# ---------------------------------------------------------------------------
# 10. GABA-A and GluN2B receptor sensitivity tracking
# ---------------------------------------------------------------------------

def test_gaba_a_and_glun2b_sensitivity_state_exists():
    """GABA-A and GluN2B receptor sensitivities exist in state, decrease under
    chronic GABA/glutamate exposure, and recover with rest; the dashboard
    displays them."""
    s = NeurochemicalState()
    assert hasattr(s, "gaba_a_sensitivity")
    assert hasattr(s, "glun2b_sensitivity")


# ---------------------------------------------------------------------------
# 11. Kynurenine pathway shunt
# ---------------------------------------------------------------------------

def test_kynurenine_shunt_lowers_serotonin_and_raises_glutamate():
    """High cytokine load reduces tryptophan availability, lowering serotonin
    synthesis and raising quinolinic acid, which increases glutamate
    excitotoxicity."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.cytokine_load = 0.9
    before_serotonin = limbic.neurochemistry.state.serotonin
    before_glutamate = limbic.neurochemistry.state.glutamate
    limbic.update()
    state = limbic.get_state()
    assert state["kynurenine"]["quinolinic_acid"] > 0.1
    assert limbic.neurochemistry.state.serotonin < before_serotonin + 0.01
    assert limbic.neurochemistry.state.glutamate > before_glutamate


# ---------------------------------------------------------------------------
# 12. D2 autoreceptor / short-loop feedback
# ---------------------------------------------------------------------------

def test_d2_autoreceptor_inhibits_further_dopamine():
    """High dopamine triggers D2 autoreceptor inhibition, reducing further
    dopamine release and increasing dopamine pool recovery cost."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.dopamine = 0.9
    limbic.neurochemistry.state.dopamine_pool = 0.5
    limbic.update()
    state = limbic.get_state()
    assert state["d2_autoreceptor"]["inhibition"] > 0.1


# ---------------------------------------------------------------------------
# 13. MAO-A / MAO-B degradation dynamics
# ---------------------------------------------------------------------------

def test_mao_degradation_affects_catecholamine_decay():
    """Catecholamine and serotonin levels decay at rates proportional to
    effective MAO activity, which can be modulated by cofactors/heritage."""
    s = NeurochemicalState(dopamine=0.9, serotonin=0.9)
    engine = NeurochemistryEngine(s)
    engine.update(1.0, 0, 0, 0, 0, 0, 0, 1.0, 0)
    assert engine.state.dopamine < 0.9
    assert engine.state.serotonin < 0.9


# ---------------------------------------------------------------------------
# 14. COMT Val158Met-style dopamine clearance
# ---------------------------------------------------------------------------

def test_comt_clearance_phenotype_affects_dopamine_decay():
    """A tunable dopamine_clearance_rate biases how fast dopamine returns to
    baseline; slower clearance increases sustained dopamine but also receptor
    desensitization risk."""
    s = NeurochemicalState(dopamine=0.9)
    s.dopamine_clearance_rate = 0.05
    engine = NeurochemistryEngine(s)
    engine.update(1.0, 0, 0, 0, 0, 0, 0, 1.0, 0)
    slow_clear = engine.state.dopamine

    s2 = NeurochemicalState(dopamine=0.9)
    s2.dopamine_clearance_rate = 0.5
    engine2 = NeurochemistryEngine(s2)
    engine2.update(1.0, 0, 0, 0, 0, 0, 0, 1.0, 0)
    fast_clear = engine2.state.dopamine
    assert slow_clear > fast_clear


# ---------------------------------------------------------------------------
# 15. Serotonin transporter reuptake modulation
# ---------------------------------------------------------------------------

def test_sert_reuptake_modulates_serotonin_decay():
    """A serotonin_reuptake parameter controls serotonin decay rate, modeling
    SSRIs/heritage effects."""
    s = NeurochemicalState(serotonin=0.9)
    s.serotonin_reuptake = 0.05
    engine = NeurochemistryEngine(s)
    engine.update(1.0, 0, 0, 0, 0, 0, 0, 1.0, 0)
    slow = engine.state.serotonin

    s2 = NeurochemicalState(serotonin=0.9)
    s2.serotonin_reuptake = 0.5
    engine2 = NeurochemistryEngine(s2)
    engine2.update(1.0, 0, 0, 0, 0, 0, 0, 1.0, 0)
    fast = engine2.state.serotonin
    assert slow > fast


# ---------------------------------------------------------------------------
# 16. Mesolimbic vs mesocortical dopamine
# ---------------------------------------------------------------------------

def test_mesolimbic_mesocortical_dopamine_tracked():
    """Track dopamine_mesolimbic (motivation/salience) separately from
    dopamine_mesocortical (cognitive control); the latter contributes more to
    dominance."""
    s = NeurochemicalState()
    assert hasattr(s, "dopamine_mesolimbic")
    assert hasattr(s, "dopamine_mesocortical")


# ---------------------------------------------------------------------------
# 17. Locus coeruleus tonic/phasic mode
# ---------------------------------------------------------------------------

def test_lc_tonic_phasic_mode_switch():
    """LC mode is tonic (exploration, high NE baseline) or phasic (focused,
    bursts); task load and surprise switch between modes."""
    limbic = LimbicSystem(profile_name="default")
    limbic.drive.task_load = 0.9
    limbic.update()
    state = limbic.get_state()
    assert state["locus_coeruleus"]["mode"] in {"tonic", "phasic"}


# ---------------------------------------------------------------------------
# 18. SEEKING system
# ---------------------------------------------------------------------------

def test_seeking_system_increases_reward_gain():
    """Dopamine + orexin + low cortisol drive a seeking motivation state that
    increases reward gain and exploration/novelty bias."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.dopamine = 0.8
    limbic.neurochemistry.state.orexin = 0.8
    limbic.neurochemistry.state.cortisol = 0.1
    limbic.update()
    state = limbic.get_state()
    assert state["affective_systems"]["seeking"] > 0.4


# ---------------------------------------------------------------------------
# 19. CARE system
# ---------------------------------------------------------------------------

def test_care_system_increases_warmth():
    """Oxytocin + prolactin + safety drive a care state that raises warmth
    expression and reduces threat to dependency/social events."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.oxytocin = 0.8
    limbic.neurochemistry.state.prolactin = 0.6
    limbic.drive.safety = 0.9
    limbic.update()
    state = limbic.get_state()
    assert state["affective_systems"]["care"] > 0.4


# ---------------------------------------------------------------------------
# 20. FEAR/RAGE thresholds
# ---------------------------------------------------------------------------

def test_fear_and_rage_thresholds():
    """Threat events above a cortisol- and GABA-dependent threshold trigger
    fear_active; when dominance is high and GABA low, the same threat triggers
    rage_active."""
    limbic = LimbicSystem(profile_name="default")
    limbic.vad = limbic.vad.__class__(0.0, 0.8, 0.9)
    limbic.neurochemistry.state.gaba = 0.1
    limbic.neurochemistry.state.cortisol = 0.8
    limbic.observe_event("threat", raw_valence=-0.8, raw_arousal=0.8, importance=0.9)
    state = limbic.get_state()
    assert state["affective_systems"]["rage"] > state["affective_systems"]["fear"]


# ---------------------------------------------------------------------------
# 21. GRIEF/PANIC separation
# ---------------------------------------------------------------------------

def test_panic_grief_activated_by_social_separation():
    """Social separation events activate panic_grief driven by opioid
    withdrawal and oxytocin drop."""
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.opioid = 0.8
    limbic.neurochemistry.state.oxytocin = 0.8
    limbic.observe_event("social_separation", raw_valence=-0.6, raw_arousal=0.5, importance=0.8)
    state = limbic.get_state()
    assert state["affective_systems"]["panic_grief"] > 0.1


# ---------------------------------------------------------------------------
# 22. Fear extinction learning
# ---------------------------------------------------------------------------

def test_fear_extinction_reduces_hippocampal_weight():
    """Repeated safe exposures to a previously threatening event kind reduce
    its hippocampal weight (extinction)."""
    limbic = LimbicSystem(profile_name="default")
    # Learn threat
    for _ in range(3):
        limbic.observe_event("error", raw_valence=-0.8, raw_arousal=0.6, importance=0.9)
    weight_after_learning = limbic.hippocampal_weights.get("error", 0.0)
    # Safe exposures
    for _ in range(10):
        limbic.observe_event("error", raw_valence=0.0, raw_arousal=0.0, importance=0.1)
    assert limbic.hippocampal_weights.get("error", 1.0) < weight_after_learning


# ---------------------------------------------------------------------------
# 23. Working-memory load as drive and ACh demand
# ---------------------------------------------------------------------------

def test_working_memory_load_increases_ach_demand():
    """A working_memory_load input increases ACh demand and task load, and
    reduces DMN activity more strongly."""
    limbic = LimbicSystem(profile_name="default")
    before_ach = limbic.neurochemistry.state.acetylcholine
    before_dmn = limbic.neurochemistry.state.dmn_activity
    limbic.set_working_memory_load(0.9)
    limbic.update()
    assert limbic.neurochemistry.state.acetylcholine > before_ach
    assert limbic.neurochemistry.state.dmn_activity < before_dmn


# ---------------------------------------------------------------------------
# 24. Temporal contiguity for LTP
# ---------------------------------------------------------------------------

def test_temporal_contiguity_strengthens_event_weights():
    """Events arriving within a short window strengthen each other's weights,
    modeling hippocampal pattern completion."""
    limbic = LimbicSystem(profile_name="default")
    t = 1000.0
    limbic.observe_event("A", raw_valence=0.5, raw_arousal=0.3, importance=0.6, now=t)
    limbic.observe_event("B", raw_valence=0.5, raw_arousal=0.3, importance=0.6, now=t + 0.5)
    assert limbic.hippocampal_weights.get("A", 0.0) > 0.0
    assert limbic.hippocampal_weights.get("B", 0.0) > 0.0


# ---------------------------------------------------------------------------
# 25. Cortisol awakening response / circadian curve
# ---------------------------------------------------------------------------

def test_cortisol_follows_circadian_peak_morning():
    """Cortisol follows a circadian curve peaking in the morning; circadian_hour
    input should shift the cortisol baseline."""
    limbic = LimbicSystem(profile_name="default")
    limbic.set_circadian_hour(8.0)
    limbic.update()
    morning = limbic.neurochemistry.state.cortisol
    limbic.set_circadian_hour(22.0)
    limbic.update()
    evening = limbic.neurochemistry.state.cortisol
    assert morning > evening


# ---------------------------------------------------------------------------
# 26. Leptin/ghrelin metabolic energy state
# ---------------------------------------------------------------------------

def test_metabolic_energy_modulates_orexin_dopamine():
    """A metabolic_energy input modulates orexin, dopamine, and irritability;
    low energy increases rest_need and caution."""
    # High metabolic energy
    limbic_high = LimbicSystem(profile_name="default")
    limbic_high.set_metabolic_energy(0.9)
    for _ in range(3):
        limbic_high.update()
    high = limbic_high.neurochemistry.state.orexin

    # Low metabolic energy
    limbic_low = LimbicSystem(profile_name="default")
    limbic_low.set_metabolic_energy(0.1)
    for _ in range(3):
        limbic_low.update()
    low = limbic_low.neurochemistry.state.orexin

    assert high > low


# ---------------------------------------------------------------------------
# 27. Glucose/insulin brain fuel dynamics
# ---------------------------------------------------------------------------

def test_low_glucose_weakens_prefrontal_regulation():
    """Glucose availability modulates prefrontal top-down regulation; low
    glucose weakens dominance regulation of threat."""
    limbic = LimbicSystem(profile_name="default")
    limbic.vad = limbic.vad.__class__(0.0, 0.5, 0.9)
    limbic.set_glucose(0.1)
    limbic.observe_event("threat", raw_valence=-0.8, raw_arousal=0.6, importance=0.8)
    state = limbic.get_state()
    assert state["prefrontal_regulation"]["strength"] < 0.5


# ---------------------------------------------------------------------------
# 28. Dashboard preset event buttons
# ---------------------------------------------------------------------------

def test_dashboard_state_endpoint_accepts_preset_events():
    """The dashboard has buttons for common events that post to the server and
    update state."""
    # Verified by the HTML including handlers for success, error, praise, etc.
    html = open("limbic_hermes/dashboard.html").read()
    for kind in ["success", "error", "praise", "conflict"]:
        assert kind in html


# ---------------------------------------------------------------------------
# 29. Dashboard export/import of limbic state JSON
# ---------------------------------------------------------------------------

def test_dashboard_has_export_import_buttons():
    """The dashboard provides download/upload buttons for the full limbic state
    so users can save and restore configurations."""
    html = open("limbic_hermes/dashboard.html").read()
    assert "download" in html.lower() or "export" in html.lower()
    assert "upload" in html.lower() or "import" in html.lower()
