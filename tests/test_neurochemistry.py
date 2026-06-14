"""Tests for the new neurochemistry layer."""
import pytest

from limbic_hermes.core import LimbicSystem


def test_neurochemistry_present_in_state():
    limbic = LimbicSystem(profile_name="default")
    state = limbic.get_state()
    assert "neurochemistry" in state
    assert "allostatic_load" in state
    assert "neurotransmitter_ratios" in state
    assert "dopamine_serotonin_ratio" in state["neurotransmitter_ratios"]


def test_dopamine_rises_after_reward():
    limbic = LimbicSystem(profile_name="default")
    before = limbic.neurochemistry.state.dopamine
    limbic.observe_event("success", raw_valence=0.9, importance=1.0)
    after = limbic.neurochemistry.state.dopamine
    assert after >= before


def test_cortisol_rises_after_error():
    limbic = LimbicSystem(profile_name="default")
    before = limbic.neurochemistry.state.cortisol
    limbic.report_error(severity=1.0)
    after = limbic.neurochemistry.state.cortisol
    assert after > before


def test_top_down_regulation_reduces_threat():
    limbic = LimbicSystem(profile_name="default")
    limbic.vad.dominance = 0.9
    limbic.observe_event("error", raw_valence=-0.8, raw_arousal=0.5, importance=1.0)
    # High dominance should buffer the negative valence somewhat
    assert limbic.vad.valence > -1.0


def test_allostatic_load_increases_with_stress():
    limbic = LimbicSystem(profile_name="default")
    before = limbic.neurochemistry.allostatic_load()
    for _ in range(5):
        limbic.report_error(severity=1.0)
    after = limbic.neurochemistry.allostatic_load()
    assert after >= before


def test_circadian_melatonin():
    limbic = LimbicSystem(profile_name="default")
    limbic.set_circadian_hour(3.0)
    limbic.update(now=limbic.last_update + 1)
    night_melatonin = limbic.neurochemistry.state.melatonin

    limbic.set_circadian_hour(12.0)
    limbic.update(now=limbic.last_update + 1)
    day_melatonin = limbic.neurochemistry.state.melatonin

    assert night_melatonin > day_melatonin


def test_neurotransmitter_pool_depletion_and_recovery():
    limbic = LimbicSystem(profile_name="default")
    before = limbic.neurochemistry.state.dopamine_pool
    for _ in range(20):
        limbic.observe_event("success", raw_valence=0.9, importance=1.0)
    depleted = limbic.neurochemistry.state.dopamine_pool
    assert depleted < before

    limbic.rest(duration_sec=10.0)
    recovered = limbic.neurochemistry.state.dopamine_pool
    assert recovered > depleted


def test_hippocampal_ltp():
    limbic = LimbicSystem(profile_name="default")
    for _ in range(5):
        limbic.observe_event("success", raw_valence=0.8, importance=1.0)
    assert limbic.hippocampal_weights.get("success", 0.0) > 0.0


def test_endocannabinoid_calms_after_arousal():
    limbic = LimbicSystem(profile_name="default")
    limbic.observe_event("tool_failure", raw_valence=-0.8, raw_arousal=0.8, importance=1.0)
    high_ecb = limbic.neurochemistry.state.endocannabinoid
    assert high_ecb > 0.25

    # Wait a bit for eCB to decay (retrograde suppression subsides)
    limbic.update(now=limbic.last_update + 5)
    assert limbic.neurochemistry.state.endocannabinoid < high_ecb


def test_receptor_desensitization():
    """D1 receptors desensitize under sustained high dopamine."""
    limbic = LimbicSystem(profile_name="default")
    # Manually sustain high dopamine with small dt updates to test desensitization
    limbic.neurochemistry.state.dopamine = 0.8
    for _ in range(10):
        limbic.neurochemistry.update(
            dt=0.1, appraisal_valence=0.0, appraisal_arousal=0.0, appraisal_dominance=0.0,
            drive_error_temperature=0.0, drive_rest_need=0.0, drive_task_load=0.0,
            drive_safety=0.5, surprise=0.0, circadian_hour=0.0, metabolic_energy=0.5,
            glucose=0.5, expected_reward=0.0, novelty=0.0,
        )
    assert limbic.neurochemistry.state.d1_sensitivity < 1.0


def test_dmn_suppressed_by_task_load():
    limbic = LimbicSystem(profile_name="default")
    baseline_dmn = limbic.neurochemistry.state.dmn_activity
    limbic.add_task_load(amount=0.8)
    limbic.update(now=limbic.last_update + 1)
    assert limbic.neurochemistry.state.dmn_activity < baseline_dmn


def test_oxytocin_rises_with_praise():
    limbic = LimbicSystem(profile_name="default")
    before = limbic.neurochemistry.state.oxytocin
    limbic.observe_event("praise", raw_valence=0.8, importance=1.0)
    after = limbic.neurochemistry.state.oxytocin
    assert after >= before


def test_vasopressin_rises_when_safety_low():
    limbic = LimbicSystem(profile_name="default")
    limbic.drive.safety = 0.1
    before = limbic.neurochemistry.state.vasopressin
    limbic.update(now=limbic.last_update + 1)
    after = limbic.neurochemistry.state.vasopressin
    assert after > before


def test_acetylcholine_attention_mode():
    limbic = LimbicSystem(profile_name="default")
    limbic.neurochemistry.state.acetylcholine = 0.1
    low_ach_gate = limbic._thalamic_gate(0.5, limbic._appraise("user_message", 0.0, 0.0, 0.0, 0.5))
    limbic.neurochemistry.state.acetylcholine = 0.9
    high_ach_gate = limbic._thalamic_gate(0.5, limbic._appraise("user_message", 0.0, 0.0, 0.0, 0.5))
    assert high_ach_gate != low_ach_gate
