"""
tests/test_profiles_and_snp.py
==============================

Tests for:
1. All 50 remedy personality profiles load correctly
2. SNP preset profiles apply neurochemical modifiers correctly
3. Dashboard API endpoints for SNP profiles
4. LimbicSystem integrates SNP profiles at init and via set_snp_profile()
"""
import pytest
from limbic_hermes.core import LimbicSystem
from limbic_hermes.profiles import full_remedy_library
from limbic_hermes.metabolic_snp import (
    get_snp_presets,
    apply_snp_profile_to_neurochemistry,
    MetabolicSNPProfile,
    SNP_REGISTRY,
)


# ---------------------------------------------------------------------------
# Remedy Profiles
# ---------------------------------------------------------------------------

class TestRemedyProfiles:
    """Verify all 50 remedy temperament profiles are present and loadable."""

    def test_all_50_profiles_exist(self):
        lib = full_remedy_library()
        assert len(lib) == 53, f"Expected 53 profiles (default + 52 remedies), got {len(lib)}"

    def test_each_profile_has_valid_vad(self):
        lib = full_remedy_library()
        for name, profile in lib.items():
            assert -1.0 <= profile.baseline_vad.valence <= 1.0, f"{name}: valence out of range"
            assert 0.0 <= profile.baseline_vad.arousal <= 1.0, f"{name}: arousal out of range"
            assert 0.0 <= profile.baseline_vad.dominance <= 1.0, f"{name}: dominance out of range"

    def test_each_profile_has_valid_gains(self):
        lib = full_remedy_library()
        for name, profile in lib.items():
            assert profile.threat_gain > 0, f"{name}: threat_gain must be positive"
            assert profile.reward_gain > 0, f"{name}: reward_gain must be positive"
            assert profile.decay_factor > 0, f"{name}: decay_factor must be positive"
            assert profile.recovery_factor > 0, f"{name}: recovery_factor must be positive"

    def test_limbic_system_loads_each_profile(self):
        """All profiles can be instantiated in a LimbicSystem without error."""
        lib = full_remedy_library()
        for name in lib:
            limbic = LimbicSystem(profile_name=name)
            assert limbic.profile.name == name
            assert limbic.vad is not None

    def test_pulsatilla_baseline(self):
        limbic = LimbicSystem(profile_name="pulsatilla")
        assert limbic.profile.name == "pulsatilla"
        assert limbic.profile.baseline_vad.valence > 0
        assert limbic.profile.expression_warmth > 0.5
        assert limbic.profile.expression_cling > 0.5

    def test_bryonia_baseline(self):
        limbic = LimbicSystem(profile_name="bryonia")
        assert limbic.profile.name == "bryonia"
        assert limbic.profile.baseline_vad.valence < 0
        assert limbic.profile.attention_safety_bias > 0.7
        assert limbic.profile.rest_sensitivity > 1.3

    def test_helleborus_extremely_low_arousal(self):
        limbic = LimbicSystem(profile_name="helleborus")
        assert limbic.profile.baseline_vad.arousal < 0.1
        assert limbic.profile.baseline_vad.valence < -0.3

    def test_veratrum_high_dominance(self):
        limbic = LimbicSystem(profile_name="veratrum")
        assert limbic.profile.baseline_vad.dominance > 0.65
        assert limbic.profile.error_sensitivity > 1.2


# ---------------------------------------------------------------------------
# SNP Profiles
# ---------------------------------------------------------------------------

class TestSNPPresets:
    """Verify SNP presets exist and produce expected neurochemical shifts."""

    def test_all_presets_exist(self):
        presets = get_snp_presets()
        expected = {
            "default", "high_dopamine", "low_dopamine", "high_serotonin",
            "low_serotonin", "anxiety_prone", "resilient", "stress_vulnerable",
            "reward_deficient", "night_owl", "pain_sensitive",
        }
        assert set(presets.keys()) == expected, f"Missing/extra: {set(presets.keys()) ^ expected}"

    def test_resilient_preset_increases_anandamide(self):
        limbic = LimbicSystem(snp_profile_name="resilient")
        default = LimbicSystem()
        # FAAH C385A A/A reduces FAAH activity = higher anandamide
        assert limbic.neurochemistry.state.anandamide > default.neurochemistry.state.anandamide
        assert limbic.neurochemistry.state.anandamide == pytest.approx(0.4, abs=0.01)

    def test_high_dopamine_preset_increases_dopamine(self):
        limbic = LimbicSystem(snp_profile_name="high_dopamine")
        default = LimbicSystem()
        assert limbic.neurochemistry.state.dopamine > default.neurochemistry.state.dopamine

    def test_night_owl_affects_circadian_chemistry(self):
        limbic = LimbicSystem(snp_profile_name="night_owl")
        default = LimbicSystem()
        # CLOCK + PER3 variants should increase cortisol baseline
        assert limbic.neurochemistry.state.cortisol > default.neurochemistry.state.cortisol

    def test_low_dopamine_preset_decreases_dopamine(self):
        limbic = LimbicSystem(snp_profile_name="low_dopamine")
        default = LimbicSystem()
        assert limbic.neurochemistry.state.dopamine < default.neurochemistry.state.dopamine

    def test_anxiety_prone_preset_increases_threat_reactivity(self):
        limbic = LimbicSystem(snp_profile_name="anxiety_prone")
        # SLC6A4 S/S + GABRA2 risk + FKBP5 T/T should increase baseline cortisol
        assert limbic.neurochemistry.state.cortisol > 0.5
        default = LimbicSystem()
        assert limbic.neurochemistry.state.cortisol > default.neurochemistry.state.cortisol

    def test_stress_vulnerable_decreases_bdnf(self):
        limbic = LimbicSystem(snp_profile_name="stress_vulnerable")
        default = LimbicSystem()
        assert limbic.neurochemistry.state.bdnf < default.neurochemistry.state.bdnf

    def test_snp_profile_in_get_state(self):
        limbic = LimbicSystem(snp_profile_name="resilient")
        state = limbic.get_state()
        assert state["snp_profile"] is not None
        assert state["snp_profile"]["name"] == "resilient"
        assert "effects" in state["snp_profile"]
        assert len(state["snp_profile"]["effects"]) > 0

    def test_set_snp_profile_runtime(self):
        limbic = LimbicSystem()
        assert limbic.snp_profile is None
        limbic.set_snp_profile("anxiety_prone")
        assert limbic.snp_profile.name == "anxiety_prone"
        state = limbic.get_state()
        assert state["snp_profile"]["name"] == "anxiety_prone"

    def test_set_snp_profile_invalid_raises(self):
        limbic = LimbicSystem()
        with pytest.raises(ValueError, match="Unknown SNP profile"):
            limbic.set_snp_profile("nonexistent_profile")

    def test_default_snp_profile_is_none(self):
        limbic = LimbicSystem()
        assert limbic.snp_profile is None
        state = limbic.get_state()
        assert state["snp_profile"] is None

    def test_snp_effects_are_cumulative(self):
        """Multiple SNPs in a preset should combine their effects."""
        profile = get_snp_presets()["stress_vulnerable"]
        effects = profile.get_variant_effects()
        # Should have effects from BDNF, FKBP5, MTHFR, APOE
        assert len(effects) >= 2
        # Cortisol should be increased
        assert effects.get("cortisol", 0) > 0
        # BDNF should be decreased
        assert effects.get("bdnf", 0) < 0


# ---------------------------------------------------------------------------
# SNP Variant Registry
# ---------------------------------------------------------------------------

class TestSNPRegistry:
    """Verify individual SNP variants are correctly defined."""

    def test_comt_val158met_effects(self):
        variant = SNP_REGISTRY["COMT_Val158Met"]
        assert variant.dopamine_mod > 0
        assert variant.threat_gain_mod < 0
        assert "dopamine clearance" in variant.description.lower()

    def test_faah_c385a_is_resilience_snp(self):
        variant = SNP_REGISTRY["FAAH_C385A"]
        assert variant.anandamide_mod > 0
        assert variant.threat_gain_mod < 0
        assert "resilience" in variant.description.lower()

    def test_bdnf_val66met_reduces_bdnf(self):
        variant = SNP_REGISTRY["BDNF_Val66Met"]
        assert variant.bdnf_mod < 0

    def test_drd2_taq1a_reduces_d2(self):
        variant = SNP_REGISTRY["DRD2_Taq1A"]
        assert variant.d2_sensitivity_mod < 0
        assert variant.reward_gain_mod < 0

    def test_mthfr_c677t_description(self):
        variant = SNP_REGISTRY["MTHFR_C677T"]
        assert "methylenetetrahydrofolate" in variant.description
        assert variant.serotonin_mod < 0
