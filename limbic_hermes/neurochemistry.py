"""
limbic_hermes/neurochemistry.py
==============================
Biochemistry layer for the limbic system.

This module models major neurotransmitters, neuromodulators, and related
signals as continuous state variables that influence appraisal, drives,
learning, and expression. All variables are normalized to [0, 1] or [-1, 1]
and updated in small, physiologically-plausible steps.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clamp11(x: float) -> float:
    return max(-1.0, min(1.0, x))


@dataclass
class NeurochemicalState:
    """Container for neurotransmitter and neuromodulator levels."""

    # Monoamines
    serotonin: float = 0.5
    dopamine: float = 0.3
    norepinephrine: float = 0.2

    # Amino-acid neurotransmitters
    gaba: float = 0.5
    glutamate: float = 0.4
    glycine: float = 0.4

    # Cholinergic
    acetylcholine: float = 0.4

    # Endocannabinoid
    endocannabinoid: float = 0.2

    # Neuropeptides / hormones
    oxytocin: float = 0.3
    vasopressin: float = 0.2
    cortisol: float = 0.1
    adrenaline: float = 0.0
    opioid: float = 0.3
    histamine: float = 0.5
    melatonin: float = 0.2
    bdnf: float = 0.5
    nitric_oxide: float = 0.2
    phenylethylamine: float = 0.2
    tyramine: float = 0.2
    neuropeptide_s: float = 0.2
    orexin: float = 0.5
    substance_p: float = 0.1
    cytokine_load: float = 0.0
    d2_autoreceptor_inhibition: float = 0.0

    # Kynurenine / neuroinflammation metabolites
    kynurenine: float = 0.1
    quinolinic_acid: float = 0.0
    picolinic_acid: float = 0.1

    # Enzyme / transporter phenotypes
    mao_activity: float = 1.0
    dopamine_clearance_rate: float = 0.2
    serotonin_reuptake: float = 0.2

    # Additional receptor sensitivities
    gaba_a_sensitivity: float = 1.0
    glun2b_sensitivity: float = 1.0

    # Respiration / interoception
    respiration_phase: float = 0.0

    # Dopamine pathway divergence
    dopamine_mesolimbic: float = 0.27
    dopamine_mesocortical: float = 0.21

    # Prolactin / care
    prolactin: float = 0.3

    # Composite / derived
    heart_rate_variability: float = 0.5
    respiration_rate: float = 0.3
    dmn_activity: float = 0.6

    # Receptor sensitivities (can be modulated by desensitization)
    d1_sensitivity: float = 1.0
    alpha1_sensitivity: float = 1.0
    _sensitivity_floor: float = 0.3

    # Finite neurotransmitter pools (deplete with use, recover with rest)
    dopamine_pool: float = 1.0
    serotonin_pool: float = 1.0
    norepinephrine_pool: float = 1.0
    acetylcholine_pool: float = 1.0
    gaba_pool: float = 1.0
    glutamate_pool: float = 1.0

    def clamp(self) -> "NeurochemicalState":
        return NeurochemicalState(
            serotonin=clamp01(self.serotonin),
            dopamine=clamp01(self.dopamine),
            norepinephrine=clamp01(self.norepinephrine),
            gaba=clamp01(self.gaba),
            glutamate=clamp01(self.glutamate),
            glycine=clamp01(self.glycine),
            acetylcholine=clamp01(self.acetylcholine),
            endocannabinoid=clamp01(self.endocannabinoid),
            oxytocin=clamp01(self.oxytocin),
            vasopressin=clamp01(self.vasopressin),
            cortisol=clamp01(self.cortisol),
            adrenaline=clamp01(self.adrenaline),
            opioid=clamp01(self.opioid),
            histamine=clamp01(self.histamine),
            melatonin=clamp01(self.melatonin),
            bdnf=clamp01(self.bdnf),
            nitric_oxide=clamp01(self.nitric_oxide),
            phenylethylamine=clamp01(self.phenylethylamine),
            tyramine=clamp01(self.tyramine),
            neuropeptide_s=clamp01(self.neuropeptide_s),
            orexin=clamp01(self.orexin),
            substance_p=clamp01(self.substance_p),
            cytokine_load=clamp01(self.cytokine_load),
            d2_autoreceptor_inhibition=clamp01(self.d2_autoreceptor_inhibition),
            heart_rate_variability=clamp01(self.heart_rate_variability),
            respiration_rate=clamp01(self.respiration_rate),
            kynurenine=clamp01(self.kynurenine),
            quinolinic_acid=clamp01(self.quinolinic_acid),
            picolinic_acid=clamp01(self.picolinic_acid),
            mao_activity=clamp01(self.mao_activity),
            dopamine_clearance_rate=clamp01(self.dopamine_clearance_rate),
            serotonin_reuptake=clamp01(self.serotonin_reuptake),
            gaba_a_sensitivity=clamp01(self.gaba_a_sensitivity),
            glun2b_sensitivity=clamp01(self.glun2b_sensitivity),
            respiration_phase=clamp01(self.respiration_phase),
            dopamine_mesolimbic=clamp01(self.dopamine_mesolimbic),
            dopamine_mesocortical=clamp01(self.dopamine_mesocortical),
            prolactin=clamp01(self.prolactin),
            dmn_activity=clamp01(self.dmn_activity),
            d1_sensitivity=clamp01(self.d1_sensitivity),
            alpha1_sensitivity=clamp01(self.alpha1_sensitivity),
            _sensitivity_floor=self._sensitivity_floor,
            dopamine_pool=clamp01(self.dopamine_pool),
            serotonin_pool=clamp01(self.serotonin_pool),
            norepinephrine_pool=clamp01(self.norepinephrine_pool),
            acetylcholine_pool=clamp01(self.acetylcholine_pool),
            gaba_pool=clamp01(self.gaba_pool),
            glutamate_pool=clamp01(self.glutamate_pool),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "NeurochemicalState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NeurochemistryEngine:
    """Update rules for neurochemical dynamics."""

    def __init__(self, state: NeurochemicalState | None = None):
        self.state = (state or NeurochemicalState()).clamp()

    def update(
        self,
        dt: float,
        appraisal_valence: float,
        appraisal_arousal: float,
        appraisal_dominance: float,
        drive_error_temperature: float,
        drive_rest_need: float,
        drive_task_load: float,
        drive_safety: float,
        surprise: float,
        circadian_hour: float = 12.0,
        metabolic_energy: float = 0.5,
    ) -> None:
        """Advance neurochemical state by one time step."""
        s = self.state

        # Normalize circadian hour to [0,1] with peak melatonin around 3 AM
        # 0 = midnight, 0.5 = noon, 1 = next midnight
        circadian_phase = (circadian_hour % 24.0) / 24.0
        # Distance from 3 AM (0.125)
        melatonin_phase = 1.0 - min(abs(circadian_phase - 0.125), 1.0 - abs(circadian_phase - 0.125)) * 4
        melatonin_phase = max(0.0, min(1.0, melatonin_phase))

        # --- Serotonin: stabilizes mood; low -> threat reactivity ---
        # Gradually recovers toward 0.5; stress lowers it
        serotonin_target = 0.5 - 0.3 * drive_error_temperature - 0.2 * (1 - drive_safety)
        s.serotonin = self._toward(s.serotonin, clamp01(serotonin_target), 0.05 * dt)

        # --- Dopamine: updates from reward prediction error ---
        # Depletes with use, recovers with rest/success
        dopamine_target = 0.3 + 0.4 * clamp01(surprise)
        # Pool availability limits effective dopamine
        pool_factor = s.dopamine_pool
        s.dopamine = self._toward(s.dopamine, dopamine_target * pool_factor, 0.04 * dt)
        # Deplete and recover pool
        if surprise > 0.2:
            s.dopamine_pool = max(0.2, s.dopamine_pool - 0.08 * dt)
        else:
            s.dopamine_pool = min(1.0, s.dopamine_pool + 0.03 * dt)

        # --- D2 autoreceptor short-loop feedback ---
        # D2 autoreceptors are slow sensors: they ramp up when dopamine is high
        # and decay when it falls, creating a persistent brake on further release.
        d2_target = clamp01((s.dopamine - 0.6) * 1.5)
        s.d2_autoreceptor_inhibition = self._toward(
            s.d2_autoreceptor_inhibition, d2_target, 0.12 * dt
        )
        s.dopamine = max(0.0, s.dopamine - s.d2_autoreceptor_inhibition * 0.15 * dt)
        s.dopamine_pool = min(1.0, s.dopamine_pool - s.d2_autoreceptor_inhibition * 0.05 * dt)

        # --- Norepinephrine: tonic + phasic ---
        ne_target = 0.2 + 0.3 * drive_task_load + 0.5 * clamp01(surprise)
        s.norepinephrine = self._toward(s.norepinephrine, ne_target * s.norepinephrine_pool, 0.08 * dt)
        if surprise > 0.2 or appraisal_arousal > 0.5:
            s.norepinephrine_pool = max(0.2, s.norepinephrine_pool - 0.08 * dt)
        else:
            s.norepinephrine_pool = min(1.0, s.norepinephrine_pool + 0.03 * dt)

        # --- Serotonin pool dynamics ---
        if drive_error_temperature > 0.5:
            s.serotonin_pool = max(0.2, s.serotonin_pool - 0.02 * dt)
        else:
            s.serotonin_pool = min(1.0, s.serotonin_pool + 0.01 * dt)
        s.serotonin = min(s.serotonin, s.serotonin_pool)

        # --- Acetylcholine pool dynamics ---
        ach_target = 0.4 + 0.4 * drive_task_load - 0.2 * clamp01(surprise)
        s.acetylcholine = self._toward(s.acetylcholine, ach_target * s.acetylcholine_pool, 0.05 * dt)
        if drive_task_load > 0.5:
            s.acetylcholine_pool = max(0.3, s.acetylcholine_pool - 0.03 * dt)
        else:
            s.acetylcholine_pool = min(1.0, s.acetylcholine_pool + 0.02 * dt)

        # --- GABA / Glutamate pools ---
        if appraisal_arousal > 0.5:
            s.glutamate_pool = max(0.3, s.glutamate_pool - 0.03 * dt)
            s.gaba_pool = min(1.0, s.gaba_pool + 0.01 * dt)
        else:
            s.glutamate_pool = min(1.0, s.glutamate_pool + 0.02 * dt)
            s.gaba_pool = max(0.3, s.gaba_pool - 0.01 * dt)
        s.glutamate = min(s.glutamate, s.glutamate_pool)
        s.gaba = min(s.gaba, s.gaba_pool)

        glu_target = 0.4 + 0.3 * appraisal_arousal + 0.2 * drive_task_load
        s.glutamate = self._toward(s.glutamate, glu_target, 0.05 * dt)

        # --- Glycine: inhibitory refinement, follows GABA with lag ---
        s.glycine = self._toward(s.glycine, s.gaba * 0.9, 0.03 * dt)

        # --- Acetylcholine: attention mode ---
        # High during focused tasks, low during exploration
        ach_target = 0.4 + 0.4 * drive_task_load - 0.2 * clamp01(surprise)
        s.acetylcholine = self._toward(s.acetylcholine, ach_target, 0.05 * dt)

        # --- Endocannabinoid: builds with arousal, calms next event ---
        ecb_target = 0.2 + 0.6 * appraisal_arousal
        s.endocannabinoid = self._toward(s.endocannabinoid, ecb_target, 0.06 * dt)

        # --- Oxytocin: social bonding ---
        oxy_target = 0.3 + 0.4 * clamp01(appraisal_valence) * (1 - drive_error_temperature)
        s.oxytocin = self._toward(s.oxytocin, oxy_target, 0.03 * dt)

        # --- Vasopressin: defense / control ---
        vaso_target = 0.2 + 0.5 * (1 - drive_safety) + 0.2 * abs(appraisal_dominance)
        s.vasopressin = self._toward(s.vasopressin, vaso_target, 0.04 * dt)

        # --- Cortisol: HPA-axis stress with slow decay ---
        # Add a circadian peak around 8 AM (phase 0.333)
        circadian_phase = (circadian_hour % 24.0) / 24.0
        car_peak = max(0.0, 1.0 - min(abs(circadian_phase - 0.333), 1.0 - abs(circadian_phase - 0.333)) * 6)
        cortisol_input = 0.1 + 0.15 * car_peak + 0.4 * drive_error_temperature + 0.2 * (1 - drive_safety)
        s.cortisol = self._toward(s.cortisol, cortisol_input, 0.015 * dt)

        # --- Adrenaline: acute fight-or-flight, fast decay ---
        # Strong valence-negative surprises produce a rapid, large adrenaline surge.
        valence_multiplier = 1.0 if appraisal_valence < -0.2 else 0.4
        adr_target = 0.0 + 1.2 * clamp01(surprise) * valence_multiplier
        s.adrenaline = self._toward(s.adrenaline, adr_target, 0.25 * dt)

        # --- Opioid: buffers pain, rises with comfort/safety ---
        op_target = 0.3 + 0.3 * clamp01(appraisal_valence) + 0.2 * drive_safety
        s.opioid = self._toward(s.opioid, op_target, 0.03 * dt)

        # --- Histamine: wakefulness ---
        his_target = 0.5 + 0.3 * drive_task_load - 0.3 * drive_rest_need
        # Lower histamine at night, higher during day
        circadian_wake = 1.0 - melatonin_phase
        his_target = 0.5 * circadian_wake + 0.5 * his_target
        # Metabolic energy raises histamine / orexin directly
        his_target += 0.15 * metabolic_energy
        s.histamine = self._toward(s.histamine, his_target, 0.04 * dt)

        # --- Melatonin: circadian ---
        mel_target = 0.2 + 0.5 * melatonin_phase + 0.1 * drive_rest_need - 0.1 * s.histamine
        s.melatonin = self._toward(s.melatonin, mel_target, 0.02 * dt)

        # --- BDNF: resilience/learning ---
        bdnf_target = 0.5 + 0.2 * clamp01(appraisal_valence) - 0.3 * s.cortisol
        s.bdnf = self._toward(s.bdnf, bdnf_target, 0.02 * dt)

        # --- Nitric oxide: local diffusion/spread ---
        no_target = 0.2 + 0.3 * appraisal_arousal
        s.nitric_oxide = self._toward(s.nitric_oxide, no_target, 0.05 * dt)

        # --- Trace amines ---
        pea_target = 0.2 + 0.4 * s.dopamine
        s.phenylethylamine = self._toward(s.phenylethylamine, pea_target, 0.04 * dt)
        tyr_target = 0.2 + 0.3 * s.norepinephrine
        s.tyramine = self._toward(s.tyramine, tyr_target, 0.04 * dt)

        # --- Neuropeptide S: alertness on novelty ---
        nps_target = 0.2 + 0.6 * clamp01(surprise)
        s.neuropeptide_s = self._toward(s.neuropeptide_s, nps_target, 0.08 * dt)

        # --- Orexin: wakefulness/motivation ---
        ox_target = 0.4 + 0.35 * s.histamine - 0.3 * drive_rest_need
        # Metabolic energy strongly raises orexin
        ox_target += 0.35 * metabolic_energy
        s.orexin = self._toward(s.orexin, ox_target, 0.15 * dt)

        # --- Substance P: pain salience under stress ---
        sp_target = 0.1 + 0.5 * s.cortisol + 0.2 * (1 - drive_safety)
        s.substance_p = self._toward(s.substance_p, sp_target, 0.03 * dt)

        # --- Cytokine load: neuroinflammation from chronic stress ---
        cyto_target = 0.0 + 0.5 * s.cortisol + 0.2 * drive_error_temperature
        s.cytokine_load = self._toward(s.cytokine_load, cyto_target, 0.01 * dt)

        # --- Kynurenine pathway: inflammation shunts tryptophan away from serotonin ---
        s.kynurenine = self._toward(s.kynurenine, 0.1 + 0.7 * s.cytokine_load, 0.2 * dt)
        s.quinolinic_acid = self._toward(s.quinolinic_acid, 0.0 + 0.6 * s.kynurenine, 0.2 * dt)
        s.picolinic_acid = self._toward(s.picolinic_acid, 0.1 + 0.3 * s.kynurenine, 0.2 * dt)
        glu_kyn = glu_target + 0.25 * s.quinolinic_acid
        s.glutamate = self._toward(s.glutamate, glu_kyn, 0.05 * dt)
        # Serotonin synthesis target is suppressed by high cytokine load
        serotonin_target = serotonin_target * (1 - 0.35 * s.cytokine_load)
        s.serotonin = self._toward(s.serotonin, clamp01(serotonin_target), 0.05 * dt)

        # --- Interoception ---
        hrv_target = 0.5 - 0.4 * s.adrenaline + 0.2 * s.oxytocin
        s.heart_rate_variability = self._toward(s.heart_rate_variability, hrv_target, 0.15 * dt)
        rr_target = 0.3 + 0.4 * s.norepinephrine
        # Respiration phase entrainment
        rr_target += 0.15 * math.sin(2 * math.pi * (s.respiration_phase + 0.25))
        s.respiration_rate = self._toward(s.respiration_rate, rr_target, 0.05 * dt)

        # --- Default mode network ---
        dmn_target = 0.6 * (1 - drive_task_load) + 0.2 * drive_rest_need
        s.dmn_activity = self._toward(s.dmn_activity, dmn_target, 0.03 * dt)


        # --- Enzymatic degradation (MAO + COMT + SERT) ---
        s.serotonin = max(0.0, s.serotonin - s.mao_activity * 0.05 * dt * s.serotonin)
        s.dopamine = max(0.0, s.dopamine - (s.mao_activity * 0.04 + s.dopamine_clearance_rate * 0.1) * dt * s.dopamine)
        s.norepinephrine = max(0.0, s.norepinephrine - s.mao_activity * 0.05 * dt * s.norepinephrine)
        s.serotonin = max(0.0, s.serotonin - s.serotonin_reuptake * 0.1 * dt * s.serotonin)

        # --- Dopamine pathway divergence ---
        s.dopamine_mesolimbic = self._toward(s.dopamine_mesolimbic, s.dopamine * 0.9, 0.05 * dt)
        s.dopamine_mesocortical = self._toward(
            s.dopamine_mesocortical, s.dopamine * 0.7 * s.dopamine_pool, 0.04 * dt
        )

        # --- Prolactin: rises with comfort/care, drops with dopamine ---
        prolactin_target = 0.3 + 0.3 * s.opioid - 0.2 * s.dopamine
        s.prolactin = self._toward(s.prolactin, clamp01(prolactin_target), 0.02 * dt)

        # Receptor desensitization: sustained high DA or NE, or chronically low pool
        if s.dopamine > 0.7 or s.dopamine_pool < 0.5:
            s.d1_sensitivity = max(s._sensitivity_floor, s.d1_sensitivity - 0.03 * dt)
        else:
            s.d1_sensitivity = min(1.0, s.d1_sensitivity + 0.01 * dt)

        if s.norepinephrine > 0.7 or s.norepinephrine_pool < 0.5:
            s.alpha1_sensitivity = max(s._sensitivity_floor, s.alpha1_sensitivity - 0.03 * dt)
        else:
            s.alpha1_sensitivity = min(1.0, s.alpha1_sensitivity + 0.01 * dt)

        # GABA-A and GluN2B desensitization
        if s.gaba > 0.7:
            s.gaba_a_sensitivity = max(s._sensitivity_floor, s.gaba_a_sensitivity - 0.02 * dt)
        else:
            s.gaba_a_sensitivity = min(1.0, s.gaba_a_sensitivity + 0.01 * dt)

        if s.glutamate > 0.7 or s.glun2b_sensitivity < 0.5:
            s.glun2b_sensitivity = max(s._sensitivity_floor, s.glun2b_sensitivity - 0.02 * dt)
        else:
            s.glun2b_sensitivity = min(1.0, s.glun2b_sensitivity + 0.01 * dt)

        self.state = s.clamp()

    def modulate_appraisal(
        self,
        raw_valence: float,
        raw_arousal: float,
        raw_dominance: float,
    ) -> Tuple[float, float, float]:
        """Apply neurochemical modulation to an appraisal delta."""
        s = self.state

        # GABA inhibition (scaled by GABA-A receptor sensitivity)
        inhibition = 1.0 - s.gaba * 0.5 * s.gaba_a_sensitivity
        # Glutamate excitation (scaled by GluN2B receptor sensitivity)
        excitation = 1.0 + s.glutamate * 0.5 * s.glun2b_sensitivity
        # Combined net gain
        net_gain = excitation * inhibition

        # Serotonin stabilizes: low serotonin amplifies, high serotonin damps
        serotonin_mod = 1.0 + (0.5 - s.serotonin)  # low sert -> x1.2, high -> x0.8

        # Oxytocin reduces threat reactivity
        threat_mod = 1.0 - s.oxytocin * 0.3

        # Cortisol / cytokine irritability
        irritability = 1.0 + s.cortisol * 0.3 + s.cytokine_load * 0.4

        # Adrenaline surge
        arousal_boost = 1.0 + s.adrenaline * 0.8

        # Opioid analgesia
        valence_buffer = 1.0 - s.opioid * 0.3

        # Vasopressin dominance defensiveness
        dominance_shift = s.vasopressin * 0.2

        valence = raw_valence * net_gain * serotonin_mod * threat_mod * valence_buffer
        arousal = raw_arousal * net_gain * arousal_boost * irritability
        dominance = raw_dominance * net_gain + dominance_shift

        return clamp11(valence), clamp01(arousal), clamp01(dominance)

    def allostatic_load(self) -> float:
        """Composite wear index across systems."""
        s = self.state
        return clamp01(
            s.cortisol * 0.3
            + s.cytokine_load * 0.25
            + s.adrenaline * 0.2
            + (1 - s.serotonin) * 0.15
            + (1 - s.heart_rate_variability) * 0.1
        )


    def compute_polyvagal_state(self, drive_safety: float) -> str:
        s = self.state
        if s.heart_rate_variability > 0.6 and s.oxytocin > 0.4 and drive_safety > 0.5:
            return "ventral_vagal"
        if s.adrenaline > 0.5 or s.norepinephrine > 0.7 or s.cortisol > 0.6:
            return "sympathetic"
        if s.heart_rate_variability < 0.25 and s.cortisol > 0.5:
            return "dorsal_vagal"
        return "mixed"

    def compute_lc_mode(self, task_load: float, surprise: float) -> str:
        s = self.state
        if s.norepinephrine > 0.7 and surprise > 0.3:
            return "phasic"
        if s.norepinephrine < 0.3 and task_load < 0.3:
            return "tonic"
        if task_load > 0.6:
            return "tonic"
        return "tonic"

    @staticmethod
    def _toward(current: float, target: float, amount: float) -> float:
        diff = target - current
        if abs(diff) <= amount:
            return target
        return current + (amount if diff > 0 else -amount)
