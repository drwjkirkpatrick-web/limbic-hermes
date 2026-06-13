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
            heart_rate_variability=clamp01(self.heart_rate_variability),
            respiration_rate=clamp01(self.respiration_rate),
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

        # Effective levels respect pool ceilings
        s.dopamine = min(s.dopamine, s.dopamine_pool)
        s.norepinephrine = min(s.norepinephrine, s.norepinephrine_pool)
        s.acetylcholine = min(s.acetylcholine, s.acetylcholine_pool)

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
        cortisol_input = 0.1 + 0.4 * drive_error_temperature + 0.2 * (1 - drive_safety)
        s.cortisol = self._toward(s.cortisol, cortisol_input, 0.015 * dt)

        # --- Adrenaline: acute fight-or-flight, fast decay ---
        adr_target = 0.0 + 0.8 * clamp01(surprise) * (1 if appraisal_valence < -0.2 else 0.3)
        s.adrenaline = self._toward(s.adrenaline, adr_target, 0.12 * dt)

        # --- Opioid: buffers pain, rises with comfort/safety ---
        op_target = 0.3 + 0.3 * clamp01(appraisal_valence) + 0.2 * drive_safety
        s.opioid = self._toward(s.opioid, op_target, 0.03 * dt)

        # --- Histamine: wakefulness ---
        his_target = 0.5 + 0.3 * drive_task_load - 0.3 * drive_rest_need
        # Lower histamine at night, higher during day
        circadian_wake = 1.0 - melatonin_phase
        his_target = 0.5 * circadian_wake + 0.5 * his_target
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
        ox_target = 0.5 + 0.2 * s.histamine - 0.3 * drive_rest_need
        s.orexin = self._toward(s.orexin, ox_target, 0.04 * dt)

        # --- Substance P: pain salience under stress ---
        sp_target = 0.1 + 0.5 * s.cortisol + 0.2 * (1 - drive_safety)
        s.substance_p = self._toward(s.substance_p, sp_target, 0.03 * dt)

        # --- Cytokine load: neuroinflammation from chronic stress ---
        cyto_target = 0.0 + 0.5 * s.cortisol + 0.2 * drive_error_temperature
        s.cytokine_load = self._toward(s.cytokine_load, cyto_target, 0.01 * dt)

        # --- Interoception ---
        hrv_target = 0.5 - 0.3 * s.adrenaline + 0.2 * s.oxytocin
        s.heart_rate_variability = self._toward(s.heart_rate_variability, hrv_target, 0.05 * dt)
        rr_target = 0.3 + 0.4 * s.norepinephrine
        s.respiration_rate = self._toward(s.respiration_rate, rr_target, 0.05 * dt)

        # --- Default mode network ---
        dmn_target = 0.6 * (1 - drive_task_load) + 0.2 * drive_rest_need
        s.dmn_activity = self._toward(s.dmn_activity, dmn_target, 0.03 * dt)

        # Receptor desensitization: sustained high DA or NE, or chronically low pool
        if s.dopamine > 0.7 or s.dopamine_pool < 0.5:
            s.d1_sensitivity = max(s._sensitivity_floor, s.d1_sensitivity - 0.03 * dt)
        else:
            s.d1_sensitivity = min(1.0, s.d1_sensitivity + 0.01 * dt)

        if s.norepinephrine > 0.7 or s.norepinephrine_pool < 0.5:
            s.alpha1_sensitivity = max(s._sensitivity_floor, s.alpha1_sensitivity - 0.03 * dt)
        else:
            s.alpha1_sensitivity = min(1.0, s.alpha1_sensitivity + 0.01 * dt)

        self.state = s.clamp()

    def modulate_appraisal(
        self,
        raw_valence: float,
        raw_arousal: float,
        raw_dominance: float,
    ) -> Tuple[float, float, float]:
        """Apply neurochemical modulation to an appraisal delta."""
        s = self.state

        # GABA inhibition
        inhibition = 1.0 - s.gaba * 0.5
        # Glutamate excitation
        excitation = 1.0 + s.glutamate * 0.5
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

    @staticmethod
    def _toward(current: float, target: float, amount: float) -> float:
        diff = target - current
        if abs(diff) <= amount:
            return target
        return current + (amount if diff > 0 else -amount)
