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

    # V3 additions
    crf: float = 0.0                      # corticotropin-releasing factor stress neuropeptide
    neuropeptide_y: float = 0.5         # NPY stress resilience
    dynorphin: float = 0.1               # kappa opioid dysphoria
    anandamide: float = 0.2              # endocannabinoid extinction gating
    faah_activity: float = 0.5           # FAAH anandamide degradation
    gat_activity: float = 0.5            # GABA transporter clearance
    glt1_activity: float = 0.7           # astrocyte glutamate reuptake
    glycogen: float = 0.8               # astrocyte glycogen reserve
    lactate: float = 0.2                 # astrocyte lactate shuttle
    microglia_state: float = 0.0         # primed neuroimmune state
    sleep_pressure: float = 0.0          # homeostatic sleep drive
    theta_gamma_coupling: float = 0.3    # oscillatory memory index

    # V4 additions
    vta_gaba: float = 0.2              # VTA interneuron brake on dopamine
    tmn_activity: float = 0.5          # tuberomammillary nucleus histamine source
    pbn_activity: float = 0.0          # parabrachial nucleus interoceptive relay
    rvlm_activity: float = 0.3         # rostral ventrolateral medulla sympathetic
    nts_activity: float = 0.3            # nucleus tractus solitarius vagal afferents
    fastigial_activity: float = 0.0    # cerebellar fastigial timing
    pvn_crf: float = 0.0               # paraventricular nucleus CRF output
    neurogenesis_rate: float = 0.5     # adult hippocampal neurogenesis
    bbb_permeability: float = 0.3      # blood-brain barrier permeability
    claustrum: float = 0.0             # salience gating / awareness
    scn_phase: float = 0.5             # suprachiasmatic nucleus circadian phase
    agrp: float = 0.3                  # arcuate AgRP hunger promoter
    pomc: float = 0.3                  # arcuate POMC satiety promoter
    il_activity: float = 0.3           # infralimbic vmPFC extinction
    pl_activity: float = 0.3           # prelimbic vmPFC fear expression
    bla: float = 0.0                   # basolateral amygdala sensory appraisal
    cea: float = 0.0                   # central amygdala fear output
    medial_habenula: float = 0.0       # value comparison / disappointment
    lc_mode: float = 0.0               # 0=tonic, 1=phasic locus coeruleus
    nucleus_reuniens: float = 0.0    # thalamic bridge hippocampus-PFC

    # V5 additions — 20 testable modules
    # 1. Cerebellar cortical layers
    purkinje_output: float = 0.2        # Purkinje cell inhibitory output
    climbing_fiber_error: float = 0.0   # Inferior olive error signal

    # 2. PAG columnar organization
    pag_dorsolateral: float = 0.0       # fight column
    pag_ventrolateral: float = 0.0      # freeze column
    pag_lateral: float = 0.0              # flight column
    pag_periaqueductal_gray: float = 0.0 # quiescence column

    # 3. Prefrontal working memory gating
    prefrontal_maintenance_bias: float = 0.5  # dlPFC D1/D2 balance
    ofc_reward_valuation: float = 0.5         # orbitofrontal expected value

    # 4. Testosterone / social dominance
    testosterone: float = 0.4             # androgen level

    # 5. Sleep architecture
    nrem_slow_wave: float = 0.0         # delta power during NREM
    rem_theta: float = 0.0              # theta power during REM

    # 6. Thermoregulation
    preoptic_warmth: float = 0.3        # preoptic area temperature sensing
    brown_adipose_activity: float = 0.2 # thermogenesis
    body_temperature: float = 0.5        # core temperature

    # 7. Thalamic relay nuclei
    md_thalamus: float = 0.2            # mediodorsal thalamus (WM gating)
    pulvinar: float = 0.1               # pulvinar (attention salience)

    # 8. Theta rhythm generation
    medial_septum: float = 0.2          # septal theta pacemaker
    hippocampal_theta: float = 0.2      # theta oscillation power
    grid_cell_modulation: float = 0.1   # entorhinal grid cell firing

    # 9. Glymphatic system
    glymphatic_flow: float = 0.2        # CSF-ISF exchange rate
    amyloid_beta: float = 0.3           # amyloid-beta accumulation
    aquaporin_4: float = 0.4            # astrocyte water channel

    # 10. Gut-brain axis
    vagal_afferent: float = 0.3         # vagus nerve gut signaling
    butyrate: float = 0.2               # SCFA from microbiome

    # 11. Estrogen / progesterone
    estrogen: float = 0.5               # estradiol level
    progesterone: float = 0.4           # progesterone level
    allopregnanolone: float = 0.2       # GABA-A modulating metabolite

    # 12. Hypoxic response
    oxygen_saturation: float = 0.95     # blood oxygen
    adenosine: float = 0.2              # protective adenosine surge
    hif1_alpha: float = 0.0             # hypoxia-inducible factor

    # 13. Descending pain control
    rvmm_activity: float = 0.2          # rostral ventromedial medulla
    spinal_opioid: float = 0.2          # spinal cord opioid release
    ab_fiber: float = 0.3               # Aβ touch/pressure fiber

    # 14. Hedonic hotspots
    nacc_shell_liking: float = 0.2      # μ-opioid hedonic pleasure

    # 15. Mast cell-neuroimmune
    mast_cell_activation: float = 0.0   # mast cell degranulation

    # 16. Amino acid precursors
    tryptophan: float = 0.5             # serotonin precursor
    tyrosine: float = 0.5               # dopamine/NE precursor

    # 17. Prepulse inhibition
    startle_response: float = 0.2       # acoustic startle magnitude
    prepulse_inhibition: float = 0.5     # PPI gating ratio

    # 18. Synaptic plasticity
    ltp_threshold: float = 0.5           # BCM sliding threshold
    synaptic_change: float = 0.0        # net LTP/LTD direction

    # 19. Mitochondrial bioenergetics
    mitochondrial_atp: float = 0.7      # cellular ATP
    reactive_oxygen_species: float = 0.1 # ROS oxidative stress

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
            crf=clamp01(self.crf),
            neuropeptide_y=clamp01(self.neuropeptide_y),
            dynorphin=clamp01(self.dynorphin),
            anandamide=clamp01(self.anandamide),
            faah_activity=clamp01(self.faah_activity),
            gat_activity=clamp01(self.gat_activity),
            glt1_activity=clamp01(self.glt1_activity),
            glycogen=clamp01(self.glycogen),
            lactate=clamp01(self.lactate),
            microglia_state=clamp01(self.microglia_state),
            sleep_pressure=clamp01(self.sleep_pressure),
            theta_gamma_coupling=clamp01(self.theta_gamma_coupling),
            vta_gaba=clamp01(self.vta_gaba),
            tmn_activity=clamp01(self.tmn_activity),
            pbn_activity=clamp01(self.pbn_activity),
            rvlm_activity=clamp01(self.rvlm_activity),
            nts_activity=clamp01(self.nts_activity),
            fastigial_activity=clamp01(self.fastigial_activity),
            pvn_crf=clamp01(self.pvn_crf),
            neurogenesis_rate=clamp01(self.neurogenesis_rate),
            bbb_permeability=clamp01(self.bbb_permeability),
            claustrum=clamp01(self.claustrum),
            scn_phase=clamp01(self.scn_phase),
            agrp=clamp01(self.agrp),
            pomc=clamp01(self.pomc),
            il_activity=clamp01(self.il_activity),
            pl_activity=clamp01(self.pl_activity),
            bla=clamp01(self.bla),
            cea=clamp01(self.cea),
            medial_habenula=clamp01(self.medial_habenula),
            lc_mode=clamp01(self.lc_mode),
            nucleus_reuniens=clamp01(self.nucleus_reuniens),
            # V5 additions
            purkinje_output=clamp01(self.purkinje_output),
            climbing_fiber_error=clamp01(self.climbing_fiber_error),
            pag_dorsolateral=clamp01(self.pag_dorsolateral),
            pag_ventrolateral=clamp01(self.pag_ventrolateral),
            pag_lateral=clamp01(self.pag_lateral),
            pag_periaqueductal_gray=clamp01(self.pag_periaqueductal_gray),
            prefrontal_maintenance_bias=clamp01(self.prefrontal_maintenance_bias),
            ofc_reward_valuation=clamp01(self.ofc_reward_valuation),
            testosterone=clamp01(self.testosterone),
            nrem_slow_wave=clamp01(self.nrem_slow_wave),
            rem_theta=clamp01(self.rem_theta),
            preoptic_warmth=clamp01(self.preoptic_warmth),
            brown_adipose_activity=clamp01(self.brown_adipose_activity),
            body_temperature=clamp01(self.body_temperature),
            md_thalamus=clamp01(self.md_thalamus),
            pulvinar=clamp01(self.pulvinar),
            medial_septum=clamp01(self.medial_septum),
            hippocampal_theta=clamp01(self.hippocampal_theta),
            grid_cell_modulation=clamp01(self.grid_cell_modulation),
            glymphatic_flow=clamp01(self.glymphatic_flow),
            amyloid_beta=clamp01(self.amyloid_beta),
            aquaporin_4=clamp01(self.aquaporin_4),
            vagal_afferent=clamp01(self.vagal_afferent),
            butyrate=clamp01(self.butyrate),
            estrogen=clamp01(self.estrogen),
            progesterone=clamp01(self.progesterone),
            allopregnanolone=clamp01(self.allopregnanolone),
            oxygen_saturation=clamp01(self.oxygen_saturation),
            adenosine=clamp01(self.adenosine),
            hif1_alpha=clamp01(self.hif1_alpha),
            rvmm_activity=clamp01(self.rvmm_activity),
            spinal_opioid=clamp01(self.spinal_opioid),
            ab_fiber=clamp01(self.ab_fiber),
            nacc_shell_liking=clamp01(self.nacc_shell_liking),
            mast_cell_activation=clamp01(self.mast_cell_activation),
            tryptophan=clamp01(self.tryptophan),
            tyrosine=clamp01(self.tyrosine),
            startle_response=clamp01(self.startle_response),
            prepulse_inhibition=clamp01(self.prepulse_inhibition),
            ltp_threshold=clamp01(self.ltp_threshold),
            synaptic_change=clamp01(self.synaptic_change),
            mitochondrial_atp=clamp01(self.mitochondrial_atp),
            reactive_oxygen_species=clamp01(self.reactive_oxygen_species),
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
        glucose: float = 0.5,
        expected_reward: float = 0.0,
        novelty: float = 0.0,
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
        dopamine_target = 0.35 + 0.55 * clamp01(surprise)
        # Pool availability limits effective dopamine
        pool_factor = s.dopamine_pool
        s.dopamine = self._toward(s.dopamine, dopamine_target * pool_factor, 0.05 * dt)
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
        cortisol_input = 0.1 + 0.25 * car_peak + 0.4 * drive_error_temperature + 0.05 * (1 - drive_safety) + 0.3 * s.crf
        s.cortisol = self._toward(s.cortisol, cortisol_input, 0.05 * dt)

        # --- Adrenaline: acute fight-or-flight, fast decay ---
        # Strong valence-negative surprises produce a rapid, large adrenaline surge.
        # CRF pre-sensitization amplifies the adrenal response.
        valence_multiplier = 1.0 if appraisal_valence < -0.2 else 0.4
        adr_target = 0.0 + (1.2 + 0.8 * s.crf) * clamp01(surprise) * valence_multiplier
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
        cyto_target = (
            0.0
            + 0.5 * s.cortisol
            + 0.2 * drive_error_temperature
            + 0.25 * clamp01(-appraisal_valence) * surprise
            + 0.15 * (1 - drive_safety)
        )
        s.cytokine_load = self._toward(s.cytokine_load, cyto_target, 0.03 * dt)

        # --- Kynurenine pathway: inflammation shunts tryptophan away from serotonin ---
        s.kynurenine = self._toward(s.kynurenine, 0.1 + 0.7 * s.cytokine_load, 0.2 * dt)
        s.quinolinic_acid = self._toward(s.quinolinic_acid, 0.0 + 0.6 * s.kynurenine, 0.2 * dt)
        s.picolinic_acid = self._toward(s.picolinic_acid, 0.1 + 0.3 * s.kynurenine, 0.2 * dt)
        glu_kyn = glu_target + 0.45 * s.quinolinic_acid
        s.glutamate = self._toward(s.glutamate, glu_kyn, 0.08 * dt)
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
        if s.dopamine > 0.6 or s.dopamine_pool < 0.5:
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

        # --- V3 additions ---------------------------------------------------

        # CRF: sustained stress neuropeptide; amplified by uncertainty and cortisol
        crf_target = 0.1 + 0.4 * (1 - drive_safety) + 0.4 * s.cortisol + 0.3 * drive_error_temperature
        s.crf = self._toward(s.crf, crf_target, 0.08 * dt)
        # Oxytocin and NPY suppress CRF
        s.crf = max(0.0, s.crf - (s.oxytocin * 0.02 + s.neuropeptide_y * 0.03) * dt)

        # Neuropeptide Y: resilience; rises with safety/success and suppresses anxiety circuits
        npy_target = 0.3 + 0.4 * drive_safety + 0.2 * s.opioid - 0.2 * s.crf
        s.neuropeptide_y = self._toward(s.neuropeptide_y, npy_target, 0.04 * dt)

        # Dynorphin / kappa opioid: counter-reward on negative surprise
        dyn_target = 0.1 + 0.7 * clamp01(-appraisal_valence) * clamp01(surprise)
        s.dynorphin = self._toward(s.dynorphin, dyn_target, 0.08 * dt)
        # Dynorphin actively suppresses dopamine / seeking
        s.dopamine = max(0.0, s.dopamine - s.dynorphin * 0.1 * dt)
        s.dopamine_mesolimbic = max(0.0, s.dopamine_mesolimbic - s.dynorphin * 0.08 * dt)

        # Anandamide / FAAH extinction gating
        # Anandamide rises with arousal and is degraded by FAAH
        ana_target = 0.2 + 0.5 * appraisal_arousal + 0.2 * s.endocannabinoid
        s.anandamide = self._toward(s.anandamide, ana_target, 0.06 * dt)
        s.anandamide = max(0.0, s.anandamide - s.faah_activity * 0.15 * dt)
        # FAAH itself rises with chronic stress
        s.faah_activity = self._toward(s.faah_activity, 0.3 + 0.4 * s.cortisol, 0.02 * dt)

        # GABA transporter (GAT) tone: high GAT clears GABA faster
        s.gaba = max(0.0, s.gaba - s.gat_activity * 0.15 * dt)
        s.gat_activity = self._toward(s.gat_activity, 0.4 + 0.3 * s.cortisol, 0.02 * dt)

        # Astrocyte GLT-1 clears glutamate; low GLT-1 raises excitotoxicity risk
        glt_clear = s.glt1_activity * 0.05 * dt
        s.glutamate = max(0.0, s.glutamate - glt_clear)
        s.glt1_activity = self._toward(s.glt1_activity, 0.5 + 0.3 * s.bdnf - 0.2 * s.cytokine_load, 0.02 * dt)

        # Glycogen-lactate shuttle: task load consumes glycogen, produces lactate
        if drive_task_load > 0.4:
            s.glycogen = max(0.1, s.glycogen - 0.04 * drive_task_load * dt)
            s.lactate = min(1.0, s.lactate + 0.05 * drive_task_load * dt)
        else:
            s.glycogen = min(1.0, s.glycogen + 0.03 * dt)
            s.lactate = max(0.0, s.lactate - 0.04 * dt)

        # Microglial priming: neuroimmune memory sensitizes future cytokine responses
        micro_target = 0.0 + 0.6 * s.cytokine_load + 0.3 * s.cortisol
        s.microglia_state = self._toward(s.microglia_state, micro_target, 0.02 * dt)
        # Primed microglia amplify new cytokine spikes
        if s.microglia_state > 0.2 and (surprise > 0.2 or drive_error_temperature > 0.3):
            s.cytokine_load = min(1.0, s.cytokine_load + 0.08 * s.microglia_state * dt)

        # Sleep pressure: rises with rest_need and time awake, reduced by rest
        sleep_target = drive_rest_need * 0.8 + 0.1 * (1 - s.melatonin)
        s.sleep_pressure = self._toward(s.sleep_pressure, sleep_target, 0.03 * dt)

        # Orexin flip: high sleep pressure turns orexin from wake-promoting to transition
        if s.sleep_pressure > 0.7:
            s.histamine = self._toward(s.histamine, s.histamine * 0.7, 0.05 * dt)

        # Theta-gamma coupling: rises with ACh and moderate NE, drops with very high arousal
        tgc_target = 0.2 + 0.5 * s.acetylcholine + 0.3 * s.norepinephrine * (1 - s.norepinephrine)
        s.theta_gamma_coupling = self._toward(s.theta_gamma_coupling, clamp01(tgc_target), 0.04 * dt)

        # ---- V4 additions ----
        # SCN master clock: phase advances with time; entrains with circadian_hour
        phase_target = (circadian_hour % 24.0) / 24.0
        s.scn_phase = self._toward(s.scn_phase, phase_target, 0.10 * dt)

        # BLA sensory appraisal: driven by glutamate + noradrenaline from salient input
        bla_target = 0.1 + 0.5 * s.glutamate + 0.3 * s.norepinephrine
        s.bla = self._toward(s.bla, clamp01(bla_target), 0.15 * dt)

        # CeA fear output: driven by BLA + low GABA
        cea_target = 0.2 + s.bla * 1.5 - s.gaba * 0.5
        s.cea = self._toward(s.cea, clamp01(cea_target), 0.20 * dt)

        # IL extinction / PL fear expression: IL rises with safety and extinction learning
        il_target = 0.2 + 0.4 * drive_safety + 0.3 * s.bdnf - 0.2 * s.cortisol
        s.il_activity = self._toward(s.il_activity, clamp01(il_target), 0.15 * dt)
        pl_target = 0.2 + 0.4 * (1 - drive_safety) + 0.3 * s.cortisol + 0.2 * s.bla
        s.pl_activity = self._toward(s.pl_activity, clamp01(pl_target), 0.15 * dt)

        # Locus coeruleus mode: flip toward phasic on high salience, tonic on low arousal
        lc_target = 1.0 if surprise > 0.5 or s.adrenaline > 0.3 else 0.0
        s.lc_mode = self._toward(s.lc_mode, lc_target, 0.60 * dt)

        # Hunger circuit: AgRP rises with low glucose, POMC with high metabolic energy
        agrp_target = 0.3 + 0.6 * max(0, 0.6 - glucose) - 0.3 * metabolic_energy
        s.agrp = self._toward(s.agrp, clamp01(agrp_target), 0.15 * dt)
        pomc_target = 0.3 + 0.5 * metabolic_energy - 0.2 * agrp_target
        s.pomc = self._toward(s.pomc, clamp01(pomc_target), 0.15 * dt)

        # VTA GABA interneuron brake: rises with BNST/RMTg activation
        vta_gaba_target = 0.2 + 0.3 * s.crf + 0.3 * s.dynorphin + 0.2 * (1 - drive_safety)
        s.vta_gaba = self._toward(s.vta_gaba, clamp01(vta_gaba_target), 0.15 * dt)

        # Medial habenula value comparison: rises when expected reward > actual dopamine
        mh_target = max(0, expected_reward - s.dopamine) * 0.8
        s.medial_habenula = self._toward(s.medial_habenula, clamp01(mh_target), 0.20 * dt)

        # Nucleus reuniens: bridges hippocampus-PFC during rest/theta-gamma
        nr_target = 0.1 + 0.5 * s.theta_gamma_coupling + 0.3 * s.bdnf - 0.2 * s.cortisol
        s.nucleus_reuniens = self._toward(s.nucleus_reuniens, clamp01(nr_target), 0.15 * dt)

        # Claustrum salience gating: rises with novel salient events, decays with habituation
        claustrum_target = 0.1 + 0.5 * novelty - 0.2 * s.sleep_pressure - 0.3 * s.claustrum
        s.claustrum = self._toward(s.claustrum, clamp01(claustrum_target), 0.20 * dt)

        # TMN histamine source: inverse to melatonin and sleep pressure
        tmn_target = 0.5 + 0.3 * (1 - s.melatonin) - 0.3 * s.sleep_pressure + 0.2 * s.orexin
        s.tmn_activity = self._toward(s.tmn_activity, clamp01(tmn_target), 0.15 * dt)
        # TMN drives histamine; sleep pressure directly suppresses it
        histamine_target = 0.4 + 0.4 * s.tmn_activity - 0.2 * s.melatonin - 0.4 * s.sleep_pressure
        s.histamine = self._toward(s.histamine, clamp01(histamine_target), 0.04 * dt)

        # Parabrachial nucleus: interoceptive relay for cytokine + pain
        pbn_target = 0.1 + 0.4 * s.cytokine_load + 0.4 * s.substance_p + 0.2 * s.cortisol
        s.pbn_activity = self._toward(s.pbn_activity, clamp01(pbn_target), 0.20 * dt)

        # RVLM sympathetic tone: drives NE and adrenaline; inhibited by high HRV
        rvlm_target = 0.3 + 0.3 * (1 - drive_safety) + 0.2 * s.cortisol - 0.3 * s.heart_rate_variability
        s.rvlm_activity = self._toward(s.rvlm_activity, clamp01(rvlm_target), 0.15 * dt)
        # RVLM drives sympathetic output
        s.norepinephrine = self._toward(s.norepinephrine, s.norepinephrine + 0.05 * s.rvlm_activity, 0.02 * dt)
        s.adrenaline = self._toward(s.adrenaline, s.adrenaline + 0.05 * s.rvlm_activity, 0.02 * dt)

        # NTS vagal afferents: integrates gut/metabolic signals
        nts_target = 0.3 + 0.3 * s.cytokine_load + 0.2 * (1 - metabolic_energy) + 0.2 * s.pbn_activity
        s.nts_activity = self._toward(s.nts_activity, clamp01(nts_target), 0.15 * dt)

        # Fastigial cerebellar timing: rises with predictable inter-event intervals
        if hasattr(self, '_last_event_time') and dt > 0:
            interval = dt
            expected_interval = getattr(self, '_expected_interval', 2.0)
            # If interval is close to expected, boost fastigial
            interval_match = max(0, 1.0 - abs(interval - expected_interval) / max(expected_interval, 0.5))
            fastigial_target = 0.2 + 0.6 * interval_match
            self._expected_interval = 0.8 * expected_interval + 0.2 * interval
        else:
            fastigial_target = 0.1
            self._last_event_time = 0.0
            self._expected_interval = 2.0
        s.fastigial_activity = self._toward(s.fastigial_activity, clamp01(fastigial_target), 0.20 * dt)

        # PVN stress integration: CRF output gated by amygdala, BNST, NTS
        pvn_target = 0.1 + 0.3 * s.bla + 0.3 * s.crf + 0.2 * s.nts_activity + 0.2 * (1 - drive_safety)
        s.pvn_crf = self._toward(s.pvn_crf, clamp01(pvn_target), 0.20 * dt)

        # Adult neurogenesis: high BDNF + low cortisol + low cytokine boosts it
        ng_target = 0.5 + 0.4 * s.bdnf - 0.4 * s.cortisol - 0.3 * s.cytokine_load
        s.neurogenesis_rate = self._toward(s.neurogenesis_rate, clamp01(ng_target), 0.15 * dt)

        # Blood-brain barrier: chronic stress increases permeability; BDNF tightens it
        bbb_target = 0.3 + 0.4 * s.cortisol + 0.3 * s.cytokine_load - 0.3 * s.bdnf
        s.bbb_permeability = self._toward(s.bbb_permeability, clamp01(bbb_target), 0.15 * dt)

        # -----------------------------------------------------------------------
        # V5 additions — 20 testable modules
        # -----------------------------------------------------------------------

        # 1. Cerebellar cortical layers
        # Climbing fiber carries motor error signal
        cf_target = 0.1 + 0.8 * drive_error_temperature
        s.climbing_fiber_error = self._toward(s.climbing_fiber_error, clamp01(cf_target), 0.30 * dt)
        # Purkinje output rises with climbing fiber error, suppresses fastigial
        purkinje_target = 0.2 + 0.7 * s.climbing_fiber_error
        s.purkinje_output = self._toward(s.purkinje_output, clamp01(purkinje_target), 0.18 * dt)
        s.fastigial_activity = max(0.0, s.fastigial_activity - s.purkinje_output * 0.25 * dt)

        # 2. PAG columnar organization
        # Dorsolateral = fight (high dominance + threat)
        pag_dl = 0.1 + 0.7 * clamp01(appraisal_dominance) + 0.3 * (1 - drive_safety)
        s.pag_dorsolateral = self._toward(s.pag_dorsolateral, clamp01(pag_dl), 0.25 * dt)
        # Ventrolateral = freeze (high threat, low dominance)
        pag_vl = 0.1 + 0.6 * (1 - drive_safety) - 0.3 * clamp01(appraisal_dominance)
        s.pag_ventrolateral = self._toward(s.pag_ventrolateral, clamp01(pag_vl), 0.15 * dt)
        # Lateral = flight (high arousal, moderate threat)
        pag_lat = 0.1 + 0.5 * appraisal_arousal * (1 - drive_safety)
        s.pag_lateral = self._toward(s.pag_lateral, clamp01(pag_lat), 0.15 * dt)
        # Periaqueductal gray = quiescence (high safety)
        pag_q = 0.1 + 0.5 * drive_safety
        s.pag_periaqueductal_gray = self._toward(s.pag_periaqueductal_gray, clamp01(pag_q), 0.10 * dt)

        # 3. Prefrontal working memory gating
        # dlPFC maintenance bias: high mesocortical DA + working memory load
        pfc_target = 0.3 + 0.5 * s.dopamine_mesocortical * s.d1_sensitivity + 0.3 * drive_task_load
        s.prefrontal_maintenance_bias = self._toward(s.prefrontal_maintenance_bias, clamp01(pfc_target), 0.10 * dt)
        # OFC reward valuation tracks expected reward vs actual dopamine
        ofc_target = 0.3 + 0.4 * expected_reward + 0.3 * s.dopamine
        s.ofc_reward_valuation = self._toward(s.ofc_reward_valuation, clamp01(ofc_target), 0.08 * dt)

        # 4. Testosterone / social dominance
        # Testosterone rises with social victories and high dominance
        t_target = 0.3 + 0.3 * s.dopamine + 0.3 * clamp01(appraisal_dominance) + 0.2 * drive_safety - 0.4 * drive_error_temperature
        s.testosterone = self._toward(s.testosterone, clamp01(t_target), 0.08 * dt)
        # Testosterone suppresses fear circuits
        s.cea = max(0.0, s.cea - s.testosterone * 0.15 * dt)
        s.bla = max(0.0, s.bla - s.testosterone * 0.08 * dt)
        # Social defeat (repeated errors) lowers T and BDNF
        if drive_error_temperature > 0.5:
            s.testosterone = max(0.0, s.testosterone - 0.08 * dt)
            s.bdnf = max(0.0, s.bdnf - 0.05 * dt)

        # 5. Sleep architecture
        # NREM delta during high sleep pressure + melatonin
        nrem_target = 0.0 + 0.6 * s.sleep_pressure + 0.4 * s.melatonin - 0.2 * s.histamine
        s.nrem_slow_wave = self._toward(s.nrem_slow_wave, clamp01(nrem_target), 0.12 * dt)
        # REM theta: high ACh, low NE, moderate sleep pressure
        rem_target = 0.0 + 0.4 * s.acetylcholine + 0.3 * s.sleep_pressure - 0.4 * s.norepinephrine
        s.rem_theta = self._toward(s.rem_theta, clamp01(rem_target), 0.10 * dt)
        # REM selectively boosts dopamine
        if s.rem_theta > 0.3:
            s.dopamine = min(1.0, s.dopamine + 0.02 * dt)

        # 6. Thermoregulation
        # Preoptic warmth sensing: body temperature + metabolic heat
        po_target = 0.3 + 0.5 * s.body_temperature + 0.3 * metabolic_energy
        s.preoptic_warmth = self._toward(s.preoptic_warmth, clamp01(po_target), 0.12 * dt)
        # High PO warmth suppresses wake circuits
        if s.preoptic_warmth > 0.4:
            s.orexin = max(0.0, s.orexin - 0.12 * dt)
            s.histamine = max(0.0, s.histamine - 0.08 * dt)
        # Brown adipose thermogenesis: NE drives heat production
        bat_target = 0.2 + 0.6 * s.norepinephrine - 0.3 * s.body_temperature
        s.brown_adipose_activity = self._toward(s.brown_adipose_activity, clamp01(bat_target), 0.12 * dt)
        s.body_temperature = self._toward(s.body_temperature, 0.5 + 0.35 * s.brown_adipose_activity - 0.2 * s.preoptic_warmth, 0.06 * dt)

        # 7. Thalamic relay nuclei
        # MD thalamus gates working memory
        md_target = 0.2 + 0.5 * drive_task_load + 0.3 * s.acetylcholine
        s.md_thalamus = self._toward(s.md_thalamus, clamp01(md_target), 0.10 * dt)
        # Pulvinar gates attention salience
        pulvinar_target = 0.1 + 0.5 * novelty + 0.3 * s.norepinephrine
        s.pulvinar = self._toward(s.pulvinar, clamp01(pulvinar_target), 0.12 * dt)

        # 8. Theta rhythm generation
        # Medial septum paces hippocampal theta
        septum_target = 0.2 + 0.5 * s.acetylcholine + 0.3 * s.gaba
        s.medial_septum = self._toward(s.medial_septum, clamp01(septum_target), 0.10 * dt)
        # Hippocampal theta
        theta_target = 0.2 + 0.6 * s.medial_septum + 0.3 * s.theta_gamma_coupling - 0.2 * s.cortisol
        s.hippocampal_theta = self._toward(s.hippocampal_theta, clamp01(theta_target), 0.08 * dt)
        # Grid cell modulation by theta
        grid_target = 0.1 + 0.5 * s.hippocampal_theta + 0.3 * s.theta_gamma_coupling
        s.grid_cell_modulation = self._toward(s.grid_cell_modulation, clamp01(grid_target), 0.08 * dt)

        # 9. Glymphatic system
        # Aquaporin-4: low NE during sleep promotes channel opening
        aqp4_target = 0.3 + 0.5 * s.nrem_slow_wave - 0.3 * s.norepinephrine
        s.aquaporin_4 = self._toward(s.aquaporin_4, clamp01(aqp4_target), 0.08 * dt)
        # Glymphatic flow: sleep + aquaporin drive clearance
        gf_target = 0.2 + 0.5 * s.aquaporin_4 + 0.3 * s.nrem_slow_wave
        s.glymphatic_flow = self._toward(s.glymphatic_flow, clamp01(gf_target), 0.10 * dt)
        # Amyloid-beta clearance by glymphatic flow
        s.amyloid_beta = max(0.0, s.amyloid_beta - s.glymphatic_flow * 0.08 * dt)
        # BBB permeability also allows amyloid entry
        s.amyloid_beta = min(1.0, s.amyloid_beta + s.bbb_permeability * 0.02 * dt)

        # 10. Gut-brain axis
        # Vagal afferent signaling from gut
        vagal_target = 0.3 + 0.3 * s.cytokine_load + 0.2 * s.butyrate - 0.2 * s.cortisol
        s.vagal_afferent = self._toward(s.vagal_afferent, clamp01(vagal_target), 0.08 * dt)
        # Vagal stimulation suppresses HPA axis
        s.cortisol = max(0.0, s.cortisol - s.vagal_afferent * 0.02 * dt)
        s.crf = max(0.0, s.crf - s.vagal_afferent * 0.02 * dt)
        # Butyrate enhances GABAergic tone
        s.gaba = min(1.0, s.gaba + s.butyrate * 0.10 * dt)

        # 11. Estrogen / progesterone
        # Estrogen modulates serotonin and BDNF
        estrogen_target = 0.5 + 0.1 * math.sin(circadian_hour * math.pi / 12)  # minimal diurnal
        s.estrogen = self._toward(s.estrogen, clamp01(estrogen_target), 0.02 * dt)
        s.bdnf = min(1.0, s.bdnf + s.estrogen * 0.03 * dt)
        s.serotonin = min(1.0, s.serotonin + s.estrogen * 0.02 * dt)
        # Progesterone → allopregnanolone → GABA-A enhancement
        prog_target = 0.4 + 0.2 * math.sin(circadian_hour * math.pi / 12 + math.pi / 4)
        s.progesterone = self._toward(s.progesterone, clamp01(prog_target), 0.04 * dt)
        allo_target = 0.2 + 0.6 * s.progesterone
        s.allopregnanolone = self._toward(s.allopregnanolone, clamp01(allo_target), 0.10 * dt)
        s.gaba_a_sensitivity = min(2.0, s.gaba_a_sensitivity + s.allopregnanolone * 0.08 * dt)

        # 12. Hypoxic response
        # Adenosine rises as oxygen falls (protective)
        adenosine_target = 0.2 + 0.8 * max(0.0, 0.6 - s.oxygen_saturation)
        s.adenosine = self._toward(s.adenosine, clamp01(adenosine_target), 0.10 * dt)
        # Adenosine suppresses glutamate (prevents excitotoxicity)
        s.glutamate = max(0.0, s.glutamate - s.adenosine * 0.08 * dt)
        # HIF-1α activates under sustained hypoxia
        hif_target = 0.0 + 0.8 * max(0.0, 0.5 - s.oxygen_saturation)
        s.hif1_alpha = self._toward(s.hif1_alpha, clamp01(hif_target), 0.08 * dt)
        # Normal oxygen recovery
        o2_target = 0.95 - 0.3 * s.cortisol - 0.2 * drive_task_load
        s.oxygen_saturation = self._toward(s.oxygen_saturation, clamp01(o2_target), 0.02 * dt)

        # 13. Descending pain control
        # RVMM activated by PAG during opioid analgesia
        rvmm_target = 0.2 + 0.4 * s.pag_periaqueductal_gray + 0.3 * s.opioid
        s.rvmm_activity = self._toward(s.rvmm_activity, clamp01(rvmm_target), 0.10 * dt)
        # Spinal opioid release gates pain
        spinal_op_target = 0.2 + 0.5 * s.rvmm_activity
        s.spinal_opioid = self._toward(s.spinal_opioid, clamp01(spinal_op_target), 0.10 * dt)
        s.substance_p = max(0.0, s.substance_p - s.spinal_opioid * 0.12 * dt)
        # Gate control: Aβ fibers inhibit pain
        s.substance_p = max(0.0, s.substance_p - s.ab_fiber * 0.10 * dt)

        # 14. Hedonic hotspots (NAcc shell liking vs wanting)
        liking_target = 0.2 + 0.6 * s.opioid
        s.nacc_shell_liking = self._toward(s.nacc_shell_liking, clamp01(liking_target), 0.10 * dt)

        # 15. Mast cell-neuroimmune
        # Mast cells degranulate under IgE-like stress signals
        mast_target = 0.0 + 0.5 * s.cytokine_load + 0.3 * drive_error_temperature + 0.2 * s.cortisol
        s.mast_cell_activation = self._toward(s.mast_cell_activation, clamp01(mast_target), 0.08 * dt)
        # Mast cell releases histamine and sensitizes pain
        s.histamine = min(1.0, s.histamine + s.mast_cell_activation * 0.10 * dt)
        s.substance_p = min(1.0, s.substance_p + s.mast_cell_activation * 0.06 * dt)

        # 16. Amino acid precursor competition
        # Tryptophan → serotonin; depletion lowers 5-HT
        trp_target = 0.5 - 0.2 * s.cytokine_load  # inflammation shunts Trp to kynurenine
        s.tryptophan = self._toward(s.tryptophan, clamp01(trp_target), 0.05 * dt)
        s.serotonin = self._toward(s.serotonin, clamp01(0.3 + 0.5 * s.tryptophan), 0.05 * dt)
        # Tyrosine → dopamine/NE; high tyrosine favors catecholamines
        tyr_target = 0.5 + 0.1 * drive_task_load
        s.tyrosine = self._toward(s.tyrosine, clamp01(tyr_target), 0.05 * dt)
        s.dopamine = min(1.0, s.dopamine + s.tyrosine * 0.03 * dt)
        s.norepinephrine = min(1.0, s.norepinephrine + s.tyrosine * 0.02 * dt)

        # 17. Prepulse inhibition
        # Startle rises with high NE and low safety
        startle_target = 0.2 + 0.5 * s.norepinephrine + 0.3 * (1 - drive_safety)
        s.startle_response = self._toward(s.startle_response, clamp01(startle_target), 0.15 * dt)
        # Prepulse reduces startle via gating
        s.startle_response = max(0.0, s.startle_response - s.prepulse_inhibition * 0.5 * dt)
        # PPI itself improves with practice
        ppi_target = 0.5 + 0.2 * s.gaba
        s.prepulse_inhibition = self._toward(s.prepulse_inhibition, clamp01(ppi_target), 0.06 * dt)

        # 18. Synaptic plasticity (BCM metaplasticity)
        # Activity history sets sliding LTP threshold
        activity_history = s.dopamine * 0.4 + s.acetylcholine * 0.3 + drive_task_load * 0.3
        ltp_thresh_target = 0.4 + 0.4 * activity_history
        s.ltp_threshold = self._toward(s.ltp_threshold, clamp01(ltp_thresh_target), 0.05 * dt)
        # Weak activation below threshold → LTD; strong → LTP
        activation_strength = s.glutamate * s.glun2b_sensitivity
        if activation_strength < s.ltp_threshold * 0.7:
            s.synaptic_change = self._toward(s.synaptic_change, -0.5, 0.05 * dt)
        elif activation_strength > s.ltp_threshold * 1.3:
            s.synaptic_change = self._toward(s.synaptic_change, 0.5, 0.05 * dt)
        else:
            s.synaptic_change = self._toward(s.synaptic_change, 0.0, 0.03 * dt)

        # 19. Mitochondrial bioenergetics
        # ATP drops with sustained task load and stress
        atp_target = 0.7 - 0.5 * drive_task_load - 0.3 * s.cortisol + 0.1 * metabolic_energy
        s.mitochondrial_atp = self._toward(s.mitochondrial_atp, clamp01(atp_target), 0.10 * dt)
        # ROS rises when ATP drops and calcium is high
        ros_target = 0.1 + 0.5 * max(0.0, 0.6 - s.mitochondrial_atp) + 0.2 * s.glutamate
        s.reactive_oxygen_species = self._toward(s.reactive_oxygen_species, clamp01(ros_target), 0.08 * dt)
        # Mitochondrial dysfunction increases excitotoxicity risk
        if s.mitochondrial_atp < 0.3 and s.glutamate > 0.6:
            s.reactive_oxygen_species = min(1.0, s.reactive_oxygen_species + 0.15 * dt)

        # 20. Neuropeptide transmission characteristics (volume transmission slower)
        # Already captured by oxytocin/dynorphin slow decay — tested separately

        self.state = s.clamp()

    def excitotoxicity_risk(self) -> float:
        s = self.state
        return clamp01(
            s.glutamate * 0.4
            + s.quinolinic_acid * 0.35
            + (1 - s.glt1_activity) * 0.2
            - s.gaba * 0.1
            + max(0.0, 0.6 - s.mitochondrial_atp) * 0.3  # V5: mitochondrial dysfunction
        )

    def rmtg_brake(self) -> float:
        """GABAergic brake on dopamine during aversion."""
        s = self.state
        return clamp01(s.dynorphin * 0.5 + s.crf * 0.3 + (1 - s.dopamine) * 0.2)

    def bnst_state(self, drive_safety: float) -> Dict[str, float]:
        s = self.state
        apprehension = clamp01(
            s.crf * 0.5 + (1 - drive_safety) * 0.4 + s.cytokine_load * 0.2 - s.neuropeptide_y * 0.3
        )
        return {"crf": s.crf, "apprehension": apprehension}

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
