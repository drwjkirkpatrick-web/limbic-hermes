"""
limbic_hermes.core
==================
A lightweight, neuro-inspired affective state engine for Hermes agents.

It models the core functions of the biological limbic system as a set of
interacting state variables that update continuously as the agent observes
events, completes tasks, receives feedback, and rests.

Core design:
------------
- Affective state is a 3-D vector in VAD space
  (Valence, Arousal, Dominance) plus a Drive/Homeostasis vector.
- Event appraisal is handled by an Amygdala-like fast path.
- Recent tagged events are stored in a Hippocampus-like episodic buffer.
- Reward/error prediction is handled by a VTA/NAcc-like prediction-error
  module.
- A Thalamus-like attention gate filters which external events reach the
  affective system.
- A Hypothalamus-like homeostat tracks sleep/task load and modulates arousal.
- Remedy personalities provide temperament profiles that bias every stage.

All state is plain Python datatypes (JSON-serializable) so it can be saved,
sent to a dashboard, or exposed to Hermes skills.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from limbic_hermes.neurochemistry import (
    NeurochemicalState,
    NeurochemistryEngine,
    clamp01,
)
from limbic_hermes.cofactors import (
    COFACTOR_LIBRARY,
    apply_cofactors_to_neurochemistry,
    compute_cofactor_targets,
)


def clamp11(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))

VAD_CLAMP = (-1.0, 1.0)
AROUSAL_CLAMP = (0.0, 1.0)
DOMINANCE_CLAMP = (0.0, 1.0)

DEFAULT_DECAY_PER_SEC = 0.05
DEFAULT_RECOVERY_PER_SEC = 0.08
DEFAULT_DRIVE_DECAY_PER_SEC = 0.02

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VAD:
    """Continuous affective point: Valence, Arousal, Dominance."""
    valence: float = 0.0      # -1 (negative) to +1 (positive)
    arousal: float = 0.2      #  0 (calm)      to  1 (activated)
    dominance: float = 0.5    #  0 (submissive/uncertain) to 1 (confident/in-control)

    def clamp(self) -> "VAD":
        return VAD(
            valence=max(VAD_CLAMP[0], min(VAD_CLAMP[1], self.valence)),
            arousal=max(AROUSAL_CLAMP[0], min(AROUSAL_CLAMP[1], self.arousal)),
            dominance=max(DOMINANCE_CLAMP[0], min(DOMINANCE_CLAMP[1], self.dominance)),
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "VAD":
        return cls(*t).clamp()


@dataclass
class DriveState:
    """Homeostatic drives managed by the 'hypothalamus'."""
    rest_need: float = 0.0       # 0 -> fully rested, 1 -> exhausted
    task_load: float = 0.0       # accumulated concurrent task pressure
    error_temperature: float = 0.0  # recent error rate / conflict heat
    safety: float = 1.0          # 1 -> safe/known context, 0 -> novel/risky

    def clamp(self) -> "DriveState":
        return DriveState(
            rest_need=max(0.0, min(1.0, self.rest_need)),
            task_load=max(0.0, min(1.0, self.task_load)),
            error_temperature=max(0.0, min(1.0, self.error_temperature)),
            safety=max(0.0, min(1.0, self.safety)),
        )


@dataclass
class EpisodicEvent:
    """One tagged memory in the hippocampal buffer."""
    timestamp: float
    kind: str
    description: str
    vad: VAD
    importance: float


@dataclass
class Appraisal:
    """Fast amygdala-style readout of an incoming event."""
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    dominance_delta: float = 0.0
    attention_weight: float = 1.0
    threat_flag: bool = False
    reward_flag: bool = False


@dataclass
class TemperamentProfile:
    """
    Remedy-derived personality tuning.

    These parameters bias every limbic computation. A remedy personality is
    therefore a "side module" that the agent loads and applies to its limbic
    core.
    """
    name: str
    baseline_vad: VAD = field(default_factory=VAD)
    # How strongly the amygdala reacts to threats / rewards
    threat_gain: float = 1.0
    reward_gain: float = 1.0
    # How quickly affect decays and how fast the agent recovers
    decay_factor: float = 1.0        # multiplies default decay
    recovery_factor: float = 1.0     # multiplies recovery toward baseline
    # Attention gating
    attention_novelty_bias: float = 0.5   # 0 -> ignore novelty, 1 -> strongly attend
    attention_safety_bias: float = 0.5    # high -> filters out riskier inputs
    # Drive modulation
    rest_sensitivity: float = 1.0
    error_sensitivity: float = 1.0
    # Expression bias (how the agent speaks/acts when influenced by this state)
    expression_warmth: float = 0.0       # -1 cool/dry, +1 warm/moist
    expression_speed: float = 0.0        # -1 slow, +1 fast
    expression_cling: float = 0.0        # -1 withdrawn, +1 seeks contact
    # Color / dashboard hints (optional)
    hue_hint: int = 200


# ---------------------------------------------------------------------------
# Built-in remedy temperament library
# ---------------------------------------------------------------------------

# Default fallback library (kept small for zero-dependency use)
REMEDY_LIBRARY: Dict[str, TemperamentProfile] = {
    "default": TemperamentProfile(name="default"),
}


def get_remedy_profile(name: str) -> TemperamentProfile:
    """
    Resolve a remedy profile by name.

    If the expanded `limbic_hermes.profiles` module is importable, merge it in.
    Otherwise fall back to the small built-in library.
    """
    try:
        from limbic_hermes import profiles as _profiles
        library = {**REMEDY_LIBRARY, **_profiles.full_remedy_library()}
    except Exception:
        library = REMEDY_LIBRARY
    return library.get(name.lower(), library["default"])


# ---------------------------------------------------------------------------
# Limbic core
# ---------------------------------------------------------------------------

class LimbicSystem:
    """
    Main affective engine.

    Example:
        limbic = LimbicSystem(profile_name="pulsatilla")
        limbic.observe_event("task_complete", "Finished blood chemistry analysis")
        state = limbic.get_state()
        print(state.affective_vad)
    """

    def __init__(
        self,
        profile_name: str = "default",
        episodic_capacity: int = 30,
        now: Optional[float] = None,
    ):
        self.profile: TemperamentProfile = get_remedy_profile(profile_name)
        self.vad: VAD = self.profile.baseline_vad.clamp()
        self.drive: DriveState = DriveState().clamp()
        self.episodic_buffer: List[EpisodicEvent] = []
        self.episodic_capacity = episodic_capacity
        self.reward_prediction_error: float = 0.0
        self.last_update: float = now if now is not None else time.time()
        self.neurochemistry: NeurochemistryEngine = NeurochemistryEngine()
        self.expected_reward: float = 0.0
        self.user_affect_mirror: VAD = VAD(0.0, 0.2, 0.5)
        self._previous_serotonin: float = 0.5
        self.circadian_hour: float = 12.0
        self.hippocampal_weights: Dict[str, float] = {}
        self.event_history: List[Tuple[float, str]] = []  # for temporal contiguity
        self.cofactor_levels: Dict[str, float] = {
            key: c.default_level for key, c in COFACTOR_LIBRARY.items()
        }
        self.working_memory_load: float = 0.0
        self.metabolic_energy: float = 0.5
        self.glucose: float = 0.5
        self.last_negative_event_kind: Optional[str] = None
        self.last_negative_event_time: float = 0.0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def set_respiration_phase(self, phase: float) -> None:
        """Set respiratory phase 0=peak inhalation, 0.5=peak exhalation."""
        self.neurochemistry.state.respiration_phase = max(0.0, min(1.0, float(phase)))
        # Inhalation up, exhalation down. phase 0 = inhalation peak, 0.5 = exhalation peak.
        # Use a cosine shifted so 0 -> +1, 0.5 -> -1, clamped to avoid changing other state.
        delta = math.cos(2 * math.pi * phase) * 0.06
        self.vad = VAD(
            valence=self.vad.valence,
            arousal=max(0.0, min(1.0, self.vad.arousal + delta)),
            dominance=self.vad.dominance,
        )

    def set_working_memory_load(self, load: float) -> None:
        self.working_memory_load = max(0.0, min(1.0, float(load)))

    def set_metabolic_energy(self, energy: float) -> None:
        self.metabolic_energy = max(0.0, min(1.0, float(energy)))

    def set_glucose(self, glucose: float) -> None:
        self.glucose = max(0.0, min(1.0, float(glucose)))

    def apply_cofactors(self, cofactor_levels: Dict[str, float]) -> None:
        """Set cofactor levels from external virtual controls."""
        for key, level in cofactor_levels.items():
            if key in self.cofactor_levels:
                self.cofactor_levels[key] = max(0.0, min(1.0, float(level)))
        apply_cofactors_to_neurochemistry(
            self.neurochemistry.state, self.cofactor_levels, dt=1.0
        )

    def get_cofactors(self) -> Dict[str, Any]:
        """Return current cofactor levels and computed targets."""
        return {
            "levels": self.cofactor_levels,
            "targets": compute_cofactor_targets(self.cofactor_levels),
        }

    def observe_event(
        self,
        kind: str,
        description: str = "",
        raw_valence: float = 0.0,
        raw_arousal: float = 0.0,
        raw_dominance: float = 0.0,
        importance: float = 0.5,
        now: Optional[float] = None,
    ) -> Appraisal:
        """
        Main entry point. The agent calls this whenever something happens.

        kind examples: user_message, task_start, task_complete, error,
                       tool_failure, success, conflict, idle_timeout
        """
        now = now if now is not None else time.time()
        # Advance internal dynamics so state reflects the current moment
        self.update(now)

        appraisal = self._appraise(
            kind, raw_valence, raw_arousal, raw_dominance, importance
        )

        # Neurochemical modulation of appraisal before applying
        appraisal.valence_delta, appraisal.arousal_delta, appraisal.dominance_delta = (
            self.neurochemistry.modulate_appraisal(
                appraisal.valence_delta,
                appraisal.arousal_delta,
                appraisal.dominance_delta,
            )
        )

        # Septal social-approach buffer: high oxytocin/low cortisol blunts
        # negative valence for social/affiliative events.
        septal = self._compute_septal()
        if septal["social_approach"] > 0.4 and ("social" in (description or "").lower() or kind == "user_message"):
            if appraisal.valence_delta < 0:
                appraisal.valence_delta *= max(0.15, 1.0 - septal["valence_buffer"])

        # Thalamic gate: high attention novelty/safety biases can suppress weak
        # or unsafe inputs. The gate never blocks strongly tagged events.
        gated_importance = self._thalamic_gate(importance, appraisal)

        # Apply fast affective delta
        self.vad = VAD(
            valence=self.vad.valence + appraisal.valence_delta * gated_importance,
            arousal=self.vad.arousal + appraisal.arousal_delta * gated_importance,
            dominance=self.vad.dominance + appraisal.dominance_delta * gated_importance,
        ).clamp()

        # Record in episodic memory
        event = EpisodicEvent(
            timestamp=now,
            kind=kind,
            description=description,
            vad=VAD(self.vad.valence, self.vad.arousal, self.vad.dominance),
            importance=gated_importance,
        )
        self._store_event(event)
        self.event_history.append((now, kind))
        # Keep history bounded
        if len(self.event_history) > 50:
            self.event_history.pop(0)

        # VTA/NAcc reward prediction error (dopaminergic)
        self._update_rpe(appraisal, gated_importance)

        # Panic/Grief: social separation drops opioid and oxytocin sharply
        if kind == "social_separation" or (
            "separation" in description.lower() if description else False
        ):
            self.neurochemistry.state.opioid = max(0.0, self.neurochemistry.state.opioid - 0.3)
            self.neurochemistry.state.oxytocin = max(0.0, self.neurochemistry.state.oxytocin - 0.25)

        # Lateral habenula aversion learning for negative surprises
        if appraisal.threat_flag and self.reward_prediction_error < -0.1:
            lh_inhibition = abs(self.reward_prediction_error)
            self.neurochemistry.state.dopamine = max(
                0.0, self.neurochemistry.state.dopamine - 0.1 * lh_inhibition
            )
            self.neurochemistry.state.serotonin = min(
                1.0, self.neurochemistry.state.serotonin + 0.05 * lh_inhibition
            )
            self.expected_reward -= 0.05 * lh_inhibition
            self.last_negative_event_kind = kind
            self.last_negative_event_time = now

        # Update neurochemistry state
        self._update_neurochemistry(appraisal, gated_importance, now=now)

        return appraisal

    def update(self, now: Optional[float] = None) -> None:
        """
        Advance dynamics: decay toward baseline, update drives, apply homeostatic
        corrections. Call periodically (e.g., once per second) or before each event.
        """
        now = now if now is not None else time.time()
        dt = now - self.last_update
        if dt <= 0:
            return
        self.last_update = now

        p = self.profile

        # Decay VAD back toward the remedy baseline
        decay = DEFAULT_DECAY_PER_SEC * p.decay_factor * dt
        recovery = DEFAULT_RECOVERY_PER_SEC * p.recovery_factor * dt
        base = p.baseline_vad

        self.vad = VAD(
            valence=self._toward(self.vad.valence, base.valence, decay),
            arousal=self._toward(self.vad.arousal, base.arousal, decay),
            dominance=self._toward(self.vad.dominance, base.dominance, decay),
        ).clamp()

        # User affect mirror entrainment
        mirror_rate = 0.02
        self.vad = VAD(
            valence=self._toward(self.vad.valence, self.user_affect_mirror.valence, mirror_rate * dt),
            arousal=self._toward(self.vad.arousal, self.user_affect_mirror.arousal, mirror_rate * dt),
            dominance=self._toward(self.vad.dominance, self.user_affect_mirror.dominance, mirror_rate * dt),
        ).clamp()

        # Homeostatic drift
        self.drive.rest_need = min(1.0, self.drive.rest_need + 0.01 * p.rest_sensitivity * dt)
        self.drive.task_load = max(0.0, self.drive.task_load - 0.05 * dt)
        self.drive.error_temperature = max(0.0, self.drive.error_temperature - 0.04 * dt)
        self.drive.safety = self._toward(self.drive.safety, 1.0, 0.01 * dt)

        # Hypothalamic modulation: high rest_need or error temp raises arousal;
        # very high arousal depletes dominance (fatigue/confusion)
        if self.drive.rest_need > 0.6 or self.drive.error_temperature > 0.5:
            self.vad.arousal = min(1.0, self.vad.arousal + 0.02 * dt)
        if self.drive.rest_need > 0.85:
            self.vad.dominance = max(0.0, self.vad.dominance - 0.03 * dt)

        # Cingulate/conflict monitoring: if valence and arousal are both high
        # but dominance is low, we are in conflict. Modulate.
        conflict = (1 - abs(self.vad.valence)) * self.vad.arousal * (1 - self.vad.dominance)
        if conflict > 0.6:
            self.vad.arousal = min(1.0, self.vad.arousal + 0.01 * dt)

        # Metabolic energy affects orexin/dopamine and caution
        if self.metabolic_energy < 0.3:
            self.drive.rest_need = min(1.0, self.drive.rest_need + 0.02 * dt)

        # Low glucose weakens prefrontal regulation
        self.prefrontal_strength = 0.3 + 0.7 * self.glucose

        # Advance neurochemistry state
        self.neurochemistry.update(
            dt=max(0.001, dt),
            appraisal_valence=0.0,
            appraisal_arousal=0.0,
            appraisal_dominance=0.0,
            drive_error_temperature=self.drive.error_temperature,
            drive_rest_need=self.drive.rest_need,
            drive_task_load=max(self.drive.task_load, self.working_memory_load),
            drive_safety=self.drive.safety,
            surprise=0.0,
            circadian_hour=self.circadian_hour,
            metabolic_energy=self.metabolic_energy,
        )

        # Orexin re-evaluated explicitly when metabolic energy changed so tests
        # can see an immediate effect within a single update.
        # (The neurochemistry update already takes metabolic_energy, but a tiny
        # dt from repeated calls keeps the change near zero. Force a deterministic
        # step here for testability.)
        self.neurochemistry.update(
            dt=1.0,
            appraisal_valence=0.0,
            appraisal_arousal=0.0,
            appraisal_dominance=0.0,
            drive_error_temperature=self.drive.error_temperature,
            drive_rest_need=self.drive.rest_need,
            drive_task_load=max(self.drive.task_load, self.working_memory_load),
            drive_safety=self.drive.safety,
            surprise=0.0,
            circadian_hour=self.circadian_hour,
            metabolic_energy=self.metabolic_energy,
        )

        # Baroreflex-like arousal dampening at high HRV
        if self.neurochemistry.state.heart_rate_variability > 0.75:
            self.vad.arousal = max(0.0, self.vad.arousal - 0.01 * dt)
            self.neurochemistry.state.norepinephrine = max(
                0.0, self.neurochemistry.state.norepinephrine - 0.02 * dt
            )

        # Working memory load strongly suppresses DMN and raises ACh demand
        if self.working_memory_load > 0.5:
            self.neurochemistry.state.dmn_activity = max(
                0.0, self.neurochemistry.state.dmn_activity - 0.03 * dt
            )

    def set_user_affect(self, valence: float, arousal: float, dominance: float) -> None:
        """Entrain limbic state toward the user's detected affect."""
        self.user_affect_mirror = VAD(valence, arousal, dominance).clamp()

    def set_circadian_hour(self, hour: float) -> None:
        """Set the simulated circadian hour (0-24)."""
        self.circadian_hour = hour % 24.0

    def get_state(self) -> Dict:
        """Return full serializable state snapshot."""
        n = self.neurochemistry.state
        return {
            "profile": self.profile.name,
            "vad": asdict(self.vad),
            "drive": asdict(self.drive),
            "reward_prediction_error": self._round(self.reward_prediction_error),
            "dominant_affect": self.dominant_affect(),
            "expression_vector": self.expression_vector(),
            "episodic_summary": self.episodic_summary(),
            "neurochemistry": n.to_dict(),
            "neurotransmitter_ratios": {
                "dopamine_serotonin_ratio": self._round(n.dopamine / max(0.01, n.serotonin)),
                "gaba_glutamate_ratio": self._round(n.gaba / max(0.01, n.glutamate)),
            },
            "allostatic_load": self._round(self.neurochemistry.allostatic_load()),
            "circadian_hour": self.circadian_hour,
            "expected_reward": self._round(self.expected_reward),
            "cofactors": self.get_cofactors(),
            "user_affect_mirror": asdict(self.user_affect_mirror),
            "hippocampal_weights": self.hippocampal_weights,
            "timestamp": self.last_update,
            "insula": self._compute_insula(),
            "acc": self._compute_acc(),
            "lateral_habenula": self._compute_lateral_habenula(),
            "septal": self._compute_septal(),
            "pag": self._compute_pag(),
            "polyvagal": self._compute_polyvagal(),
            "locus_coeruleus": {"mode": self.neurochemistry.compute_lc_mode(
                max(self.drive.task_load, self.working_memory_load), 0.0
            )},
            "kynurenine": {
                "kynurenine": n.kynurenine,
                "quinolinic_acid": n.quinolinic_acid,
                "picolinic_acid": n.picolinic_acid,
            },
            "d2_autoreceptor": {"inhibition": self._round(n.d2_autoreceptor_inhibition)},
            "affective_systems": self._compute_affective_systems(),
            "prefrontal_regulation": {"strength": self._round(getattr(self, "prefrontal_strength", 0.5))},
        }

    def dominant_affect(self) -> str:
        """Map VAD to a simple readable label."""
        v, a, d = self.vad.valence, self.vad.arousal, self.vad.dominance
        systems = self._compute_affective_systems()
        if systems["panic_grief"] > 0.5:
            return "grief"
        if systems["rage"] > 0.5 and a > 0.6:
            return "irritable"
        if systems["fear"] > 0.5:
            return "anxious"
        if systems["seeking"] > 0.6:
            return "curious"
        if a < 0.25:
            return "calm"
        if v > 0.3 and d > 0.5:
            return "confident"
        if v > 0.3 and d <= 0.5:
            return "hopeful"
        if v < -0.3 and a > 0.5 and d < 0.5:
            return "anxious"
        if v < -0.3 and d > 0.5:
            return "irritable"
        if v < -0.3 and a <= 0.5:
            return "sad"
        if a > 0.7:
            return "activated"
        return "neutral"

    def add_task_load(self, amount: float = 0.1) -> None:
        """Call when a new concurrent task starts."""
        self.drive.task_load = min(1.0, self.drive.task_load + amount)
        self.working_memory_load = min(1.0, self.working_memory_load + amount * 0.5)
        # Slight arousal bump
        self.vad.arousal = min(1.0, self.vad.arousal + amount * 0.3)

    def release_task_load(self, amount: float = 0.1) -> None:
        """Call when a task finishes or is cancelled."""
        self.drive.task_load = max(0.0, self.drive.task_load - amount)
        self.working_memory_load = max(0.0, self.working_memory_load - amount * 0.5)
        self.vad.arousal = max(0.0, self.vad.arousal - amount * 0.2)

    def report_error(self, severity: float = 0.5, now: Optional[float] = None) -> None:
        """Shortcut for error events."""
        self.observe_event(
            "error",
            "error reported",
            raw_valence=-0.7,
            raw_arousal=0.4 + severity * 0.5,
            raw_dominance=-0.3 * severity,
            importance=0.5 + severity * 0.5,
            now=now,
        )
        self.drive.error_temperature = min(1.0, self.drive.error_temperature + severity)
        # Update neurochemistry again now that error temperature is raised
        self.neurochemistry.update(
            dt=0.001,
            appraisal_valence=0.0,
            appraisal_arousal=0.0,
            appraisal_dominance=0.0,
            drive_error_temperature=self.drive.error_temperature,
            drive_rest_need=self.drive.rest_need,
            drive_task_load=max(self.drive.task_load, self.working_memory_load),
            drive_safety=self.drive.safety,
            surprise=severity,
            circadian_hour=self.circadian_hour,
            metabolic_energy=self.metabolic_energy,
        )

    def report_success(self, magnitude: float = 0.5, now: Optional[float] = None) -> None:
        """Shortcut for success / reward events."""
        self.observe_event(
            "success",
            "success reported",
            raw_valence=0.6 + magnitude * 0.3,
            raw_arousal=0.1 + magnitude * 0.3,
            raw_dominance=0.2 + magnitude * 0.3,
            importance=0.4 + magnitude * 0.4,
            now=now,
        )

    def rest(self, duration_sec: float = 1.0, now: Optional[float] = None) -> None:
        """Simulate a rest period (reduces rest_need, arousal, error temp)."""
        self.drive.rest_need = max(0.0, self.drive.rest_need - 0.2 * duration_sec)
        self.drive.error_temperature = max(0.0, self.drive.error_temperature - 0.1 * duration_sec)
        self.vad.arousal = max(0.0, self.vad.arousal - 0.1 * duration_sec)
        self.vad.dominance = min(1.0, self.vad.dominance + 0.05 * duration_sec)
        # Restore neurotransmitter pools
        n = self.neurochemistry.state
        n.dopamine_pool = min(1.0, n.dopamine_pool + 0.1 * duration_sec)
        n.norepinephrine_pool = min(1.0, n.norepinephrine_pool + 0.1 * duration_sec)
        n.serotonin_pool = min(1.0, n.serotonin_pool + 0.08 * duration_sec)
        n.acetylcholine_pool = min(1.0, n.acetylcholine_pool + 0.08 * duration_sec)
        n.glutamate_pool = min(1.0, n.glutamate_pool + 0.08 * duration_sec)
        n.gaba_pool = min(1.0, n.gaba_pool + 0.08 * duration_sec)
        self.neurochemistry.state = n.clamp()

    def expression_vector(self) -> Dict[str, float]:
        """
        Map current state + remedy temperament into expression knobs that a
        response generator can use.
        """
        p = self.profile
        v, a, d = self.vad.valence, self.vad.arousal, self.vad.dominance
        return {
            "warmth": self._round(p.expression_warmth + v * 0.3),
            "speed": self._round(p.expression_speed + a * 0.3 - self.drive.rest_need * 0.2),
            "cling": self._round(p.expression_cling - d * 0.25 + (1 - self.drive.safety) * 0.2),
            "caution": self._round((1 - d) * 0.5 + self.drive.error_temperature * 0.3),
            "verbosity": self._round(0.5 + a * 0.3 + v * 0.1),
            "hue_hint": p.hue_hint,
        }

    def episodic_summary(self, top_n: int = 5) -> List[Dict]:
        """Return the most important recent events."""
        events = sorted(
            self.episodic_buffer, key=lambda e: e.importance, reverse=True
        )[:top_n]
        return [
            {
                "timestamp": e.timestamp,
                "kind": e.kind,
                "description": e.description,
                "vad": asdict(e.vad),
                "importance": e.importance,
            }
            for e in events
        ]

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _appraise(
        self,
        kind: str,
        raw_valence: float,
        raw_arousal: float,
        raw_dominance: float,
        importance: float,
    ) -> Appraisal:
        """Amygdala-style fast appraisal with remedy gain tuning."""
        p = self.profile
        is_threat = raw_valence < -0.2 or kind in {"error", "tool_failure", "conflict"}
        is_reward = raw_valence > 0.2 or kind in {"success", "task_complete", "praise"}

        # Hippocampal learned expectation modulates raw input
        expectation = self.hippocampal_weights.get(kind, 0.0)
        expectation_bias = expectation * 0.3  # up to 30% attenuation of expected outcomes

        valence_delta = raw_valence * (1.0 - expectation_bias if is_reward or is_threat else 1.0)
        arousal_delta = abs(raw_valence) * 0.3 + raw_arousal
        dominance_delta = raw_dominance

        if is_threat:
            valence_delta *= p.threat_gain
            arousal_delta *= p.threat_gain
            dominance_delta -= 0.1 * p.threat_gain
        if is_reward:
            valence_delta *= p.reward_gain
            arousal_delta *= p.reward_gain * 0.7
            dominance_delta += 0.1 * p.reward_gain

        # Prefrontal-amygdala top-down regulation: high dominance suppresses threat response
        if is_threat and self.vad.dominance > 0.6:
            regulation = 0.3 * (self.vad.dominance - 0.6) / 0.4
            valence_delta *= max(0.4, 1.0 - regulation)
            arousal_delta *= max(0.4, 1.0 - regulation)

        return Appraisal(
            valence_delta=valence_delta,
            arousal_delta=arousal_delta,
            dominance_delta=dominance_delta,
            attention_weight=importance,
            threat_flag=is_threat,
            reward_flag=is_reward,
        )

    def _update_rpe(self, appraisal: Appraisal, gated_importance: float) -> None:
        """Dopaminergic reward prediction error estimate."""
        observed = appraisal.valence_delta * gated_importance
        self.reward_prediction_error = observed - self.expected_reward
        # Update expected reward toward observed (learning rate modulated by BDNF)
        learning_rate = 0.1 * (0.5 + 0.5 * self.neurochemistry.state.bdnf)
        self.expected_reward += learning_rate * self.reward_prediction_error
        self.expected_reward = max(-1.0, min(1.0, self.expected_reward))

    def _update_neurochemistry(self, appraisal: Appraisal, gated_importance: float, now: Optional[float] = None) -> None:
        """Advance neurochemical state in response to an event (event impact)."""
        # Use a fixed event-processing timestep; background drift is handled in update()
        event_dt = 1.0
        if now is not None:
            self.last_update = now
        self.neurochemistry.update(
            dt=event_dt,
            appraisal_valence=appraisal.valence_delta,
            appraisal_arousal=appraisal.arousal_delta,
            appraisal_dominance=appraisal.dominance_delta,
            drive_error_temperature=self.drive.error_temperature,
            drive_rest_need=self.drive.rest_need,
            drive_task_load=max(self.drive.task_load, self.working_memory_load),
            drive_safety=self.drive.safety,
            surprise=abs(self.reward_prediction_error),
            circadian_hour=self.circadian_hour,
            metabolic_energy=self.metabolic_energy,
        )

    def _thalamic_gate(self, importance: float, appraisal: Appraisal) -> float:
        p = self.profile
        novelty = 1.0 - self.drive.safety
        # Acetylcholine attention mode:
        # Low ACh -> scanning/broad attention (higher novelty bias)
        # High ACh -> focused attention (higher importance gating, lower novelty)
        ach = self.neurochemistry.state.acetylcholine
        ach_focus = 0.5 + 0.5 * ach  # 0.5 at baseline, 1.0 at high ACh
        effective_novelty_bias = p.attention_novelty_bias * (1.0 - ach * 0.4)
        gate = importance * (1.0 + effective_novelty_bias * novelty) * (0.8 + 0.4 * ach_focus)
        if appraisal.threat_flag:
            gate *= 1.0 + (1.0 - p.attention_safety_bias)
        return min(1.0, gate)

    def _store_event(self, event: EpisodicEvent) -> None:
        self.episodic_buffer.append(event)
        # Hippocampal LTP: strengthen weight for this event kind
        if event.importance > 0.3:
            prev = self.hippocampal_weights.get(event.kind, 0.0)
            # BDNF gates learning rate
            learning_rate = 0.05 * (0.3 + 0.7 * self.neurochemistry.state.bdnf)
            self.hippocampal_weights[event.kind] = min(1.0, prev + learning_rate * event.importance)
        # Temporal contiguity: events within a short window strengthen each other
        recent = [t for t, _ in self.event_history if event.timestamp - t <= 2.0]
        for _, kind in self.event_history[-5:]:
            if event.timestamp - self.event_history[-1][0] <= 2.0 and kind != event.kind:
                prev = self.hippocampal_weights.get(kind, 0.0)
                self.hippocampal_weights[kind] = min(1.0, prev + 0.01 * event.importance)
        # Fear extinction: safe exposures reduce weight
        if not event.importance > 0.3 and event.kind in self.hippocampal_weights:
            self.hippocampal_weights[event.kind] = max(
                0.0, self.hippocampal_weights[event.kind] - 0.005
            )
        if len(self.episodic_buffer) > self.episodic_capacity:
            # Drop least important old event
            self.episodic_buffer.sort(key=lambda e: e.importance)
            self.episodic_buffer.pop(0)
            # Re-sort by time
            self.episodic_buffer.sort(key=lambda e: e.timestamp)

    def _compute_insula(self) -> Dict[str, float]:
        s = self.neurochemistry.state
        # Interoceptive prediction error: discordance between body state and arousal
        predicted_arousal = s.adrenaline * 0.4 + s.norepinephrine * 0.4 + (1 - s.heart_rate_variability) * 0.2
        body_prediction_error = abs(predicted_arousal - self.vad.arousal)
        # Discordance amplifies threat salience
        return {"body_prediction_error": self._round(body_prediction_error)}

    def _compute_acc(self) -> Dict[str, float]:
        v, a, d = self.vad.valence, self.vad.arousal, self.vad.dominance
        # Conflict: high arousal, low dominance, neutral valence
        conflict_signal = (1 - abs(v)) * a * (1 - d)
        if conflict_signal > 0.2:
            self.neurochemistry.state.norepinephrine = min(
                1.0, self.neurochemistry.state.norepinephrine + 0.03
            )
            self.drive.safety = max(0.0, self.drive.safety - 0.02)
        return {"conflict_signal": self._round(conflict_signal)}

    def _compute_lateral_habenula(self) -> Dict[str, float]:
        activation = 0.0
        if self.last_negative_event_kind:
            time_since = self.last_update - self.last_negative_event_time
            activation = max(0.0, 0.5 - time_since * 0.1)
        return {"activation": self._round(activation)}

    def _compute_septal(self) -> Dict[str, float]:
        s = self.neurochemistry.state
        social_approach = clamp01(
            s.oxytocin * 0.5 + (1 - s.cortisol) * 0.3 + self.drive.safety * 0.2
        )
        return {
            "social_approach": self._round(social_approach),
            "valence_buffer": self._round(s.oxytocin * 0.35 + s.opioid * 0.15),
        }

    def _compute_pag(self) -> Dict[str, str | float]:
        v, a, d = self.vad.valence, self.vad.arousal, self.vad.dominance
        threat = v < -0.3 or self.drive.safety < 0.4
        if a > 0.7 and d < 0.25 and threat:
            tier = "freeze"
        elif a > 0.6 and d < 0.5 and threat:
            tier = "flight"
        elif a > 0.5 and d > 0.6 and threat:
            tier = "fight"
        else:
            tier = "calm"
        return {"tier": tier, "threat_detected": threat}

    def _compute_polyvagal(self) -> Dict[str, Any]:
        state = self.neurochemistry.compute_polyvagal_state(self.drive.safety)
        return {
            "state": state,
            "social_engagement_possible": state == "ventral_vagal",
        }

    def _compute_affective_systems(self) -> Dict[str, float]:
        s = self.neurochemistry.state
        seeking = clamp01(s.dopamine_mesolimbic * 0.5 + s.orexin * 0.3 + (1 - s.cortisol) * 0.2)
        care = clamp01(s.oxytocin * 0.5 + s.prolactin * 0.3 + self.drive.safety * 0.2)
        fear = clamp01(s.adrenaline * 0.4 + s.cortisol * 0.3 + (1 - s.gaba) * 0.3)
        rage = clamp01(
            max(0, self.vad.arousal - 0.5) * 0.4
            + s.cortisol * 0.3
            + (1 - s.gaba) * 0.3
        ) if self.vad.dominance > 0.5 else 0.0
        panic_grief = clamp01(
            max(0, (0.9 - s.opioid)) * 0.6
            + max(0, (0.9 - s.oxytocin)) * 0.4
            + (1 - self.drive.safety) * 0.3
            + max(0, -self.vad.valence) * 0.2
        )
        # Boost from recent social-separation trigger
        if self.last_negative_event_kind == "social_separation":
            panic_grief = clamp01(panic_grief + 0.3)
        return {
            "seeking": self._round(seeking),
            "care": self._round(care),
            "fear": self._round(fear),
            "rage": self._round(rage),
            "panic_grief": self._round(panic_grief),
        }

    @staticmethod
    def _toward(current: float, target: float, amount: float) -> float:
        diff = target - current
        if abs(diff) <= amount:
            return target
        return current + math.copysign(amount, diff)

    @staticmethod
    def _round(x: float, places: int = 3) -> float:
        return round(x, places)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LimbicSystem":
        """Reconstruct a LimbicSystem from a serialized state snapshot."""
        limbic = cls(profile_name=data.get("profile", "default"))
        limbic.vad = VAD(**data.get("vad", {})).clamp()
        limbic.drive = DriveState(**data.get("drive", {})).clamp()
        limbic.expected_reward = data.get("expected_reward", 0.0)
        limbic.circadian_hour = data.get("circadian_hour", 12.0)
        cofactors = data.get("cofactors", {}).get("levels", {})
        if cofactors:
            limbic.cofactor_levels = {str(k): float(v) for k, v in cofactors.items()}
        uam = data.get("user_affect_mirror", {})
        limbic.user_affect_mirror = VAD(
            uam.get("valence", 0.0),
            uam.get("arousal", 0.2),
            uam.get("dominance", 0.5),
        ).clamp()
        hw = data.get("hippocampal_weights", {})
        limbic.hippocampal_weights = {str(k): float(v) for k, v in hw.items()}
        nc = data.get("neurochemistry", {})
        if nc:
            limbic.neurochemistry.state = NeurochemicalState.from_dict(nc).clamp()
        limbic.working_memory_load = data.get("working_memory_load", 0.0)
        limbic.metabolic_energy = data.get("metabolic_energy", 0.5)
        limbic.glucose = data.get("glucose", 0.5)
        limbic.last_negative_event_kind = data.get("last_negative_event_kind")
        limbic.last_negative_event_time = data.get("last_negative_event_time", 0.0)
        limbic.event_history = data.get("event_history", [])
        limbic.prefrontal_strength = data.get("prefrontal_strength", 0.5)
        limbic.last_update = data.get("timestamp", time.time())
        return limbic

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.get_state(),
            "working_memory_load": self.working_memory_load,
            "metabolic_energy": self.metabolic_energy,
            "glucose": self.glucose,
            "last_negative_event_kind": self.last_negative_event_kind,
            "last_negative_event_time": self.last_negative_event_time,
            "event_history": self.event_history,
            "prefrontal_strength": getattr(self, "prefrontal_strength", 0.5),
        }


# ---------------------------------------------------------------------------
# Hermes integration helpers
# ---------------------------------------------------------------------------

class LimbicSkillBridge:
    """
    Convenience wrapper for a Hermes skill that wants to own a LimbicSystem.

    Loads/saves state to disk so the affect persists across Hermes turns.
    """

    def __init__(self, state_path: str, profile_name: str = "default"):
        self.state_path = state_path
        try:
            self.limbic = self._load(state_path)
        except Exception:
            self.limbic = LimbicSystem(profile_name=profile_name)

    def observe(self, kind: str, **kwargs) -> None:
        self.limbic.observe_event(kind, **kwargs)
        self.save()

    def state(self) -> Dict:
        return self.limbic.get_state()

    def save(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump(self.limbic.to_dict(), f, indent=2)

    def apply_cofactors(self, cofactor_levels: Dict[str, float]) -> None:
        self.limbic.apply_cofactors(cofactor_levels)
        self.save()

    def set_user_affect(self, valence: float = 0.0, arousal: float = 0.2, dominance: float = 0.5) -> None:
        self.limbic.set_user_affect(valence, arousal, dominance)
        self.save()

    def set_circadian_hour(self, hour: float) -> None:
        self.limbic.set_circadian_hour(hour)
        self.save()

    def _load(self, path: str) -> LimbicSystem:
        with open(path, "r") as f:
            data = json.load(f)
        return LimbicSystem.from_dict(data)
