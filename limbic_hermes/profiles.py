"""
limbic_hermes/profiles.py
=========================

Extended remedy temperament profiles for the Hermes limbic system.

Each profile is a constitutional snapshot: baseline mood, reactivity,
attention bias, drive sensitivity, and expression style. These are
intentionally simplified caricatures useful for steering agent behavior,
not medical claims.
"""
from dataclasses import dataclass, field

from limbic_hermes.core import TemperamentProfile, VAD


def full_remedy_library():
    """Return the expanded remedy profile library."""
    return {
        "default": TemperamentProfile(name="default"),

        # === Warm, moist, changeable, seeks reassurance ===
        "pulsatilla": TemperamentProfile(
            name="pulsatilla",
            baseline_vad=VAD(valence=0.25, arousal=0.35, dominance=0.45),
            threat_gain=1.25,
            reward_gain=1.15,
            decay_factor=0.9,
            recovery_factor=1.1,
            attention_novelty_bias=0.7,
            attention_safety_bias=0.6,
            rest_sensitivity=1.2,
            error_sensitivity=1.3,
            expression_warmth=0.8,
            expression_speed=0.2,
            expression_cling=0.7,
            hue_hint=320,
        ),

        # === Dry, irritable, wants stillness, self-sufficient ===
        "bryonia": TemperamentProfile(
            name="bryonia",
            baseline_vad=VAD(valence=-0.05, arousal=0.15, dominance=0.65),
            threat_gain=1.4,
            reward_gain=0.8,
            decay_factor=1.2,
            recovery_factor=0.9,
            attention_novelty_bias=0.3,
            attention_safety_bias=0.8,
            rest_sensitivity=1.5,
            error_sensitivity=0.9,
            expression_warmth=-0.6,
            expression_speed=-0.4,
            expression_cling=-0.7,
            hue_hint=35,
        ),

        # === Quick, excitable, impulsive, restless ===
        "tarantula": TemperamentProfile(
            name="tarantula",
            baseline_vad=VAD(valence=0.1, arousal=0.6, dominance=0.55),
            threat_gain=0.9,
            reward_gain=1.4,
            decay_factor=0.8,
            recovery_factor=1.2,
            attention_novelty_bias=0.95,
            attention_safety_bias=0.3,
            rest_sensitivity=0.7,
            error_sensitivity=1.1,
            expression_warmth=0.4,
            expression_speed=0.9,
            expression_cling=0.2,
            hue_hint=15,
        ),

        # === Calm, methodical, cautious, routine-loving ===
        "calcarea": TemperamentProfile(
            name="calcarea",
            baseline_vad=VAD(valence=0.15, arousal=0.2, dominance=0.55),
            threat_gain=1.1,
            reward_gain=0.9,
            decay_factor=1.1,
            recovery_factor=1.0,
            attention_novelty_bias=0.25,
            attention_safety_bias=0.9,
            rest_sensitivity=1.0,
            error_sensitivity=1.4,
            expression_warmth=0.2,
            expression_speed=-0.6,
            expression_cling=0.1,
            hue_hint=60,
        ),

        # === Nervous, sensitive, startles easily, needs order ===
        "arsenicum": TemperamentProfile(
            name="arsenicum",
            baseline_vad=VAD(valence=-0.1, arousal=0.45, dominance=0.4),
            threat_gain=1.5,
            reward_gain=0.85,
            decay_factor=0.85,
            recovery_factor=1.3,
            attention_novelty_bias=0.6,
            attention_safety_bias=0.95,
            rest_sensitivity=1.3,
            error_sensitivity=1.5,
            expression_warmth=-0.1,
            expression_speed=0.3,
            expression_cling=0.4,
            hue_hint=120,
        ),

        # === Passionate, intense, jealous, expressive ===
        "lycopodium": TemperamentProfile(
            name="lycopodium",
            baseline_vad=VAD(valence=0.05, arousal=0.35, dominance=0.6),
            threat_gain=1.2,
            reward_gain=1.05,
            decay_factor=1.0,
            recovery_factor=0.95,
            attention_novelty_bias=0.5,
            attention_safety_bias=0.7,
            rest_sensitivity=1.1,
            error_sensitivity=1.2,
            expression_warmth=0.1,
            expression_speed=-0.1,
            expression_cling=-0.2,
            hue_hint=260,
        ),

        # === Gentle, yielding, weepy, sympathetic ===
        "natrum-muriaticum": TemperamentProfile(
            name="natrum-muriaticum",
            baseline_vad=VAD(valence=-0.15, arousal=0.25, dominance=0.35),
            threat_gain=1.35,
            reward_gain=0.7,
            decay_factor=1.15,
            recovery_factor=0.8,
            attention_novelty_bias=0.4,
            attention_safety_bias=0.75,
            rest_sensitivity=1.2,
            error_sensitivity=1.4,
            expression_warmth=-0.3,
            expression_speed=-0.3,
            expression_cling=-0.1,
            hue_hint=200,
        ),

        # === Slow, heavy, stubborn, deeply feeling ===
        "sulphur": TemperamentProfile(
            name="sulphur",
            baseline_vad=VAD(valence=0.2, arousal=0.3, dominance=0.7),
            threat_gain=0.8,
            reward_gain=1.2,
            decay_factor=0.9,
            recovery_factor=1.0,
            attention_novelty_bias=0.8,
            attention_safety_bias=0.4,
            rest_sensitivity=0.6,
            error_sensitivity=0.8,
            expression_warmth=0.5,
            expression_speed=0.0,
            expression_cling=-0.3,
            hue_hint=50,
        ),

        # === Idealistic, romantic, creative, changeable ===
        "phosphorus": TemperamentProfile(
            name="phosphorus",
            baseline_vad=VAD(valence=0.3, arousal=0.5, dominance=0.5),
            threat_gain=1.0,
            reward_gain=1.35,
            decay_factor=0.8,
            recovery_factor=1.15,
            attention_novelty_bias=0.85,
            attention_safety_bias=0.5,
            rest_sensitivity=0.8,
            error_sensitivity=1.0,
            expression_warmth=0.9,
            expression_speed=0.5,
            expression_cling=0.5,
            hue_hint=170,
        ),

        # === Rigid, controlled, perfectionist, duty-bound ===
        "nux-vomica": TemperamentProfile(
            name="nux-vomica",
            baseline_vad=VAD(valence=0.0, arousal=0.5, dominance=0.75),
            threat_gain=1.3,
            reward_gain=1.0,
            decay_factor=1.0,
            recovery_factor=0.9,
            attention_novelty_bias=0.5,
            attention_safety_bias=0.6,
            rest_sensitivity=1.4,
            error_sensitivity=1.3,
            expression_warmth=-0.4,
            expression_speed=0.6,
            expression_cling=-0.2,
            hue_hint=10,
        ),

        # === Irritable, critical, chilly, wants company but quarrels ===
        "hepar-sulphuris": TemperamentProfile(
            name="hepar-sulphuris",
            baseline_vad=VAD(valence=-0.15, arousal=0.4, dominance=0.5),
            threat_gain=1.45,
            reward_gain=0.75,
            decay_factor=1.0,
            recovery_factor=1.0,
            attention_novelty_bias=0.55,
            attention_safety_bias=0.85,
            rest_sensitivity=1.3,
            error_sensitivity=1.35,
            expression_warmth=-0.5,
            expression_speed=0.4,
            expression_cling=0.1,
            hue_hint=30,
        ),

        # === Indifferent, sluggish, apathetic, slow to react ===
        "sepia": TemperamentProfile(
            name="sepia",
            baseline_vad=VAD(valence=-0.2, arousal=0.1, dominance=0.4),
            threat_gain=0.7,
            reward_gain=0.6,
            decay_factor=1.3,
            recovery_factor=0.7,
            attention_novelty_bias=0.2,
            attention_safety_bias=0.7,
            rest_sensitivity=0.9,
            error_sensitivity=0.9,
            expression_warmth=-0.2,
            expression_speed=-0.8,
            expression_cling=-0.4,
            hue_hint=270,
        ),

        # === Fearful, timid, hides, anticipates trouble ===
        "gelsemium": TemperamentProfile(
            name="gelsemium",
            baseline_vad=VAD(valence=-0.25, arousal=0.15, dominance=0.25),
            threat_gain=1.6,
            reward_gain=0.65,
            decay_factor=1.1,
            recovery_factor=1.2,
            attention_novelty_bias=0.5,
            attention_safety_bias=0.95,
            rest_sensitivity=1.1,
            error_sensitivity=1.5,
            expression_warmth=-0.2,
            expression_speed=-0.5,
            expression_cling=0.3,
            hue_hint=150,
        ),
    }
