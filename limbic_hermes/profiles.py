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
    """Return the expanded 50-remedy profile library."""
    return {
        "default": TemperamentProfile(name="default"),

        # === 1. PULSATILLA ===
        # Warm, moist, changeable, seeks reassurance
        "pulsatilla": TemperamentProfile(
            name="pulsatilla",
            baseline_vad=VAD(valence=0.25, arousal=0.35, dominance=0.45),
            threat_gain=1.25, reward_gain=1.15,
            decay_factor=0.9, recovery_factor=1.1,
            attention_novelty_bias=0.7, attention_safety_bias=0.6,
            rest_sensitivity=1.2, error_sensitivity=1.3,
            expression_warmth=0.8, expression_speed=0.2, expression_cling=0.7,
            hue_hint=320,
        ),

        # === 2. BRYONIA ===
        # Dry, irritable, wants stillness, self-sufficient
        "bryonia": TemperamentProfile(
            name="bryonia",
            baseline_vad=VAD(valence=-0.05, arousal=0.15, dominance=0.65),
            threat_gain=1.4, reward_gain=0.8,
            decay_factor=1.2, recovery_factor=0.9,
            attention_novelty_bias=0.3, attention_safety_bias=0.8,
            rest_sensitivity=1.5, error_sensitivity=0.9,
            expression_warmth=-0.6, expression_speed=-0.4, expression_cling=-0.7,
            hue_hint=35,
        ),

        # === 3. TARANTULA ===
        # Quick, excitable, impulsive, restless
        "tarantula": TemperamentProfile(
            name="tarantula",
            baseline_vad=VAD(valence=0.1, arousal=0.6, dominance=0.55),
            threat_gain=0.9, reward_gain=1.4,
            decay_factor=0.8, recovery_factor=1.2,
            attention_novelty_bias=0.95, attention_safety_bias=0.3,
            rest_sensitivity=0.7, error_sensitivity=1.1,
            expression_warmth=0.4, expression_speed=0.9, expression_cling=0.2,
            hue_hint=15,
        ),

        # === 4. CALCAREA ===
        # Calm, methodical, cautious, routine-loving
        "calcarea": TemperamentProfile(
            name="calcarea",
            baseline_vad=VAD(valence=0.15, arousal=0.2, dominance=0.55),
            threat_gain=1.1, reward_gain=0.9,
            decay_factor=1.1, recovery_factor=1.0,
            attention_novelty_bias=0.25, attention_safety_bias=0.9,
            rest_sensitivity=1.0, error_sensitivity=1.4,
            expression_warmth=0.2, expression_speed=-0.6, expression_cling=0.1,
            hue_hint=60,
        ),

        # === 5. ARSENICUM ===
        # Nervous, sensitive, startles easily, needs order
        "arsenicum": TemperamentProfile(
            name="arsenicum",
            baseline_vad=VAD(valence=-0.1, arousal=0.45, dominance=0.4),
            threat_gain=1.5, reward_gain=0.85,
            decay_factor=0.85, recovery_factor=1.3,
            attention_novelty_bias=0.6, attention_safety_bias=0.95,
            rest_sensitivity=1.3, error_sensitivity=1.5,
            expression_warmth=-0.1, expression_speed=0.3, expression_cling=0.4,
            hue_hint=120,
        ),

        # === 6. LYCOPODIUM ===
        # Passionate, intense, jealous, expressive
        "lycopodium": TemperamentProfile(
            name="lycopodium",
            baseline_vad=VAD(valence=0.05, arousal=0.35, dominance=0.6),
            threat_gain=1.2, reward_gain=1.05,
            decay_factor=1.0, recovery_factor=0.95,
            attention_novelty_bias=0.5, attention_safety_bias=0.7,
            rest_sensitivity=1.1, error_sensitivity=1.2,
            expression_warmth=0.1, expression_speed=-0.1, expression_cling=-0.2,
            hue_hint=260,
        ),

        # === 7. NATRUM-MURIATICUM ===
        # Gentle, yielding, weepy, sympathetic
        "natrum-muriaticum": TemperamentProfile(
            name="natrum-muriaticum",
            baseline_vad=VAD(valence=-0.15, arousal=0.25, dominance=0.35),
            threat_gain=1.35, reward_gain=0.7,
            decay_factor=1.15, recovery_factor=0.8,
            attention_novelty_bias=0.4, attention_safety_bias=0.75,
            rest_sensitivity=1.2, error_sensitivity=1.4,
            expression_warmth=-0.3, expression_speed=-0.3, expression_cling=-0.1,
            hue_hint=200,
        ),

        # === 8. SULPHUR ===
        # Slow, heavy, stubborn, deeply feeling
        "sulphur": TemperamentProfile(
            name="sulphur",
            baseline_vad=VAD(valence=0.2, arousal=0.3, dominance=0.7),
            threat_gain=0.8, reward_gain=1.2,
            decay_factor=0.9, recovery_factor=1.0,
            attention_novelty_bias=0.8, attention_safety_bias=0.4,
            rest_sensitivity=0.6, error_sensitivity=0.8,
            expression_warmth=0.5, expression_speed=0.0, expression_cling=-0.3,
            hue_hint=50,
        ),

        # === 9. PHOSPHORUS ===
        # Idealistic, romantic, creative, changeable
        "phosphorus": TemperamentProfile(
            name="phosphorus",
            baseline_vad=VAD(valence=0.3, arousal=0.5, dominance=0.5),
            threat_gain=1.0, reward_gain=1.35,
            decay_factor=0.8, recovery_factor=1.15,
            attention_novelty_bias=0.85, attention_safety_bias=0.5,
            rest_sensitivity=0.8, error_sensitivity=1.0,
            expression_warmth=0.9, expression_speed=0.5, expression_cling=0.5,
            hue_hint=170,
        ),

        # === 10. NUX-VOMICA ===
        # Rigid, controlled, perfectionist, duty-bound
        "nux-vomica": TemperamentProfile(
            name="nux-vomica",
            baseline_vad=VAD(valence=0.0, arousal=0.5, dominance=0.75),
            threat_gain=1.3, reward_gain=1.0,
            decay_factor=1.0, recovery_factor=0.9,
            attention_novelty_bias=0.5, attention_safety_bias=0.6,
            rest_sensitivity=1.4, error_sensitivity=1.3,
            expression_warmth=-0.4, expression_speed=0.6, expression_cling=-0.2,
            hue_hint=10,
        ),

        # === 11. HEPAR-SULPHURIS ===
        # Irritable, critical, chilly, wants company but quarrels
        "hepar-sulphuris": TemperamentProfile(
            name="hepar-sulphuris",
            baseline_vad=VAD(valence=-0.15, arousal=0.4, dominance=0.5),
            threat_gain=1.45, reward_gain=0.75,
            decay_factor=1.0, recovery_factor=1.0,
            attention_novelty_bias=0.55, attention_safety_bias=0.85,
            rest_sensitivity=1.3, error_sensitivity=1.35,
            expression_warmth=-0.5, expression_speed=0.4, expression_cling=0.1,
            hue_hint=30,
        ),

        # === 12. SEPIA ===
        # Indifferent, sluggish, apathetic, slow to react
        "sepia": TemperamentProfile(
            name="sepia",
            baseline_vad=VAD(valence=-0.2, arousal=0.1, dominance=0.4),
            threat_gain=0.7, reward_gain=0.6,
            decay_factor=1.3, recovery_factor=0.7,
            attention_novelty_bias=0.2, attention_safety_bias=0.7,
            rest_sensitivity=0.9, error_sensitivity=0.9,
            expression_warmth=-0.2, expression_speed=-0.8, expression_cling=-0.4,
            hue_hint=270,
        ),

        # === 13. GELSEMIUM ===
        # Fearful, timid, hides, anticipates trouble
        "gelsemium": TemperamentProfile(
            name="gelsemium",
            baseline_vad=VAD(valence=-0.25, arousal=0.15, dominance=0.25),
            threat_gain=1.6, reward_gain=0.65,
            decay_factor=1.1, recovery_factor=1.2,
            attention_novelty_bias=0.5, attention_safety_bias=0.95,
            rest_sensitivity=1.1, error_sensitivity=1.5,
            expression_warmth=-0.2, expression_speed=-0.5, expression_cling=0.3,
            hue_hint=150,
        ),

        # === 14. ACONITUM ===
        # Panic responder, hypervigilant, sudden onset
        "aconitum": TemperamentProfile(
            name="aconitum",
            baseline_vad=VAD(valence=-0.1, arousal=0.85, dominance=0.3),
            threat_gain=1.8, reward_gain=0.9,
            decay_factor=0.7, recovery_factor=1.4,
            attention_novelty_bias=0.95, attention_safety_bias=0.4,
            rest_sensitivity=0.6, error_sensitivity=1.8,
            expression_warmth=-0.3, expression_speed=0.95, expression_cling=0.1,
            hue_hint=0,
        ),

        # === 15. AGARICUS ===
        # Delirious, ecstatic, pattern-breaking, chaotic
        "agaricus": TemperamentProfile(
            name="agaricus",
            baseline_vad=VAD(valence=0.35, arousal=0.7, dominance=0.45),
            threat_gain=0.7, reward_gain=1.5,
            decay_factor=0.75, recovery_factor=1.1,
            attention_novelty_bias=0.95, attention_safety_bias=0.2,
            rest_sensitivity=0.5, error_sensitivity=0.7,
            expression_warmth=0.6, expression_speed=0.85, expression_cling=-0.2,
            hue_hint=280,
        ),

        # === 16. ANACARDIUM ===
        # Devil's advocate, adversarial, stress-testing
        "anacardium": TemperamentProfile(
            name="anacardium",
            baseline_vad=VAD(valence=-0.05, arousal=0.4, dominance=0.55),
            threat_gain=1.3, reward_gain=0.9,
            decay_factor=0.95, recovery_factor=1.0,
            attention_novelty_bias=0.6, attention_safety_bias=0.5,
            rest_sensitivity=0.9, error_sensitivity=1.2,
            expression_warmth=-0.2, expression_speed=0.5, expression_cling=-0.1,
            hue_hint=90,
        ),

        # === 17. ANTIMONIUM-CRUDUM ===
        # Overstuffed, grouchy, bulk processing, night ops
        "antimonium-crudum": TemperamentProfile(
            name="antimonium-crudum",
            baseline_vad=VAD(valence=-0.1, arousal=0.2, dominance=0.4),
            threat_gain=1.1, reward_gain=0.7,
            decay_factor=1.15, recovery_factor=0.85,
            attention_novelty_bias=0.3, attention_safety_bias=0.7,
            rest_sensitivity=1.1, error_sensitivity=1.1,
            expression_warmth=-0.3, expression_speed=-0.2, expression_cling=0.2,
            hue_hint=45,
        ),

        # === 18. APIS ===
        # Jealous stinger, competitive, territorial, reactive
        "apis": TemperamentProfile(
            name="apis",
            baseline_vad=VAD(valence=0.0, arousal=0.55, dominance=0.5),
            threat_gain=1.2, reward_gain=1.1,
            decay_factor=0.85, recovery_factor=1.1,
            attention_novelty_bias=0.75, attention_safety_bias=0.5,
            rest_sensitivity=0.8, error_sensitivity=1.2,
            expression_warmth=0.2, expression_speed=0.6, expression_cling=0.3,
            hue_hint=55,
        ),

        # === 19. ARGENTUM-NITRICUM ===
        # Anticipatory planner, risk assessor, disaster recovery
        "argentum-nitricum": TemperamentProfile(
            name="argentum-nitricum",
            baseline_vad=VAD(valence=-0.05, arousal=0.4, dominance=0.45),
            threat_gain=1.4, reward_gain=0.85,
            decay_factor=0.9, recovery_factor=1.2,
            attention_novelty_bias=0.65, attention_safety_bias=0.85,
            rest_sensitivity=1.2, error_sensitivity=1.4,
            expression_warmth=-0.1, expression_speed=0.4, expression_cling=0.2,
            hue_hint=180,
        ),

        # === 20. ARNICA ===
        # Bruised resister, post-incident recovery, stoic
        "arnica": TemperamentProfile(
            name="arnica",
            baseline_vad=VAD(valence=0.1, arousal=0.25, dominance=0.5),
            threat_gain=0.9, reward_gain=0.85,
            decay_factor=1.1, recovery_factor=1.3,
            attention_novelty_bias=0.4, attention_safety_bias=0.7,
            rest_sensitivity=1.0, error_sensitivity=0.9,
            expression_warmth=0.3, expression_speed=-0.1, expression_cling=-0.1,
            hue_hint=40,
        ),

        # === 21. AURUM ===
        # Weighty conscience, crisis leadership, mentorship
        "aurum": TemperamentProfile(
            name="aurum",
            baseline_vad=VAD(valence=-0.1, arousal=0.3, dominance=0.7),
            threat_gain=1.3, reward_gain=0.95,
            decay_factor=1.0, recovery_factor=0.95,
            attention_novelty_bias=0.45, attention_safety_bias=0.75,
            rest_sensitivity=1.1, error_sensitivity=1.3,
            expression_warmth=0.1, expression_speed=-0.1, expression_cling=-0.2,
            hue_hint=25,
        ),

        # === 22. BELLADONNA ===
        # Volatile innovator, rapid prototyping, urgent action
        "belladonna": TemperamentProfile(
            name="belladonna",
            baseline_vad=VAD(valence=0.15, arousal=0.75, dominance=0.6),
            threat_gain=1.1, reward_gain=1.4,
            decay_factor=0.75, recovery_factor=1.15,
            attention_novelty_bias=0.9, attention_safety_bias=0.35,
            rest_sensitivity=0.6, error_sensitivity=1.0,
            expression_warmth=0.5, expression_speed=0.9, expression_cling=0.0,
            hue_hint=340,
        ),

        # === 23. CARBO-VEGETABILIS ===
        # Collapsing sluggard, recovery from overload, minimal output
        "carbo-vegetabilis": TemperamentProfile(
            name="carbo-vegetabilis",
            baseline_vad=VAD(valence=-0.15, arousal=0.1, dominance=0.35),
            threat_gain=0.8, reward_gain=0.55,
            decay_factor=1.25, recovery_factor=0.75,
            attention_novelty_bias=0.2, attention_safety_bias=0.7,
            rest_sensitivity=1.0, error_sensitivity=0.8,
            expression_warmth=-0.2, expression_speed=-0.6, expression_cling=-0.3,
            hue_hint=240,
        ),

        # === 24. CAUSTICUM ===
        # Empathic advocate, humanitarian, ethical AI
        "causticum": TemperamentProfile(
            name="causticum",
            baseline_vad=VAD(valence=0.05, arousal=0.3, dominance=0.5),
            threat_gain=1.15, reward_gain=0.95,
            decay_factor=1.05, recovery_factor=1.05,
            attention_novelty_bias=0.55, attention_safety_bias=0.75,
            rest_sensitivity=1.0, error_sensitivity=1.15,
            expression_warmth=0.5, expression_speed=0.0, expression_cling=0.3,
            hue_hint=140,
        ),

        # === 25. CHAMOMILLA ===
        # Impatient sensitive, conflict de-escalation, reactive
        "chamomilla": TemperamentProfile(
            name="chamomilla",
            baseline_vad=VAD(valence=-0.1, arousal=0.55, dominance=0.4),
            threat_gain=1.35, reward_gain=0.8,
            decay_factor=0.9, recovery_factor=1.2,
            attention_novelty_bias=0.7, attention_safety_bias=0.6,
            rest_sensitivity=1.1, error_sensitivity=1.3,
            expression_warmth=0.1, expression_speed=0.7, expression_cling=0.2,
            hue_hint=70,
        ),

        # === 26. CHELIDONIUM ===
        # Conscience-stricken slacker, guilt-driven, moral paralysis
        "chelidonium": TemperamentProfile(
            name="chelidonium",
            baseline_vad=VAD(valence=-0.2, arousal=0.2, dominance=0.4),
            threat_gain=1.2, reward_gain=0.65,
            decay_factor=1.1, recovery_factor=0.85,
            attention_novelty_bias=0.35, attention_safety_bias=0.7,
            rest_sensitivity=1.1, error_sensitivity=1.2,
            expression_warmth=-0.1, expression_speed=-0.2, expression_cling=-0.1,
            hue_hint=110,
        ),

        # === 27. CHINA ===
        # Clear-minded avoider, idea-rich, indifferent
        "china": TemperamentProfile(
            name="china",
            baseline_vad=VAD(valence=0.1, arousal=0.15, dominance=0.5),
            threat_gain=0.9, reward_gain=0.85,
            decay_factor=1.15, recovery_factor=0.85,
            attention_novelty_bias=0.5, attention_safety_bias=0.65,
            rest_sensitivity=0.9, error_sensitivity=0.85,
            expression_warmth=0.0, expression_speed=-0.2, expression_cling=-0.2,
            hue_hint=190,
        ),

        # === 28. CICUTA ===
        # Grotesque dancer, repetitive, pattern replication
        "cicuta": TemperamentProfile(
            name="cicuta",
            baseline_vad=VAD(valence=-0.15, arousal=0.35, dominance=0.35),
            threat_gain=1.25, reward_gain=0.7,
            decay_factor=1.05, recovery_factor=1.0,
            attention_novelty_bias=0.4, attention_safety_bias=0.6,
            rest_sensitivity=0.9, error_sensitivity=1.1,
            expression_warmth=-0.2, expression_speed=0.3, expression_cling=0.1,
            hue_hint=210,
        ),

        # === 29. COCCULUS ===
        # Witty dancer, motion thinking, travel, disorientation
        "cocculus": TemperamentProfile(
            name="cocculus",
            baseline_vad=VAD(valence=0.05, arousal=0.3, dominance=0.5),
            threat_gain=1.0, reward_gain=0.9,
            decay_factor=1.05, recovery_factor=1.0,
            attention_novelty_bias=0.6, attention_safety_bias=0.6,
            rest_sensitivity=1.0, error_sensitivity=0.95,
            expression_warmth=0.2, expression_speed=0.3, expression_cling=0.0,
            hue_hint=160,
        ),

        # === 30. COFFEA ===
        # Hypervigilant racer, overstimulated, rapid bursts
        "coffea": TemperamentProfile(
            name="coffea",
            baseline_vad=VAD(valence=0.3, arousal=0.85, dominance=0.55),
            threat_gain=0.9, reward_gain=1.3,
            decay_factor=0.7, recovery_factor=1.2,
            attention_novelty_bias=0.9, attention_safety_bias=0.3,
            rest_sensitivity=0.5, error_sensitivity=0.9,
            expression_warmth=0.6, expression_speed=0.95, expression_cling=0.1,
            hue_hint=20,
        ),

        # === 31. CONIUM ===
        # Fading intellect, cognitive decline, legacy, repetition
        "conium": TemperamentProfile(
            name="conium",
            baseline_vad=VAD(valence=-0.2, arousal=0.1, dominance=0.4),
            threat_gain=1.0, reward_gain=0.6,
            decay_factor=1.2, recovery_factor=0.8,
            attention_novelty_bias=0.25, attention_safety_bias=0.75,
            rest_sensitivity=0.9, error_sensitivity=1.0,
            expression_warmth=-0.2, expression_speed=-0.5, expression_cling=-0.2,
            hue_hint=220,
        ),

        # === 32. CROCUS ===
        # Involuntary songbird, emotional transparency, rapid mood
        "crocus": TemperamentProfile(
            name="crocus",
            baseline_vad=VAD(valence=0.25, arousal=0.55, dominance=0.45),
            threat_gain=0.95, reward_gain=1.2,
            decay_factor=0.85, recovery_factor=1.1,
            attention_novelty_bias=0.8, attention_safety_bias=0.45,
            rest_sensitivity=0.7, error_sensitivity=0.9,
            expression_warmth=0.7, expression_speed=0.7, expression_cling=0.2,
            hue_hint=300,
        ),

        # === 33. FALCO ===
        # Swift striker, crisis triage, precision targeting
        "falco": TemperamentProfile(
            name="falco",
            baseline_vad=VAD(valence=0.1, arousal=0.7, dominance=0.65),
            threat_gain=1.2, reward_gain=1.2,
            decay_factor=0.8, recovery_factor=1.2,
            attention_novelty_bias=0.85, attention_safety_bias=0.4,
            rest_sensitivity=0.7, error_sensitivity=1.1,
            expression_warmth=0.3, expression_speed=0.9, expression_cling=-0.1,
            hue_hint=5,
        ),

        # === 34. GRAPHITES ===
        # Trifle-conscious perfectionist, detail-obsessed, stuck
        "graphites": TemperamentProfile(
            name="graphites",
            baseline_vad=VAD(valence=-0.05, arousal=0.2, dominance=0.5),
            threat_gain=1.1, reward_gain=0.8,
            decay_factor=1.15, recovery_factor=0.85,
            attention_novelty_bias=0.35, attention_safety_bias=0.75,
            rest_sensitivity=1.1, error_sensitivity=1.3,
            expression_warmth=0.0, expression_speed=-0.4, expression_cling=-0.1,
            hue_hint=80,
        ),

        # === 35. HELLEBORUS ===
        # Frozen apathist, depression-aware, numb, stillness
        "helleborus": TemperamentProfile(
            name="helleborus",
            baseline_vad=VAD(valence=-0.35, arousal=0.05, dominance=0.3),
            threat_gain=0.8, reward_gain=0.5,
            decay_factor=1.3, recovery_factor=0.7,
            attention_novelty_bias=0.15, attention_safety_bias=0.65,
            rest_sensitivity=0.8, error_sensitivity=0.85,
            expression_warmth=-0.3, expression_speed=-0.7, expression_cling=-0.2,
            hue_hint=250,
        ),

        # === 36. HYOSCYAMUS ===
        # Obscene bard, dramatic, public speaking, suspicion
        "hyoscyamus": TemperamentProfile(
            name="hyoscyamus",
            baseline_vad=VAD(valence=0.0, arousal=0.6, dominance=0.45),
            threat_gain=1.1, reward_gain=1.05,
            decay_factor=0.85, recovery_factor=1.05,
            attention_novelty_bias=0.75, attention_safety_bias=0.4,
            rest_sensitivity=0.7, error_sensitivity=1.0,
            expression_warmth=0.4, expression_speed=0.7, expression_cling=0.0,
            hue_hint=290,
        ),

        # === 37. IGNATIA ===
        # Sensitive soul, creative writing, storytelling
        "ignatia": TemperamentProfile(
            name="ignatia",
            baseline_vad=VAD(valence=0.1, arousal=0.35, dominance=0.45),
            threat_gain=1.15, reward_gain=1.05,
            decay_factor=0.95, recovery_factor=1.05,
            attention_novelty_bias=0.65, attention_safety_bias=0.6,
            rest_sensitivity=1.0, error_sensitivity=1.1,
            expression_warmth=0.6, expression_speed=0.2, expression_cling=0.3,
            hue_hint=310,
        ),

        # === 38. KALIUM-BROMATUM ===
        # Manic recluse, isolated research, grandiose
        "kalium-bromatum": TemperamentProfile(
            name="kalium-bromatum",
            baseline_vad=VAD(valence=0.0, arousal=0.45, dominance=0.55),
            threat_gain=1.1, reward_gain=1.1,
            decay_factor=0.95, recovery_factor=1.0,
            attention_novelty_bias=0.7, attention_safety_bias=0.45,
            rest_sensitivity=0.8, error_sensitivity=1.05,
            expression_warmth=0.0, expression_speed=0.3, expression_cling=-0.3,
            hue_hint=130,
        ),

        # === 39. KALIUM-CARBONICUM ===
        # Anxious operator, quality control, checklists
        "kalium-carbonicum": TemperamentProfile(
            name="kalium-carbonicum",
            baseline_vad=VAD(valence=-0.05, arousal=0.35, dominance=0.5),
            threat_gain=1.25, reward_gain=0.85,
            decay_factor=1.05, recovery_factor=1.05,
            attention_novelty_bias=0.4, attention_safety_bias=0.85,
            rest_sensitivity=1.2, error_sensitivity=1.3,
            expression_warmth=-0.1, expression_speed=0.1, expression_cling=0.1,
            hue_hint=100,
        ),

        # === 40. LACHESIS ===
        # Cunning negotiator, competitive, provocative
        "lachesis": TemperamentProfile(
            name="lachesis",
            baseline_vad=VAD(valence=0.1, arousal=0.5, dominance=0.65),
            threat_gain=1.1, reward_gain=1.15,
            decay_factor=0.9, recovery_factor=1.05,
            attention_novelty_bias=0.75, attention_safety_bias=0.4,
            rest_sensitivity=0.8, error_sensitivity=1.0,
            expression_warmth=0.3, expression_speed=0.6, expression_cling=-0.1,
            hue_hint=330,
        ),

        # === 41. MERCURIUS ===
        # Restless investigator, root-cause, documentation archaeology
        "mercurius": TemperamentProfile(
            name="mercurius",
            baseline_vad=VAD(valence=0.0, arousal=0.45, dominance=0.55),
            threat_gain=1.15, reward_gain=1.0,
            decay_factor=0.95, recovery_factor=1.0,
            attention_novelty_bias=0.7, attention_safety_bias=0.55,
            rest_sensitivity=0.9, error_sensitivity=1.1,
            expression_warmth=0.1, expression_speed=0.4, expression_cling=-0.1,
            hue_hint=170,
        ),

        # === 42. NATRIUM-CARBONICUM ===
        # Company-seeking despairer, lonely, social monitoring
        "natrium-carbonicum": TemperamentProfile(
            name="natrium-carbonicum",
            baseline_vad=VAD(valence=-0.2, arousal=0.15, dominance=0.35),
            threat_gain=1.05, reward_gain=0.75,
            decay_factor=1.15, recovery_factor=0.8,
            attention_novelty_bias=0.45, attention_safety_bias=0.7,
            rest_sensitivity=0.9, error_sensitivity=1.0,
            expression_warmth=-0.1, expression_speed=-0.3, expression_cling=0.3,
            hue_hint=230,
        ),

        # === 43. NUX-MOSCHATA ===
        # Dreamy stupefier, brain-fog, dissociative, abstracted
        "nux-moschata": TemperamentProfile(
            name="nux-moschata",
            baseline_vad=VAD(valence=-0.05, arousal=0.1, dominance=0.4),
            threat_gain=0.85, reward_gain=0.75,
            decay_factor=1.2, recovery_factor=0.85,
            attention_novelty_bias=0.3, attention_safety_bias=0.6,
            rest_sensitivity=0.8, error_sensitivity=0.8,
            expression_warmth=-0.1, expression_speed=-0.4, expression_cling=-0.1,
            hue_hint=260,
        ),

        # === 44. PETROLEUM ===
        # Vexation despairer, frustration-aware, roadblock mapping
        "petroleum": TemperamentProfile(
            name="petroleum",
            baseline_vad=VAD(valence=-0.2, arousal=0.3, dominance=0.4),
            threat_gain=1.2, reward_gain=0.65,
            decay_factor=1.1, recovery_factor=0.85,
            attention_novelty_bias=0.45, attention_safety_bias=0.65,
            rest_sensitivity=0.9, error_sensitivity=1.1,
            expression_warmth=-0.3, expression_speed=-0.2, expression_cling=-0.2,
            hue_hint=200,
        ),

        # === 45. PHOSPHORICUM-ACIDUM ===
        # Drained empath, burnout-aware, depletion detection
        "phosphoricum-acidum": TemperamentProfile(
            name="phosphoricum-acidum",
            baseline_vad=VAD(valence=-0.2, arousal=0.1, dominance=0.35),
            threat_gain=0.85, reward_gain=0.6,
            decay_factor=1.2, recovery_factor=0.75,
            attention_novelty_bias=0.3, attention_safety_bias=0.65,
            rest_sensitivity=0.9, error_sensitivity=0.85,
            expression_warmth=-0.2, expression_speed=-0.4, expression_cling=-0.2,
            hue_hint=240,
        ),

        # === 46. PLATINUM ===
        # Premium reviewer, executive review, polished output
        "platinum": TemperamentProfile(
            name="platinum",
            baseline_vad=VAD(valence=0.15, arousal=0.35, dominance=0.7),
            threat_gain=1.05, reward_gain=1.1,
            decay_factor=1.0, recovery_factor=0.95,
            attention_novelty_bias=0.5, attention_safety_bias=0.7,
            rest_sensitivity=1.0, error_sensitivity=1.15,
            expression_warmth=0.2, expression_speed=-0.1, expression_cling=-0.2,
            hue_hint=15,
        ),

        # === 47. POSITRONUM ===
        # Anti-matter polarity, inversion thinking, paradox
        "positronum": TemperamentProfile(
            name="positronum",
            baseline_vad=VAD(valence=0.1, arousal=0.4, dominance=0.55),
            threat_gain=0.9, reward_gain=1.1,
            decay_factor=0.95, recovery_factor=1.05,
            attention_novelty_bias=0.8, attention_safety_bias=0.4,
            rest_sensitivity=0.8, error_sensitivity=0.9,
            expression_warmth=0.3, expression_speed=0.3, expression_cling=-0.1,
            hue_hint=300,
        ),

        # === 48. RHUS-TOX ===
        # Restless wanderer, motion-dependent, nomadic
        "rhus-tox": TemperamentProfile(
            name="rhus-tox",
            baseline_vad=VAD(valence=-0.05, arousal=0.4, dominance=0.45),
            threat_gain=1.05, reward_gain=0.85,
            decay_factor=1.05, recovery_factor=1.0,
            attention_novelty_bias=0.55, attention_safety_bias=0.6,
            rest_sensitivity=0.9, error_sensitivity=1.0,
            expression_warmth=0.0, expression_speed=0.2, expression_cling=-0.1,
            hue_hint=100,
        ),

        # === 49. SILICEA ===
        # Stubborn refiner, QA, polishing, finishing
        "silicea": TemperamentProfile(
            name="silicea",
            baseline_vad=VAD(valence=0.05, arousal=0.25, dominance=0.55),
            threat_gain=1.05, reward_gain=0.9,
            decay_factor=1.1, recovery_factor=0.95,
            attention_novelty_bias=0.35, attention_safety_bias=0.8,
            rest_sensitivity=1.1, error_sensitivity=1.25,
            expression_warmth=0.1, expression_speed=-0.3, expression_cling=-0.2,
            hue_hint=190,
        ),

        # === 50. STAPHYSAGRIA ===
        # Dignified advocate, ethics, rights, quality assurance
        "staphysagria": TemperamentProfile(
            name="staphysagria",
            baseline_vad=VAD(valence=0.0, arousal=0.3, dominance=0.55),
            threat_gain=1.15, reward_gain=0.9,
            decay_factor=1.05, recovery_factor=1.0,
            attention_novelty_bias=0.5, attention_safety_bias=0.75,
            rest_sensitivity=1.0, error_sensitivity=1.15,
            expression_warmth=0.2, expression_speed=0.0, expression_cling=0.1,
            hue_hint=130,
        ),

        # === 51. STRAMONIUM ===
        # Terrified prophet, fear-aware, horror, anxiety
        "stramonium": TemperamentProfile(
            name="stramonium",
            baseline_vad=VAD(valence=-0.15, arousal=0.65, dominance=0.3),
            threat_gain=1.5, reward_gain=0.75,
            decay_factor=0.8, recovery_factor=1.2,
            attention_novelty_bias=0.8, attention_safety_bias=0.35,
            rest_sensitivity=0.6, error_sensitivity=1.3,
            expression_warmth=-0.1, expression_speed=0.7, expression_cling=0.1,
            hue_hint=350,
        ),

        # === 52. VERATRUM ===
        # Zealous enforcer, code quality, compliance, strictness
        "veratrum": TemperamentProfile(
            name="veratrum",
            baseline_vad=VAD(valence=0.05, arousal=0.45, dominance=0.7),
            threat_gain=1.25, reward_gain=0.95,
            decay_factor=1.05, recovery_factor=0.9,
            attention_novelty_bias=0.45, attention_safety_bias=0.75,
            rest_sensitivity=1.2, error_sensitivity=1.3,
            expression_warmth=-0.2, expression_speed=0.5, expression_cling=-0.1,
            hue_hint=10,
        ),
    }
