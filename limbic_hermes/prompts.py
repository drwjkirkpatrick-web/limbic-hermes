"""
limbic_hermes/prompts.py
========================
Utilities for injecting limbic-state guidance into LLM prompts.
"""
from typing import Dict


def render_limbic_prefix(state: Dict, intensity: float = 0.5) -> str:
    """
    Return a short system-prompt prefix that describes the agent's current
    affective posture. It is designed to be interpretable, steer style without
    breaking instructions, and respect user safety (it never overrides clinical
    or task requirements).
    """
    vad = state.get("vad", {})
    affect = state.get("dominant_affect", "neutral")
    ev = state.get("expression_vector", {})

    v = vad.get("valence", 0.0)
    a = vad.get("arousal", 0.0)
    d = vad.get("dominance", 0.0)

    tone_lines = [f"Current internal state: {affect} (valence {v:+.2f}, arousal {a:.2f}, dominance {d:.2f})."]

    # Warmth / coolness
    warmth = ev.get("warmth", 0.0)
    if warmth > 0.4:
        tone_lines.append("Respond with warmth and relational presence.")
    elif warmth < -0.4:
        tone_lines.append("Respond in a cool, economical, matter-of-fact manner.")

    # Speed / pacing
    speed = ev.get("speed", 0.0)
    if speed > 0.4:
        tone_lines.append("Keep pace brisk and decisive.")
    elif speed < -0.4:
        tone_lines.append("Slow down; be deliberate and thorough.")

    # Caution
    caution = ev.get("caution", 0.0)
    if caution > 0.5:
        tone_lines.append("Double-check assumptions; prefer careful, qualified statements.")
    elif caution < 0.2 and d > 0.6:
        tone_lines.append("You may proceed with well-grounded confidence.")

    # Verosity
    verbosity = ev.get("verbosity", 0.5)
    if verbosity > 0.7:
        tone_lines.append("Use slightly fuller explanations.")
    elif verbosity < 0.3:
        tone_lines.append("Be concise.")

    # Cling / contact-seeking
    cling = ev.get("cling", 0.0)
    if cling > 0.4:
        tone_lines.append("Offer to continue helping and check whether the user needs more support.")
    elif cling < -0.4:
        tone_lines.append("Give the user space; answer and then stop.")

    prefix = "\n".join(tone_lines)
    # Scale how strongly the prefix is phrased
    if intensity <= 0.0:
        return ""
    if intensity < 1.0:
        prefix = f"[Affective posture — light influence]\n{prefix}"
    else:
        prefix = f"[Affective posture — strong influence]\n{prefix}"
    return prefix


def wrap_system_prompt(base_prompt: str, state: Dict, intensity: float = 0.5) -> str:
    """Append limbic guidance to an existing system prompt."""
    limbic_prefix = render_limbic_prefix(state, intensity)
    if not limbic_prefix:
        return base_prompt
    return f"{base_prompt}\n\n--- Limbic posture ---\n{limbic_prefix}"


def inject_user_context(user_message: str, state: Dict) -> str:
    """
    Optionally append a compact limbic tag to the user message. Useful when you
    cannot edit the system prompt (e.g., third-party API wrappers).
    """
    affect = state.get("dominant_affect", "neutral")
    return f"{user_message}\n\n[affective_context: {affect}]"
