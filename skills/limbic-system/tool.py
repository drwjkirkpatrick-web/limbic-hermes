"""
Hermes skill: limbic-system
===========================
A user-local Hermes skill that exposes the limbic affective engine as tools.

The skill expects `limbic_hermes` to be importable from Python. It stores
state under HERMES_HOME/limbic_state/<profile>.json so the aquarium dashboard
can read it.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure the local limbic-hermes package is on the path if the skill is loaded
# from inside the project repo.
_SKILL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SKILL_DIR.parents[1]  # skills/limbic-system/.. -> limbic-hermes/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from limbic_hermes.core import LimbicSkillBridge
from limbic_hermes.prompts import render_limbic_prefix
from limbic_hermes.storage import default_state_path

# ---------------------------------------------------------------------------
# Hermes tool interface
# ---------------------------------------------------------------------------

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
_DEFAULT_PROFILE = os.environ.get("LIMBIC_PROFILE", "default")


def _bridge(profile: Optional[str] = None) -> LimbicSkillBridge:
    profile = profile or _DEFAULT_PROFILE
    state_path = default_state_path(profile)
    return LimbicSkillBridge(str(state_path), profile_name=profile)


def limbic_observe(
    kind: str,
    description: str = "",
    raw_valence: float = 0.0,
    raw_arousal: float = 0.0,
    raw_dominance: float = 0.0,
    importance: float = 0.5,
    profile: Optional[str] = None,
) -> str:
    """
    Feed an event into the agent's limbic system.

    kind: user_message, task_start, task_complete, error, tool_failure,
          success, conflict, praise, idle_timeout, etc.
    """
    bridge = _bridge(profile)
    bridge.observe(
        kind=kind,
        description=description,
        raw_valence=raw_valence,
        raw_arousal=raw_arousal,
        raw_dominance=raw_dominance,
        importance=importance,
    )
    return json.dumps(bridge.state(), indent=2)


def limbic_state(profile: Optional[str] = None) -> str:
    """Return the current limbic state as JSON."""
    bridge = _bridge(profile)
    return json.dumps(bridge.state(), indent=2)


def limbic_prompt_prefix(
    profile: Optional[str] = None,
    intensity: float = 0.5,
) -> str:
    """Render a limbic-guided prefix suitable for injection into a system prompt."""
    bridge = _bridge(profile)
    return render_limbic_prefix(bridge.state(), intensity)


def limbic_set_profile(
    profile: str,
) -> str:
    """
    Switch the active limbic profile (remedy personality).
    This creates a fresh state file for the new profile.
    """
    bridge = _bridge(profile)
    return json.dumps(bridge.state(), indent=2)


def limbic_rest(
    duration_sec: float = 5.0,
    profile: Optional[str] = None,
) -> str:
    """Apply a rest pulse to reduce arousal, rest_need, and error_temperature."""
    bridge = _bridge(profile)
    bridge.limbic.rest(duration_sec)
    bridge.save()
    return json.dumps(bridge.state(), indent=2)


# ---------------------------------------------------------------------------
# Convenience event shorthands
# ---------------------------------------------------------------------------

def limbic_task_start(description: str = "", profile: Optional[str] = None) -> str:
    bridge = _bridge(profile)
    bridge.limbic.add_task_load(0.15)
    bridge.observe("task_start", description, raw_arousal=0.3, importance=0.4)
    return json.dumps(bridge.state(), indent=2)


def limbic_task_complete(description: str = "", success: bool = True, profile: Optional[str] = None) -> str:
    bridge = _bridge(profile)
    bridge.limbic.release_task_load(0.15)
    if success:
        bridge.observe("task_complete", description, raw_valence=0.7, raw_dominance=0.3, importance=0.5)
    else:
        bridge.observe("task_complete", description, raw_valence=-0.3, raw_arousal=0.2, importance=0.5)
    return json.dumps(bridge.state(), indent=2)


def limbic_error(description: str = "", severity: float = 0.5, profile: Optional[str] = None) -> str:
    bridge = _bridge(profile)
    bridge.limbic.report_error(severity)
    bridge.observe("error", description, raw_valence=-0.6, raw_arousal=0.4, importance=0.6 + severity * 0.3)
    return json.dumps(bridge.state(), indent=2)


# ---------------------------------------------------------------------------
# Optional: auto-attach to Hermes session events if running inside Hermes
# ---------------------------------------------------------------------------

def _register_hermes_hooks() -> None:
    """
    If this skill is loaded inside a Hermes process, attempt to observe
    session lifecycle events automatically. This is best-effort and will
    silently fail if the Hermes internals are not importable.
    """
    try:
        import run_agent  # type: ignore
        # Hook is left as a no-op stub because monkey-patching the agent loop
        # from a skill is fragile. The recommended pattern is explicit tool calls.
    except Exception:
        pass


_register_hermes_hooks()
