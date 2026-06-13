from limbic_hermes.core import LimbicSystem, VAD, DriveState, get_remedy_profile
from limbic_hermes.profiles import full_remedy_library


def test_vad_clamp():
    v = VAD(valence=2.0, arousal=-1.0, dominance=0.3).clamp()
    assert v.valence == 1.0
    assert v.arousal == 0.0
    assert v.dominance == 0.3


def test_default_profile():
    p = get_remedy_profile("default")
    assert p.name == "default"


def test_remedy_profiles():
    for name in ["pulsatilla", "bryonia", "tarantula", "calcarea"]:
        p = get_remedy_profile(name)
        assert p.name == name


def test_event_changes_state():
    limbic = LimbicSystem(profile_name="default")
    before = limbic.vad.valence
    limbic.observe_event("success", importance=1.0, raw_valence=0.9)
    assert limbic.vad.valence > before


def test_error_heat():
    limbic = LimbicSystem(profile_name="bryonia")
    limbic.report_error(severity=1.0)
    assert limbic.drive.error_temperature > 0.5


def test_rest_lowers_arousal():
    limbic = LimbicSystem(profile_name="tarantula")
    limbic.observe_event("tool_failure", raw_valence=-0.8, raw_arousal=0.8)
    arousal_before = limbic.vad.arousal
    limbic.rest(duration_sec=5.0)
    assert limbic.vad.arousal < arousal_before


def test_episodic_capacity():
    limbic = LimbicSystem(profile_name="default", episodic_capacity=3)
    for i in range(5):
        limbic.observe_event("user_message", f"msg {i}")
    assert len(limbic.episodic_buffer) <= 3


def test_state_serializable():
    limbic = LimbicSystem(profile_name="pulsatilla")
    limbic.observe_event("success", raw_valence=0.5)
    state = limbic.get_state()
    assert "vad" in state
    assert "drive" in state
    assert "dominant_affect" in state
    assert "expression_vector" in state


def test_bridge(tmp_path):
    from limbic_hermes.core import LimbicSkillBridge
    path = tmp_path / "limbic.json"
    bridge = LimbicSkillBridge(str(path), profile_name="calcarea")
    bridge.observe("task_start", raw_arousal=0.3)
    state = bridge.state()
    assert state["profile"] == "calcarea"
    # Reload
    bridge2 = LimbicSkillBridge(str(path))
    assert bridge2.limbic.profile.name == "calcarea"


if __name__ == "__main__":
    test_vad_clamp()
    test_default_profile()
    test_remedy_profiles()
    test_event_changes_state()
    test_error_heat()
    test_rest_lowers_arousal()
    test_episodic_capacity()
    test_state_serializable()
    print("All inline tests passed.")
