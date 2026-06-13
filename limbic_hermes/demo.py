"""CLI demo / REPL for the Hermes limbic system."""
import argparse
import json
import time

from limbic_hermes.core import LimbicSystem
from limbic_hermes.profiles import full_remedy_library


def run_simulation(profile: str, steps: int, step_sec: float = 1.0) -> None:
    limbic = LimbicSystem(profile_name=profile)
    print(f"\nProfile: {profile}")
    print("Available remedies:", sorted(full_remedy_library().keys()))

    scenario = [
        ("user_message", "User asks a warm clinical question", 0.3, 0.2, 0.0, 0.5),
        ("task_start", "Starting a multi-tool research task", 0.0, 0.4, 0.1, 0.4),
        ("tool_failure", "A web search timed out", -0.5, 0.5, -0.3, 0.7),
        ("task_complete", "Task finished successfully", 0.7, 0.1, 0.4, 0.6),
        ("conflict", "User corrected an earlier answer", -0.4, 0.3, -0.2, 0.6),
        ("idle_timeout", "No new messages for a while", -0.1, -0.2, 0.0, 0.2),
        ("praise", "User says 'thank you'", 0.6, 0.2, 0.2, 0.5),
    ]

    for i in range(steps):
        kind, desc, v, a, d, imp = scenario[i % len(scenario)]
        now = time.time() + i * step_sec
        appraisal = limbic.observe_event(kind, desc, v, a, d, imp, now=now)
        state = limbic.get_state()
        neuro = state["neurochemistry"]
        print(f"\n[{i}] {kind}: {desc}")
        print("  appraisal ->", vars(appraisal))
        print("  VAD       ->", state["vad"])
        print("  affect    ->", state["dominant_affect"])
        print("  RPE       ->", state["reward_prediction_error"])
        print("  allostatic->", state["allostatic_load"])
        print("  key NTs   -> DA=%.2f 5HT=%.2f NE=%.2f CORT=%.2f OXY=%.2f" % (
            neuro["dopamine"], neuro["serotonin"], neuro["norepinephrine"],
            neuro["cortisol"], neuro["oxytocin"]))
        print("  pools     -> DA_pool=%.2f NE_pool=%.2f 5HT_pool=%.2f" % (
            neuro["dopamine_pool"], neuro["norepinephrine_pool"], neuro["serotonin_pool"]))
        # simulate real time
        if step_sec > 0:
            time.sleep(step_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Limbic System demo")
    parser.add_argument("--profile", default="pulsatilla", help="Remedy temperament profile")
    parser.add_argument("--steps", type=int, default=7, help="Number of scenario steps")
    parser.add_argument("--pause", type=float, default=0.5, help="Seconds between steps")
    args = parser.parse_args()
    run_simulation(args.profile, args.steps, args.pause)


if __name__ == "__main__":
    main()
