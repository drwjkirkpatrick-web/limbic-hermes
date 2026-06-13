# Limbic Hermes: 31 Biochemistry-Reflecting Improvements

Each item below is a testable prompt. Implement in order, verifying after each batch.

## 1. Serotonergic baselines
**Prompt:** Add a `serotonin` state variable (0..1) to `LimbicSystem`. Default 0.5. When serotonin < 0.3, multiply threat_gain by 1.5 and recovery_factor by 0.7. Test that a low-serotonin profile shows amplified threat response and slower recovery.

## 2. Dopaminergic RPE
**Prompt:** Replace the heuristic RPE with a dopamine signal: `dopamine = observed_reward - expected_reward`, where expected_reward decays toward observed with a learnable rate. Expose `dopamine` in state. Test that unexpected rewards produce positive dopamine and omitted rewards produce negative dopamine.

## 3. Noradrenergic arousal
**Prompt:** Add a `norepinephrine` variable (0..1) with tonic baseline 0.2 and phasic bursts on salient events. NE amplifies arousal_delta by `(1 + ne)`. Test that high-salience events raise NE and arousal more than low-salience events.

## 4. GABAergic inhibition
**Prompt:** Add `GABA` (0..1) as an inhibitory modulator. GABA reduces absolute valence/arousal deltas by `(1 - gaba * 0.5)`. Test that high GABA profiles are less reactive to threats.

## 5. Glutamatergic excitation
**Prompt:** Add `glutamate` (0..1) as excitatory drive. Glutamate amplifies all outgoing appraisal deltas by `(1 + glu * 0.5)`. Test that high glutamate increases state change magnitude.

## 6. Acetylcholine attention
**Prompt:** Add `acetylcholine` (0..1) that switches attention mode: low ACh = scanning/broad attention (higher novelty bias), high ACh = focused attention (lower novelty bias, higher importance gating). Test mode-dependent thalamic gating.

## 7. Endocannabinoid decay
**Prompt:** Add `endocannabinoid` (eCB) that builds with arousal and retrogradely suppresses the next event's arousal_delta by `ecb * 0.4`. Test that high activation is followed by calming.

## 8. Oxytocin social bond
**Prompt:** Add `oxytocin` (0..1) that increases on positive social events (praise, user thanks) and boosts valence/safety while reducing threat reactivity. Test social event handling.

## 9. Vasopressin defense
**Prompt:** Add `vasopressin` (0..1) that rises on control/safety threats and increases dominance-oriented defensive arousal. Test control-challenge events.

## 10. Cortisol stress load
**Prompt:** Add `cortisol` (0..1) as delayed-decay HPA-axis stress. Errors/conflict raise cortisol; cortisol above 0.5 impairs hippocampal memory formation and slows recovery. Test delayed decay and memory effects.

## 11. Adrenaline surge
**Prompt:** Add `adrenaline` (0..1) for acute fight-or-flight on sudden high-threat events. Adrenaline causes a sharp arousal spike and brief dominance drop. Test sudden threat response.

## 12. Opioid analgesia/reward
**Prompt:** Add `opioid` (0..1) that reduces the negative saliency of adverse events and enhances positive valence. Test pain-buffering effect.

## 13. Histamine wakefulness
**Prompt:** Add `histamine` (0..1) as a wakefulness/arousal floor. Low histamine allows rest to lower arousal faster; high histamine raises baseline arousal. Test wakefulness modulation.

## 14. Melatonin circadian
**Prompt:** Add `melatonin` (0..1) driven by simulated time-of-day. High melatonin lowers arousal ceiling and increases rest_need recovery. Test circadian effects.

## 15. Neurotransmitter balance ratios
**Prompt:** Expose `dopamine_serotonin_ratio` and `gaba_glutamate_ratio` in state output. Test that these update correctly after events.

## 16. Receptor desensitization
**Prompt:** Add receptor desensitization: sustained high dopamine or adrenaline reduces receptor sensitivity temporarily, diminishing future response. Test habituation to repeated rewards/threats.

## 17. Synaptic plasticity / LTP
**Prompt:** Add learnable event-outcome weights in hippocampus: repeated event->outcome pairs strengthen association, biasing future appraisals. Test learned expectations.

## 18. Prefrontal-amygdala top-down regulation
**Prompt:** Use dominance as a top-down regulator: high dominance suppresses amygdala threat response by up to 30%. Test dominance-dependent threat attenuation.

## 19. Interoceptive signal
**Prompt:** Add `heart_rate_variability` and `respiration_rate` signals that modulate arousal and perceived safety. Test interoceptive influence.

## 20. Neuroinflammatory cytokine load
**Prompt:** Add `cytokine_load` (0..1) that rises with chronic stress/cortisol and increases irritability (negative valence drift) and slows recovery. Test chronic stress simulation.

## 21. BDNF neurotrophic factor
**Prompt:** Add `bdnf` (0..1) that modulates learning rate and resilience. High BDNF speeds LTP and recovery; low BDNF does the opposite. Test resilience modulation.

## 22. Nitric oxide vasodilation signal
**Prompt:** Add `nitric_oxide` as a local diffusion signal that spreads activation state to nearby variables. Test spreading activation.

## 23. Trace amines (PEA, tyramine)
**Prompt:** Add `phenylethylamine` and `tyramine` as modulators of reward salience and alertness. Test combined effects with dopamine.

## 24. Neuropeptide S alertness
**Prompt:** Add `neuropeptide_s` that produces brief alertness boosts on novel stimuli. Test novelty response.

## 25. Orexin wake/arousal
**Prompt:** Add `orexin` stabilizing wakefulness and motivation. Low orexin increases rest_need and lowers reward gain. Test sleepiness modulation.

## 26. Substance P pain salience
**Prompt:** Add `substance_p` that rises with stress/cortisol and amplifies negative valence. Test stress-pain amplification.

## 27. Glycine inhibition refinement
**Prompt:** Add `glycine` inhibitory tone that refines expressive output, reducing jitter/impulsivity. Test expression smoothing.

## 28. Neurotransmitter depletion/recovery
**Prompt:** Model finite pools for dopamine, serotonin, norepinephrine. Heavy use depletes; rest/recovery rebuilds. Test depletion and recovery cycles.

## 29. Entrainment to user affect
**Prompt:** Add `user_affect_mirror` that shifts baseline valence/arousal toward detected user affect at a configurable rate. Test convergence.

## 30. Locus coeruleus phasic bursts
**Prompt:** Model LC-NE phasic bursts: probability of burst increases with surprise (|RPE|), causing transient attention/arousal spikes. Test surprise-locked bursts.

## 31. Default mode network suppression
**Prompt:** Add `dmn_activity` (0..1) that is high at rest and suppressed by task_load. High DMN increases self-referential thought (valence drift toward baseline). Test task-load suppression.

## Implementation notes
- Keep all variables clamped [0,1] or [-1,1] as appropriate.
- Maintain backward compatibility: existing API (`observe_event`, `get_state`, etc.) must continue to work.
- Add tests for each new module in `tests/test_limbic.py`.
- Update `demo.py` to optionally show neurotransmitter panel.
